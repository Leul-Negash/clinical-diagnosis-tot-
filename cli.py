#!/usr/bin/env python3
"""Rich CLI chatbot for the Tree-of-Thoughts diagnosis researcher.

Renders the tree of thoughts *as it grows and prunes*, then prints the
synthesised differential. Run `python cli.py --help` for options.

Examples
--------
    # Offline demo, no API key needed:
    python cli.py --provider mock

    # Live with the best free tier (export GROQ_API_KEY first):
    python cli.py --provider groq

    # One-shot question + side-by-side IO/CoT/ToT comparison:
    python cli.py --provider mock --compare \\
        --case "32F sudden pleuritic chest pain and breathlessness after a long flight"
"""

from __future__ import annotations

import argparse
import sys

from tot.env import load_dotenv

load_dotenv()  # pick up GROQ_API_KEY / GEMINI_API_KEY from a local .env file

from tot import (
    LLMConfig,
    SearchStrategy,
    GenerationStrategy,
    EvaluationStrategy,
    ToTConfig,
    ToTEngine,
    build_client,
    compare,
)

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.tree import Tree as RichTree
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    _RICH = True
except ImportError:  # pragma: no cover - graceful degradation
    _RICH = False

DISCLAIMER = (
    "[bold yellow]Educational research tool — not medical advice.[/] "
    "It explores diagnostic hypotheses to demonstrate Tree-of-Thoughts "
    "reasoning. Always consult a qualified clinician."
)

_STATUS_STYLE = {
    "active": ("white", "•"),
    "expanded": ("cyan", "▸"),
    "solved": ("bold green", "✓"),
    "pruned": ("red dim", "✗"),
}


def _node_label(node: dict) -> str:
    style, glyph = _STATUS_STYLE.get(node["status"], ("white", "•"))
    score = node["score"]
    label = node["label"] or "—"
    text = node["content"]
    if node["status"] == "pruned":
        text = f"[strike]{text}[/strike]"
    return f"[{style}]{glyph} {text}[/]  [dim]({score:.2f} · {label})[/dim]"


def build_rich_tree(root: dict, console_label: str = "🧠 Case") -> "RichTree":
    rtree = RichTree(f"[bold]{console_label}[/bold]")

    def walk(node: dict, parent):
        for child in node["children"]:
            branch = parent.add(_node_label(child))
            walk(child, branch)

    walk(root, rtree)
    return rtree


def run_one(engine_cfg: ToTConfig, case: str, console) -> None:
    if not _RICH:
        result = ToTEngine(engine_cfg).run(case)
        _plain_output(result)
        return

    engine_holder: dict = {}

    with Live(console=console, refresh_per_second=12, screen=False) as live:
        def on_event(event):
            tree = engine_holder.get("tree")
            if tree is None:
                return
            rt = build_rich_tree(tree.to_render_dict())
            phase = event["type"].upper()
            live.update(Panel(rt, title=f"[cyan]Deliberate search — {phase}[/cyan]",
                              border_style="cyan"))

        engine = ToTEngine(engine_cfg, on_event=on_event)
        result = _run_with_live(engine, case, engine_holder)

    _rich_output(result, console)


def _run_with_live(engine: ToTEngine, case: str, holder: dict):
    """Run the engine, capturing the growing tree the first time an event fires.

    The engine owns its `Tree` internally; we recover it from any event's node
    by walking up to the root, then expose it to the live-render callback.
    """
    original = engine.on_event

    def hook(event):
        node = event.get("node")
        if node is not None and "tree" not in holder:
            root = node
            while root.parent is not None:
                root = root.parent
            holder["tree"] = _RootView(root)
        original(event)

    engine.on_event = hook
    return engine.run(case)


class _RootView:
    """Minimal adapter so the live callback can serialise the growing tree."""

    def __init__(self, root):
        self.root = root

    def to_render_dict(self):
        def encode(n):
            return {
                "id": n.id, "content": n.content, "depth": n.depth,
                "score": round(n.score, 3), "label": n.label,
                "status": n.status.value, "rationale": n.rationale,
                "eval_notes": n.eval_notes,
                "children": [encode(c) for c in n.children],
            }
        return encode(self.root)


def _rich_output(result, console) -> None:
    console.print(build_rich_tree(result.render_tree()))

    t = result.telemetry
    table = Table(title="Search telemetry", show_header=False, border_style="dim")
    table.add_row("LLM calls", str(t["llm_calls"]))
    table.add_row("Thoughts generated", str(t["thoughts_generated"]))
    table.add_row("Depth reached", str(t["reached_depth"]))
    table.add_row("Stopped because", t["stopped_reason"])
    table.add_row("Elapsed", f"{result.elapsed_s:.1f}s")
    console.print(table)

    console.print(Panel(Markdown(result.summary),
                        title="[green]Differential diagnosis (research summary)[/green]",
                        border_style="green"))


def _plain_output(result) -> None:
    print("\n=== Differential (research summary) ===")
    print(result.summary)
    print("\nTelemetry:", result.telemetry)


def make_config(args) -> ToTConfig:
    llm = LLMConfig().apply_preset(args.provider)
    if args.model:
        llm.model = args.model
    cfg = ToTConfig(
        max_depth=args.depth,
        breadth=args.breadth,
        n_generate=args.k,
        prune_threshold=args.prune,
        n_evaluate_votes=args.votes,
        search_strategy=SearchStrategy(args.search),
        generation_strategy=GenerationStrategy(args.generate),
        evaluation_strategy=EvaluationStrategy(args.evaluate),
        llm=llm,
    )
    return cfg.validate()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Tree-of-Thoughts diagnosis researcher")
    p.add_argument("--provider", default="mock",
                   choices=["mock", "groq", "gemini", "openai", "ollama"])
    p.add_argument("--model", default=None, help="override the provider's model")
    p.add_argument("--depth", type=int, default=2, help="T: max search depth")
    p.add_argument("--breadth", type=int, default=3, help="b: states kept per level")
    p.add_argument("--k", type=int, default=4, help="k: hypotheses generated per state")
    p.add_argument("--prune", type=float, default=0.3, help="v_th: prune threshold")
    p.add_argument("--votes", type=int, default=1, help="evaluator samples to aggregate")
    p.add_argument("--search", default="bfs", choices=["bfs", "dfs"])
    p.add_argument("--generate", default="propose", choices=["propose", "sample"])
    p.add_argument("--evaluate", default="value", choices=["value", "vote"])
    p.add_argument("--case", default=None, help="run one case then exit")
    p.add_argument("--compare", action="store_true",
                   help="show IO vs CoT vs ToT for the case")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = make_config(args)
    console = Console() if _RICH else None

    if _RICH:
        console.print(Panel(DISCLAIMER, border_style="yellow",
                            title="Clinical ToT Researcher"))
        console.print(f"[dim]provider={cfg.llm.provider} model={cfg.llm.model} "
                      f"search={cfg.search_strategy.value} T={cfg.max_depth} "
                      f"b={cfg.breadth} k={cfg.n_generate}[/dim]\n")

    # Validate key presence early for live providers.
    if cfg.llm.provider != "mock" and not cfg.llm.api_key:
        msg = (f"No API key in ${cfg.llm.api_key_env}. "
               f"Export it or use --provider mock.")
        (console.print if _RICH else print)(f"[red]{msg}[/red]" if _RICH else msg)
        return 2

    def handle(case: str):
        if args.compare:
            _show_comparison(case, cfg, console)
        else:
            run_one(cfg, case, console)

    if args.case:
        handle(args.case)
        return 0

    # Interactive REPL.
    while True:
        try:
            case = (Prompt.ask("\n[bold cyan]Describe the case[/bold cyan] "
                               "(or 'quit')") if _RICH
                    else input("\nDescribe the case (or 'quit'): "))
        except (EOFError, KeyboardInterrupt):
            break
        if case.strip().lower() in {"quit", "exit", "q"}:
            break
        if case.strip():
            handle(case.strip())
    return 0


def _show_comparison(case: str, cfg: ToTConfig, console) -> None:
    client = build_client(cfg.llm)
    cmp = compare(case, cfg, client=client)
    if not _RICH:
        print("IO :", cmp.io.answer)
        print("CoT:", cmp.cot.answer)
        print("ToT:", cmp.tot.summary)
        return
    table = Table(title="IO vs CoT vs ToT (same case)", border_style="magenta")
    table.add_column("Method"); table.add_column("LLM calls", justify="right")
    table.add_column("Answer")
    table.add_row("IO", str(cmp.io.llm_calls), cmp.io.answer)
    table.add_row("CoT", str(cmp.cot.llm_calls), cmp.cot.answer)
    table.add_row("ToT", str(cmp.tot.telemetry["llm_calls"]),
                  cmp.tot.summary.split("\n")[0] + " …")
    console.print(table)
    console.print(Panel(build_rich_tree(cmp.tot.render_tree()),
                        title="ToT explored this tree", border_style="green"))


if __name__ == "__main__":
    sys.exit(main())
