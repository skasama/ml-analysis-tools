import argparse
import sys

COMMANDS = {
    "classification": "analysis.cli.plot_classification",
    "classification-dependencies": "analysis.cli.plot_classification_dependencies",
    "direction": "analysis.cli.plot_direction_regression",
    "energy": "analysis.cli.plot_energy_regression",
    "loss": "analysis.cli.plot_loss",
    "resolution-dependencies": "analysis.cli.plot_resolution_dependencies",
    "vertex": "analysis.cli.plot_vertex_regression",
    "event-display": "analysis.cli.save_event_display_3d",
    "inputs": "analysis.cli.inputs",
    "compare": "analysis.cli.compare",
    "compare-dependencies": "analysis.cli.compare_dependencies",
    "compare-resolution-dependencies": "analysis.cli.compare_resolution_dependencies",
}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="analysis",
        description="Generate analysis plots and event displays.",
        add_help=False,
    )
    parser.add_argument(
        "command",
        choices=COMMANDS,
        help="Analysis command to run.",
    )
    return parser


def main():
    """Dispatch an analysis command to its task-specific CLI."""
    parser = build_parser()
    if len(sys.argv) < 2:
        parser.print_help()
        return
    if sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return

    args, remainder = parser.parse_known_args()
    module = __import__(COMMANDS[args.command], fromlist=["main"])
    sys.argv = [f"analysis {args.command}", *remainder]
    module.main()


if __name__ == "__main__":
    main()
