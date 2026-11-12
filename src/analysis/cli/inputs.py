import argparse
from pathlib import Path

from analysis.io import list_input_variables


def build_parser():
    parser = argparse.ArgumentParser(
        description="List selectable analysis input files and variables."
    )
    parser.add_argument("--run-dir", type=Path, default=Path("."))
    return parser


def main():
    args = build_parser().parse_args()
    variables = list_input_variables(args.run_dir)
    if not variables:
        print(f"No supported input files found below {args.run_dir}.")
        return
    print("FILE\tVARIABLE\tSHAPE\tDTYPE")
    for item in variables:
        print(
            f"{item.path.relative_to(args.run_dir)}\t"
            f"{item.variable or '-'}\t{item.shape}\t{item.dtype}"
        )


if __name__ == "__main__":
    main()
