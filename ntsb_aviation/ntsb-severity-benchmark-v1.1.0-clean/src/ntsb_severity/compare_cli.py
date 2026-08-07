from __future__ import annotations

import argparse
import json
from .audit import compare_sheets


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("sheet_a")
    parser.add_argument("sheet_b")
    args = parser.parse_args(argv)
    print(json.dumps(compare_sheets(args.path, args.sheet_a, args.sheet_b)))


if __name__ == "__main__":
    main()
