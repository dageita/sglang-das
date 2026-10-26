#!/usr/bin/env python3
"""Print the final PREFILL_GRAPH_STATS snapshot for each process and device."""

import argparse
import json
import re
import sys


PATTERN = re.compile(r"PREFILL_GRAPH_STATS\s+(\{.*\})")


def lines(paths):
    if not paths:
        yield from sys.stdin
        return
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as file:
            yield from file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="*", help="server logs; stdin if omitted")
    args = parser.parse_args()
    latest = {}
    invalid = 0
    for line in lines(args.logs):
        match = PATTERN.search(line)
        if not match:
            continue
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            invalid += 1
            continue
        latest[(payload.get("pid"), payload.get("device"))] = payload
    if not latest:
        parser.error("no valid PREFILL_GRAPH_STATS records found")
    for key in sorted(latest, key=lambda item: (str(item[0]), str(item[1]))):
        print(json.dumps(latest[key], sort_keys=True))
    if invalid:
        print(f"warning: ignored {invalid} malformed records", file=sys.stderr)


if __name__ == "__main__":
    main()
