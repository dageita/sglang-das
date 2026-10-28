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
    parser.add_argument("--check-adaptive", action="store_true",
                        help="fail if any rank lacks capture/replay or a numerical check failed")
    args = parser.parse_args()
    latest = {}
    invalid = 0
    numerical_failures = 0
    for line in lines(args.logs):
        if 'PREFILL_BUCKET_DISABLED' in line or (
            'PREFILL_ISLAND_NUMERICS' in line and '"passed": false' in line
        ):
            numerical_failures += 1
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
    if args.check_adaptive:
        failed = numerical_failures > 0
        for payload in latest.values():
            graph = payload.get('graph_layers', {})
            failed |= not (graph.get('capture_count', 0) > 0
                           and graph.get('steady_replays', 0) > 0)
            failed |= graph.get('padding_failures', 0) > 0
            failed |= graph.get('padding_checks', 0) == 0
        if len({p.get('device') for p in latest.values()}) < 4:
            print('warning: fewer than four devices; confirm the expected TP size', file=sys.stderr)
        if failed:
            parser.exit(1, 'adaptive check FAILED: missing replay or numerical failure; inspect per-rank logs\n')
        print('adaptive check PASSED (logged snapshots; counters include warmup)', file=sys.stderr)


if __name__ == "__main__":
    main()
