#!/usr/bin/env python3
"""Analyze captured traffic files."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
# ]
# ///

import argparse
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze captured traffic files"
    )
    parser.add_argument(
        "file",
        help="Traffic file to analyze"
    )
    parser.add_argument(
        "--types", "-t",
        action="store_true",
        help="Show message type distribution"
    )
    parser.add_argument(
        "--unique", "-u",
        action="store_true",
        help="Show unique message types"
    )

    args = parser.parse_args()

    traffic_file = Path(args.file)
    if not traffic_file.exists():
        print(f"Error: File not found: {traffic_file}")
        return

    print(f"Analyzing: {traffic_file}\n")

    message_types = Counter()
    total_lines = 0

    with open(traffic_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            total_lines += 1

            # Extract message type (format: [timestamp] $TYPE,...)
            if '] $' in line:
                try:
                    parts = line.split('] $', 1)
                    if len(parts) > 1:
                        msg = parts[1].split(',')[0]
                        message_types[msg] += 1
                except:
                    pass

    print(f"Total lines: {total_lines}")
    print(f"Message types found: {len(message_types)}\n")

    if args.unique or args.types:
        print("Message type distribution:")
        print("-" * 50)
        for msg_type, count in message_types.most_common():
            print(f"  ${msg_type:<15} {count:>6} ({count/total_lines*100:>5.1f}%)")
    else:
        print("\nUse --types to see message distribution")
        print("Use --unique to see unique message types")


if __name__ == "__main__":
    main()
