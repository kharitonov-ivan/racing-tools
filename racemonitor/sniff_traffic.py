#!/usr/bin/env python3
"""Traffic sniffer - Capture raw data from racing timing system."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests>=2.31.0",
# ]
# ///

import argparse
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Sniff and save traffic from racing timing system"
    )
    parser.add_argument(
        "--url",
        default="http://10.10.31.20:50000/",
        help="URL to sniff traffic from"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: traffic_YYYYMMDD_HHMMSS.txt)"
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        help="Duration in seconds to capture (default: run until Ctrl+C)"
    )

    args = parser.parse_args()

    # Generate output filename if not provided
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"traffic_{timestamp}.txt")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Sniffing traffic from: {args.url}")
    print(f"Saving to: {output_path}")
    print("Press Ctrl+C to stop\n")

    import requests

    try:
        # Connect to the stream
        response = requests.get(args.url, stream=True, timeout=None)
        response.raise_for_status()

        print("✓ Connected! Capturing traffic...\n")

        line_count = 0
        start_time = datetime.now()

        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"# Traffic capture started: {start_time.isoformat()}\n")
            f.write(f"# Source: {args.url}\n")
            f.write(f"# Format: Raw line-by-line capture\n")
            f.write("#' + '='*76 + '\n\n")

            # Read and save lines
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    line_count += 1
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                    # Write with timestamp
                    f.write(f"[{timestamp}] {line}\n")
                    f.flush()

                    # Print progress every 100 lines
                    if line_count % 100 == 0:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        rate = line_count / elapsed if elapsed > 0 else 0
                        print(f"\rCaptured: {line_count} lines ({rate:.1f} lines/sec)", end="", flush=True)

                    # Check duration limit
                    if args.duration:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        if elapsed >= args.duration:
                            print(f"\n\nDuration limit reached ({args.duration}s)")
                            break

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        print(f"\n\n✓ Capture complete!")
        print(f"  Total lines: {line_count}")
        print(f"  Duration: {elapsed:.1f} seconds")
        print(f"  Output: {output_path}")
        print(f"  Rate: {line_count/elapsed:.1f} lines/second")

    except KeyboardInterrupt:
        print("\n\n✓ Capture stopped by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
