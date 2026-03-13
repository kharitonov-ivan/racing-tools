#!/usr/bin/env python3
import argparse
import subprocess
import os

IMAGE = "racing-tools"
DOCKER_RUN = [
    "docker", "run", "-it", "--rm",
    "--gpus", "all",
    "-e", f"DISPLAY={os.environ.get('DISPLAY', ':0')}",
    "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
    "-v", "./data:/app/data",
]


def build(_args: argparse.Namespace) -> None:
    subprocess.run(["docker", "build", "-t", IMAGE, "."], check=True)


def attach(_args: argparse.Namespace) -> None:
    subprocess.run([*DOCKER_RUN, "--entrypoint", "bash", IMAGE], check=True)


def run(args: argparse.Namespace) -> None:
    subprocess.run([*DOCKER_RUN, IMAGE, *args.script_args], check=True)


def main() -> None:
    p = argparse.ArgumentParser(prog="./x", description="Docker helper for racing-tools")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("build", aliases=["b"], help="Build the Docker image")
    sub.add_parser("attach", aliases=["a"], help="Attach interactive shell")

    run_p = sub.add_parser("run", aliases=["r"], help="Run: uv run <script> [args...]")
    run_p.add_argument("script_args", nargs=argparse.REMAINDER, help="Script and its arguments")

    args = p.parse_args()
    if args.command in ("build", "b"):
        build(args)
    elif args.command in ("attach", "a"):
        attach(args)
    elif args.command in ("run", "r"):
        run(args)


if __name__ == "__main__":
    main()
