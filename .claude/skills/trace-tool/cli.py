#!/usr/bin/env python3
"""
Trace Tool CLI

PyTorch Profiler Trace trim and diff tool:
- trim: Extract a subset of operators from a large trace
- diff: Compare two traces (sequence, call tree, duration, implementation)
"""

import sys
from pathlib import Path

# Add skill directory to path for imports
skill_dir = Path(__file__).parent
if str(skill_dir) not in sys.path:
    sys.path.insert(0, str(skill_dir))

import click

from trim import trim
from diff import diff
from merge import merge
from blocks import blocks


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Trace Tool - PyTorch Profiler Trace trim, diff, merge, and blocks"""
    pass


cli.add_command(trim)
cli.add_command(diff)
cli.add_command(merge)
cli.add_command(blocks)


def main():
    cli()


if __name__ == "__main__":
    main()
