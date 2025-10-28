#!/usr/bin/env python3
"""
Simple CLI test for entry point verification
"""

import click


@click.group()
def cli():
    """Simple test CLI"""
    pass


@cli.command()
def hello():
    """Say hello"""
    click.echo("Hello from HMM Futures Analysis!")


@cli.command()
def version():
    """Show version"""
    click.echo("HMM Futures Analysis v0.1.0")


if __name__ == "__main__":
    cli()
