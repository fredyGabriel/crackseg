"""CLI commands for health check system."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import click

from crackseg.dataclasses import asdict

from ..orchestration import HealthOrchestrator, ServiceRegistry


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool = False) -> None:
    """CrackSeg Health Check System CLI."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option("--output", "-o", type=click.Path(), help="Save report to file")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "dashboard"]),
    default="json",
    help="Output format",
)
def check(output: str | None, output_format: str) -> None:
    """Run health check for all services."""

    async def run_check() -> None:
        service_registry = ServiceRegistry()
        services = service_registry.load_configuration()

        orchestrator = HealthOrchestrator()
        report = await orchestrator.check_all_services(services)

        if output_format == "dashboard":
            data: dict[str, Any] = orchestrator.generate_dashboard_data()
            output_data = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            # Convert report to JSON
            output_data = json.dumps(
                asdict(report), indent=2, default=str, ensure_ascii=False
            )

        if output:
            with open(output, "w") as f:
                f.write(output_data)
            click.echo(f"Report saved to {output}")
        else:
            click.echo(output_data)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_check())


@cli.command()
@click.option("--interval", "-i", default=30, help="Check interval in seconds")
@click.option(
    "--output-dir", "-d", type=click.Path(), help="Directory for output files"
)
def monitor(interval: int, output_dir: str | None) -> None:
    """Start continuous health monitoring."""

    async def run_monitor() -> None:
        service_registry = ServiceRegistry()
        service_registry.load_configuration()

        orchestrator = HealthOrchestrator()

        from ..orchestration import ContinuousMonitor

        monitor = ContinuousMonitor(orchestrator, service_registry)

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

        try:
            await monitor.start_monitoring(interval)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            click.echo("Monitoring stopped")

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_monitor())


@cli.command()
def dashboard() -> None:
    """Generate dashboard data for monitoring UI."""

    async def run_dashboard() -> None:
        service_registry = ServiceRegistry()
        services = service_registry.load_configuration()

        orchestrator = HealthOrchestrator()
        await orchestrator.check_all_services(services)

        data: dict[str, Any] = orchestrator.generate_dashboard_data()
        click.echo(json.dumps(data, indent=2, ensure_ascii=False))

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_dashboard())
