#!/usr/bin/env python3
"""Cache management CLI tool for petroleum RAG benchmark.

Provides commands for viewing cache statistics, clearing caches,
and analyzing cache performance.
"""

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

# Add parent directory to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from utils.cache import get_embedding_cache, get_llm_cache

console = Console()


@click.group()
def cli():
    """Manage caching for petroleum RAG benchmark system."""
    pass


@cli.command()
def stats():
    """Show cache statistics."""
    try:
        embedding_cache = get_embedding_cache()
        llm_cache = get_llm_cache()

        # Create table for embedding cache
        emb_table = Table(title="Embedding Cache Statistics", show_header=True)
        emb_table.add_column("Metric", style="cyan")
        emb_table.add_column("Value", style="magenta")

        emb_stats = embedding_cache.get_stats()
        emb_size = embedding_cache.get_size()

        emb_table.add_row("Enabled", str(emb_stats["enabled"]))
        emb_table.add_row("Cache Directory", str(emb_stats["cache_dir"]))
        emb_table.add_row("Total Requests", str(emb_stats["total_requests"]))
        emb_table.add_row("Cache Hits", str(emb_stats["hits"]))
        emb_table.add_row("Cache Misses", str(emb_stats["misses"]))
        emb_table.add_row("Hit Rate", f"{emb_stats['hit_rate']:.1%}")
        emb_table.add_row("Memory Hits", str(emb_stats["memory_hits"]))
        emb_table.add_row("Disk Hits", str(emb_stats["disk_hits"]))
        emb_table.add_row("Sets (Writes)", str(emb_stats["sets"]))
        emb_table.add_row("Evictions", str(emb_stats["evictions"]))
        emb_table.add_row("Memory Items", f"{emb_size['memory_items']:,}")
        emb_table.add_row("Disk Items", f"{emb_size['disk_items']:,}")
        emb_table.add_row("Disk Size", f"{emb_size['disk_mb']:.2f} MB")

        # Create table for LLM cache
        llm_table = Table(title="LLM Cache Statistics", show_header=True)
        llm_table.add_column("Metric", style="cyan")
        llm_table.add_column("Value", style="magenta")

        llm_stats = llm_cache.get_stats()
        llm_size = llm_cache.get_size()

        llm_table.add_row("Enabled", str(llm_stats["enabled"]))
        llm_table.add_row("Cache Directory", str(llm_stats["cache_dir"]))
        llm_table.add_row("Total Requests", str(llm_stats["total_requests"]))
        llm_table.add_row("Cache Hits", str(llm_stats["hits"]))
        llm_table.add_row("Cache Misses", str(llm_stats["misses"]))
        llm_table.add_row("Hit Rate", f"{llm_stats['hit_rate']:.1%}")
        llm_table.add_row("Memory Hits", str(llm_stats["memory_hits"]))
        llm_table.add_row("Disk Hits", str(llm_stats["disk_hits"]))
        llm_table.add_row("Sets (Writes)", str(llm_stats["sets"]))
        llm_table.add_row("Evictions", str(llm_stats["evictions"]))
        llm_table.add_row("Memory Items", f"{llm_size['memory_items']:,}")
        llm_table.add_row("Disk Items", f"{llm_size['disk_items']:,}")
        llm_table.add_row("Disk Size", f"{llm_size['disk_mb']:.2f} MB")

        console.print(emb_table)
        console.print()
        console.print(llm_table)

        # Summary
        total_disk_mb = emb_size["disk_mb"] + llm_size["disk_mb"]
        total_items = emb_size["disk_items"] + llm_size["disk_items"]

        console.print()
        console.print(
            f"[bold green]Total cache size:[/] {total_disk_mb:.2f} MB ({total_items:,} items)"
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}", style="red")
        sys.exit(1)


@cli.command()
@click.option(
    "--cache",
    type=click.Choice(["embedding", "llm", "all"], case_sensitive=False),
    default="all",
    help="Which cache to clear",
)
@click.confirmation_option(prompt="Are you sure you want to clear the cache?")
def clear(cache: str):
    """Clear cache(s)."""

    async def _clear():
        try:
            if cache in ["embedding", "all"]:
                embedding_cache = get_embedding_cache()
                await embedding_cache.clear()
                console.print("[green]Embedding cache cleared[/]")

            if cache in ["llm", "all"]:
                llm_cache = get_llm_cache()
                await llm_cache.clear()
                console.print("[green]LLM cache cleared[/]")

            console.print("[bold green]Cache clearing complete[/]")

        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}", style="red")
            sys.exit(1)

    asyncio.run(_clear())


@cli.command()
def config():
    """Show cache configuration."""
    try:
        cache_config = settings.get_cache_config()

        table = Table(title="Cache Configuration", show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Caching Enabled", str(cache_config["enabled"]))
        table.add_row("Base Directory", str(cache_config["base_dir"]))
        table.add_row("Embedding Directory", str(cache_config["embedding_dir"]))
        table.add_row("LLM Directory", str(cache_config["llm_dir"]))
        table.add_row("Max Memory Items", f"{cache_config['max_memory_items']:,}")
        table.add_row("Embedding Cache Enabled", str(cache_config["embedding_enabled"]))
        table.add_row("LLM Cache Enabled", str(cache_config["llm_enabled"]))

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}", style="red")
        sys.exit(1)


@cli.command()
def savings():
    """Estimate API cost savings from caching."""
    try:
        embedding_cache = get_embedding_cache()
        llm_cache = get_llm_cache()

        emb_stats = embedding_cache.get_stats()
        llm_stats = llm_cache.get_stats()

        # Cost estimates (approximate, as of 2025)
        # OpenAI text-embedding-3-small: $0.02 per 1M tokens (avg ~750 tokens per request)
        # Claude Sonnet 4: ~$3.00 per 1M input tokens
        embedding_cost_per_call = 0.02 / 1_000_000 * 750  # ~$0.000015 per embedding
        llm_cost_per_call = 3.00 / 1_000_000 * 500  # ~$0.0015 per call (500 tokens avg)

        emb_hits = emb_stats["hits"]
        llm_hits = llm_stats["hits"]

        emb_savings = emb_hits * embedding_cost_per_call
        llm_savings = llm_hits * llm_cost_per_call
        total_savings = emb_savings + llm_savings

        table = Table(title="Estimated API Cost Savings", show_header=True)
        table.add_column("Cache", style="cyan")
        table.add_column("Cache Hits", style="magenta")
        table.add_column("Estimated Savings", style="green")

        table.add_row(
            "Embeddings", f"{emb_hits:,}", f"${emb_savings:.4f} ({emb_savings * 1000:.2f}¢)"
        )
        table.add_row("LLM Responses", f"{llm_hits:,}", f"${llm_savings:.4f} ({llm_savings * 1000:.2f}¢)")
        table.add_row("[bold]Total[/]", f"[bold]{emb_hits + llm_hits:,}[/]", f"[bold]${total_savings:.4f}[/]")

        console.print(table)
        console.print()
        console.print(
            "[dim]Note: Cost estimates are approximate and based on average token counts.[/]"
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}", style="red")
        sys.exit(1)


@cli.command()
def performance():
    """Show cache performance metrics."""
    try:
        embedding_cache = get_embedding_cache()
        llm_cache = get_llm_cache()

        emb_stats = embedding_cache.get_stats()
        llm_stats = llm_cache.get_stats()

        table = Table(title="Cache Performance Metrics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Embedding Cache", style="magenta")
        table.add_column("LLM Cache", style="yellow")

        table.add_row(
            "Hit Rate",
            f"{emb_stats['hit_rate']:.1%}",
            f"{llm_stats['hit_rate']:.1%}",
        )

        # Memory vs Disk hit distribution
        emb_mem_pct = (
            emb_stats["memory_hits"] / emb_stats["hits"] * 100
            if emb_stats["hits"] > 0
            else 0
        )
        llm_mem_pct = (
            llm_stats["memory_hits"] / llm_stats["hits"] * 100
            if llm_stats["hits"] > 0
            else 0
        )

        table.add_row(
            "Memory Hit %", f"{emb_mem_pct:.1f}%", f"{llm_mem_pct:.1f}%"
        )

        emb_disk_pct = (
            emb_stats["disk_hits"] / emb_stats["hits"] * 100
            if emb_stats["hits"] > 0
            else 0
        )
        llm_disk_pct = (
            llm_stats["disk_hits"] / llm_stats["hits"] * 100
            if llm_stats["hits"] > 0
            else 0
        )

        table.add_row("Disk Hit %", f"{emb_disk_pct:.1f}%", f"{llm_disk_pct:.1f}%")

        # Cache efficiency
        emb_efficiency = (
            emb_stats["hits"] / emb_stats["total_requests"]
            if emb_stats["total_requests"] > 0
            else 0
        )
        llm_efficiency = (
            llm_stats["hits"] / llm_stats["total_requests"]
            if llm_stats["total_requests"] > 0
            else 0
        )

        table.add_row(
            "Cache Efficiency", f"{emb_efficiency:.1%}", f"{llm_efficiency:.1%}"
        )

        console.print(table)

        # Performance interpretation
        console.print()
        console.print("[bold]Performance Interpretation:[/]")

        avg_hit_rate = (emb_stats["hit_rate"] + llm_stats["hit_rate"]) / 2

        if avg_hit_rate > 0.7:
            console.print(
                "[green]Excellent: High cache hit rate indicates significant API cost savings[/]"
            )
        elif avg_hit_rate > 0.4:
            console.print(
                "[yellow]Good: Moderate cache hit rate, consider reviewing cache configuration[/]"
            )
        else:
            console.print(
                "[red]Low: Cache hit rate is low, may indicate unique requests or small cache[/]"
            )

    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    cli()
