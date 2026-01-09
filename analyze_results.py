"""Analyze and visualize benchmark results.

This module provides tools for:
1. Loading benchmark results
2. Creating comparison charts and visualizations
3. Identifying the best parser-storage combination
4. Generating comprehensive reports
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.logging import get_logger

logger = get_logger(__name__)

# Set style for better-looking charts
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


class ResultsAnalyzer:
    """Analyze and visualize benchmark results."""

    def __init__(self, results_file: Path) -> None:
        """Initialize analyzer with results file.

        Args:
            results_file: Path to raw_results.json
        """
        self.results_file = results_file
        self.results_data: dict[str, Any] = {}
        self.df: pd.DataFrame | None = None

    def load_results(self) -> pd.DataFrame:
        """Load benchmark results into DataFrame.

        Returns:
            DataFrame with all benchmark results
        """
        logger.info(f"Loading results from {self.results_file}")

        with open(self.results_file) as f:
            self.results_data = json.load(f)

        # Convert results to DataFrame
        results_list = self.results_data.get("results", [])

        if not results_list:
            raise ValueError("No results found in file")

        # Flatten metrics into columns
        rows = []
        for result in results_list:
            row = {
                "combination": result["combination"],
                "parser_name": result["parser_name"],
                "storage_backend": result["storage_backend"],
                "query_id": result["query_id"],
                "retrieval_time": result["retrieval_time_seconds"],
                "generation_time": result["generation_time_seconds"],
                "total_time": result["total_time_seconds"],
                "success": result["success"],
                "retrieved_count": result["retrieved_count"],
            }

            # Add metrics
            for metric_name, metric_value in result["metrics"].items():
                row[metric_name] = metric_value

            rows.append(row)

        self.df = pd.DataFrame(rows)

        logger.info(f"Loaded {len(self.df)} results")
        logger.info(f"Combinations: {self.df['combination'].nunique()}")
        logger.info(f"Queries: {self.df['query_id'].nunique()}")

        return self.df

    def calculate_aggregate_metrics(self) -> pd.DataFrame:
        """Calculate aggregate metrics per combination.

        Returns:
            DataFrame with mean metrics per combination
        """
        if self.df is None:
            raise ValueError("Must call load_results() first")

        logger.info("Calculating aggregate metrics")

        # Group by combination and calculate means
        numeric_cols = self.df.select_dtypes(include=["float64", "int64"]).columns
        numeric_cols = [c for c in numeric_cols if c != "success"]  # Exclude boolean

        agg_metrics = self.df.groupby("combination")[numeric_cols].mean()

        # Add success rate
        agg_metrics["success_rate"] = self.df.groupby("combination")["success"].mean()

        # Sort by a composite score (you can adjust weights)
        # Higher is better for most metrics
        agg_metrics["composite_score"] = (
            agg_metrics.get("precision@5", 0) * 0.2
            + agg_metrics.get("recall@5", 0) * 0.2
            + agg_metrics.get("f1@5", 0) * 0.15
            + agg_metrics.get("mrr", 0) * 0.15
            + agg_metrics.get("ndcg@5", 0) * 0.1
            + agg_metrics.get("context_relevance", 0) * 0.05
            + agg_metrics.get("answer_correctness", 0) * 0.1
            + agg_metrics.get("faithfulness", 0) * 0.05
        )

        agg_metrics = agg_metrics.sort_values("composite_score", ascending=False)

        return agg_metrics

    def create_comparison_charts(self, output_dir: Path) -> None:
        """Create comparison charts and save to output directory.

        Args:
            output_dir: Directory to save charts
        """
        if self.df is None:
            raise ValueError("Must call load_results() first")

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating comparison charts in {output_dir}")

        # Calculate aggregates
        agg_metrics = self.calculate_aggregate_metrics()

        # 1. Heatmap: Parser × Storage → Composite Score
        self._create_heatmap(agg_metrics, output_dir)

        # 2. Bar charts: Key metrics comparison
        self._create_metric_bars(agg_metrics, output_dir)

        # 3. Timing comparison
        self._create_timing_chart(agg_metrics, output_dir)

        # 4. Radar chart: Top 3 combinations
        self._create_radar_chart(agg_metrics, output_dir)

        # 5. Precision-Recall curves
        self._create_precision_recall(output_dir)

        logger.info(f"All charts saved to {output_dir}")

    def _create_heatmap(self, agg_metrics: pd.DataFrame, output_dir: Path) -> None:
        """Create heatmap of parser × storage performance.

        Args:
            agg_metrics: Aggregated metrics DataFrame
            output_dir: Output directory
        """
        logger.info("Creating heatmap")

        # Extract parser and storage from combination name
        parsers = []
        storages = []
        for combo in agg_metrics.index:
            parts = combo.rsplit("_", 1)
            if len(parts) == 2:
                parsers.append(parts[0])
                storages.append(parts[1])
            else:
                parsers.append(combo)
                storages.append("unknown")

        agg_metrics["parser"] = parsers
        agg_metrics["storage"] = storages

        # Pivot for heatmap
        pivot_data = agg_metrics.pivot(
            index="parser",
            columns="storage",
            values="composite_score",
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            cbar_kws={"label": "Composite Score"},
        )
        plt.title("Parser × Storage Backend Performance", fontsize=16, fontweight="bold")
        plt.xlabel("Storage Backend", fontsize=12)
        plt.ylabel("Parser", fontsize=12)
        plt.tight_layout()

        output_file = output_dir / "heatmap_performance.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved heatmap to {output_file}")

    def _create_metric_bars(self, agg_metrics: pd.DataFrame, output_dir: Path) -> None:
        """Create bar charts comparing key metrics.

        Args:
            agg_metrics: Aggregated metrics DataFrame
            output_dir: Output directory
        """
        logger.info("Creating metric bar charts")

        # Select key metrics
        key_metrics = [
            "precision@5",
            "recall@5",
            "f1@5",
            "mrr",
            "ndcg@5",
            "context_relevance",
            "answer_correctness",
            "faithfulness",
        ]

        available_metrics = [m for m in key_metrics if m in agg_metrics.columns]

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]

            data = agg_metrics[metric].sort_values(ascending=False)

            ax.barh(range(len(data)), data.values)
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels(data.index, fontsize=8)
            ax.set_xlabel("Score", fontsize=10)
            ax.set_title(metric.replace("_", " ").title(), fontsize=11, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)

            # Add value labels
            for i, v in enumerate(data.values):
                ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)

        # Hide unused subplots
        for idx in range(len(available_metrics), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            "Key Metrics Comparison Across All Combinations",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        output_file = output_dir / "metric_bars.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved metric bars to {output_file}")

    def _create_timing_chart(self, agg_metrics: pd.DataFrame, output_dir: Path) -> None:
        """Create timing comparison chart.

        Args:
            agg_metrics: Aggregated metrics DataFrame
            output_dir: Output directory
        """
        logger.info("Creating timing chart")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Retrieval time
        data = agg_metrics["retrieval_time"].sort_values()
        ax1.barh(range(len(data)), data.values, color="skyblue")
        ax1.set_yticks(range(len(data)))
        ax1.set_yticklabels(data.index, fontsize=9)
        ax1.set_xlabel("Seconds", fontsize=11)
        ax1.set_title("Average Retrieval Time", fontsize=13, fontweight="bold")
        ax1.grid(axis="x", alpha=0.3)

        for i, v in enumerate(data.values):
            ax1.text(v + 0.01, i, f"{v:.3f}s", va="center", fontsize=8)

        # Total time
        data = agg_metrics["total_time"].sort_values()
        ax2.barh(range(len(data)), data.values, color="lightcoral")
        ax2.set_yticks(range(len(data)))
        ax2.set_yticklabels(data.index, fontsize=9)
        ax2.set_xlabel("Seconds", fontsize=11)
        ax2.set_title("Average Total Time (Retrieval + Generation)", fontsize=13, fontweight="bold")
        ax2.grid(axis="x", alpha=0.3)

        for i, v in enumerate(data.values):
            ax2.text(v + 0.01, i, f"{v:.3f}s", va="center", fontsize=8)

        plt.suptitle("Timing Comparison", fontsize=16, fontweight="bold")
        plt.tight_layout()

        output_file = output_dir / "timing_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved timing chart to {output_file}")

    def _create_radar_chart(self, agg_metrics: pd.DataFrame, output_dir: Path) -> None:
        """Create radar chart for top 3 combinations.

        Args:
            agg_metrics: Aggregated metrics DataFrame
            output_dir: Output directory
        """
        logger.info("Creating radar chart for top 3")

        import numpy as np

        # Select top 3 by composite score
        top3 = agg_metrics.head(3)

        # Select metrics for radar
        radar_metrics = [
            "precision@5",
            "recall@5",
            "mrr",
            "ndcg@5",
            "context_relevance",
            "answer_correctness",
            "faithfulness",
        ]

        available_metrics = [m for m in radar_metrics if m in agg_metrics.columns]

        if len(available_metrics) < 3:
            logger.warning("Not enough metrics for radar chart")
            return

        # Number of metrics
        num_metrics = len(available_metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        for idx, (combo, row) in enumerate(top3.iterrows()):
            values = [row[m] for m in available_metrics]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, "o-", linewidth=2, label=combo)
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("_", " ").title() for m in available_metrics], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title("Top 3 Combinations - Multi-Metric Comparison", fontsize=14, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True)

        plt.tight_layout()

        output_file = output_dir / "radar_top3.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved radar chart to {output_file}")

    def _create_precision_recall(self, output_dir: Path) -> None:
        """Create precision-recall comparison at different K values.

        Args:
            output_dir: Output directory
        """
        logger.info("Creating precision-recall chart")

        if self.df is None:
            return

        # Extract precision and recall at different K values
        k_values = [1, 3, 5, 10]
        available_k = [k for k in k_values if f"precision@{k}" in self.df.columns]

        if not available_k:
            logger.warning("No precision/recall metrics found")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Calculate mean per combination
        combinations = self.df["combination"].unique()

        for combo in combinations:
            combo_data = self.df[self.df["combination"] == combo]

            precisions = [combo_data[f"precision@{k}"].mean() for k in available_k]
            recalls = [combo_data[f"recall@{k}"].mean() for k in available_k]

            ax1.plot(available_k, precisions, marker="o", label=combo, linewidth=2)
            ax2.plot(available_k, recalls, marker="o", label=combo, linewidth=2)

        ax1.set_xlabel("K", fontsize=12)
        ax1.set_ylabel("Precision", fontsize=12)
        ax1.set_title("Precision@K", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("K", fontsize=12)
        ax2.set_ylabel("Recall", fontsize=12)
        ax2.set_title("Recall@K", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.suptitle("Precision and Recall at Different K Values", fontsize=16, fontweight="bold")
        plt.tight_layout()

        output_file = output_dir / "precision_recall.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved precision-recall chart to {output_file}")

    def print_winner(self) -> dict[str, Any]:
        """Print the winning combination with all metrics.

        Returns:
            Dictionary with winner information
        """
        if self.df is None:
            raise ValueError("Must call load_results() first")

        logger.info("Identifying winner")

        agg_metrics = self.calculate_aggregate_metrics()

        winner = agg_metrics.index[0]
        winner_metrics = agg_metrics.iloc[0].to_dict()

        print("\n" + "=" * 80)
        print("WINNER")
        print("=" * 80)
        print(f"\nBest Combination: {winner}")
        print(f"Composite Score: {winner_metrics['composite_score']:.4f}")
        print("\nKey Metrics:")

        # Print key metrics in a formatted way
        key_metrics = [
            ("Precision@5", "precision@5"),
            ("Recall@5", "recall@5"),
            ("F1@5", "f1@5"),
            ("MRR", "mrr"),
            ("NDCG@5", "ndcg@5"),
            ("Context Relevance", "context_relevance"),
            ("Answer Correctness", "answer_correctness"),
            ("Faithfulness", "faithfulness"),
        ]

        for display_name, metric_key in key_metrics:
            if metric_key in winner_metrics:
                print(f"  {display_name:.<30} {winner_metrics[metric_key]:.4f}")

        print("\nTiming:")
        print(f"  Retrieval Time:............... {winner_metrics['retrieval_time']:.3f}s")
        print(f"  Generation Time:.............. {winner_metrics['generation_time']:.3f}s")
        print(f"  Total Time:................... {winner_metrics['total_time']:.3f}s")

        print("\n" + "=" * 80)

        return {
            "combination": winner,
            "metrics": winner_metrics,
        }

    def generate_report(self, output_dir: Path) -> None:
        """Generate comprehensive markdown report.

        Args:
            output_dir: Directory to save report
        """
        logger.info("Generating report")

        if self.df is None:
            raise ValueError("Must call load_results() first")

        agg_metrics = self.calculate_aggregate_metrics()
        winner_info = self.print_winner()

        report_file = output_dir / "REPORT.md"

        with open(report_file, "w") as f:
            f.write("# Petroleum RAG Benchmark Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")

            # Summary
            f.write("## Summary\n\n")
            summary = self.results_data.get("summary", {})
            f.write(f"- **Total Combinations**: {summary.get('total_combinations', 'N/A')}\n")
            f.write(f"- **Total Queries**: {summary.get('total_queries', 'N/A')}\n")
            f.write(f"- **Total Results**: {summary.get('total_results', 'N/A')}\n")
            f.write(f"- **Total Time**: {summary.get('total_time_seconds', 0) / 60:.1f} minutes\n\n")

            # Cache Statistics
            cache_stats = summary.get("cache_statistics", {})
            if cache_stats:
                f.write("## Cache Performance\n\n")
                f.write("Caching significantly improves performance by avoiding redundant API calls.\n\n")

                # Create cache statistics table
                f.write("| Cache Type | Hit Rate | Hits | Misses | Total Requests | Memory Items | Disk Items | Disk Size |\n")
                f.write("|------------|----------|------|--------|----------------|--------------|------------|------------|\n")

                for cache_type, stats in cache_stats.items():
                    hit_rate = stats.get("hit_rate", 0)
                    hits = stats.get("hits", 0)
                    misses = stats.get("misses", 0)
                    total_requests = stats.get("total_requests", 0)
                    memory_size = stats.get("memory_size", 0)
                    disk_items = stats.get("disk_items", 0)
                    disk_mb = stats.get("disk_mb", 0)

                    f.write(
                        f"| {cache_type.title()} | {hit_rate:.1%} | {hits} | {misses} | "
                        f"{total_requests} | {memory_size} | {disk_items} | {disk_mb:.2f} MB |\n"
                    )

                f.write("\n")

                # Calculate overall cache hit rate
                total_hits = sum(s.get("hits", 0) for s in cache_stats.values())
                total_misses = sum(s.get("misses", 0) for s in cache_stats.values())
                total_cache_requests = total_hits + total_misses
                overall_hit_rate = total_hits / total_cache_requests if total_cache_requests > 0 else 0

                f.write(f"**Overall Cache Hit Rate**: {overall_hit_rate:.1%}\n\n")

                # Cost savings estimate
                if overall_hit_rate > 0:
                    cost_savings = overall_hit_rate * 100
                    f.write(f"**Estimated Cost Savings**: ~{cost_savings:.0f}% on cached operations\n\n")
                    f.write(
                        "Cache hit rates vary based on query patterns and document reuse. "
                        "First runs will have lower hit rates, while subsequent runs with similar "
                        "queries can achieve 90%+ hit rates.\n\n"
                    )

            # Winner
            f.write("## Winner\n\n")
            f.write(f"**{winner_info['combination']}**\n\n")
            f.write(f"Composite Score: **{winner_info['metrics']['composite_score']:.4f}**\n\n")

            # All results table
            f.write("## All Results\n\n")
            f.write("Results sorted by composite score (higher is better):\n\n")

            # Create markdown table
            f.write("| Rank | Combination | Composite | Precision@5 | Recall@5 | F1@5 | MRR | NDCG@5 |\n")
            f.write("|------|-------------|-----------|-------------|----------|------|-----|--------|\n")

            for rank, (combo, row) in enumerate(agg_metrics.iterrows(), start=1):
                f.write(
                    f"| {rank} | {combo} | {row['composite_score']:.4f} | "
                    f"{row.get('precision@5', 0):.4f} | {row.get('recall@5', 0):.4f} | "
                    f"{row.get('f1@5', 0):.4f} | {row.get('mrr', 0):.4f} | "
                    f"{row.get('ndcg@5', 0):.4f} |\n"
                )

            f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")

            winner = agg_metrics.index[0]
            parser_name = winner.rsplit("_", 1)[0] if "_" in winner else winner
            storage_name = winner.rsplit("_", 1)[1] if "_" in winner else "unknown"

            f.write(f"Based on the benchmark results, we recommend:\n\n")
            f.write(f"1. **Parser**: {parser_name}\n")
            f.write(f"2. **Storage Backend**: {storage_name}\n\n")

            f.write("### Why This Combination?\n\n")
            f.write("This combination achieved the highest composite score across all evaluation metrics, ")
            f.write("balancing retrieval accuracy, answer quality, and performance.\n\n")

            # Charts
            f.write("## Visualizations\n\n")
            f.write("See the `charts/` directory for detailed visualizations:\n\n")
            f.write("- `heatmap_performance.png` - Parser × Storage performance heatmap\n")
            f.write("- `metric_bars.png` - Key metrics comparison\n")
            f.write("- `timing_comparison.png` - Retrieval and total timing\n")
            f.write("- `radar_top3.png` - Multi-metric comparison of top 3\n")
            f.write("- `precision_recall.png` - Precision and recall at different K values\n\n")

        logger.info(f"Saved report to {report_file}")


def main():
    """Main entry point for analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("data/results/raw_results.json"),
        help="Path to raw results JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/charts"),
        help="Directory to save charts",
    )

    args = parser.parse_args()

    if not args.results_file.exists():
        logger.error(f"Results file not found: {args.results_file}")
        logger.error("Run benchmark.py first to generate results")
        return

    # Analyze results
    analyzer = ResultsAnalyzer(args.results_file)

    # Load and analyze
    analyzer.load_results()

    # Print winner
    analyzer.print_winner()

    # Create charts
    analyzer.create_comparison_charts(args.output_dir)

    # Generate report
    analyzer.generate_report(args.output_dir.parent)

    logger.info("\nAnalysis complete!")
    logger.info(f"Charts saved to: {args.output_dir}")
    logger.info(f"Report saved to: {args.output_dir.parent / 'REPORT.md'}")


if __name__ == "__main__":
    main()
