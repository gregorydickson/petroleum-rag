"""Streamlit demo application for Petroleum RAG Benchmark.

This app provides:
1. Interactive display of benchmark results
2. Chat interface using the winning combination
3. Source attribution and visualization
4. Comparison charts
"""

import asyncio
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from config import settings
from embeddings import UnifiedEmbedder
from evaluation import Evaluator
from parsers import (
    DoclingParser,
    LlamaParseParser,
    PageIndexParser,
    VertexDocAIParser,
)
from storage import ChromaStore, FalkorDBStore, WeaviateStore

# Page config
st.set_page_config(
    page_title="Petroleum RAG Benchmark",
    page_icon="🛢️",
    layout="wide",
)


@st.cache_data
def load_results(results_file: Path) -> dict:
    """Load benchmark results from JSON file.

    Args:
        results_file: Path to raw_results.json

    Returns:
        Dictionary with results data
    """
    with open(results_file) as f:
        return json.load(f)


@st.cache_data
def load_comparison_data(results_data: dict) -> pd.DataFrame:
    """Convert results to comparison DataFrame.

    Args:
        results_data: Raw results dictionary

    Returns:
        DataFrame with aggregate metrics per combination
    """
    from collections import defaultdict

    results_list = results_data.get("results", [])

    # Calculate aggregate metrics per combination
    combination_metrics = defaultdict(lambda: defaultdict(list))

    for result in results_list:
        combo = result["combination"]
        for metric_name, metric_value in result["metrics"].items():
            combination_metrics[combo][metric_name].append(metric_value)

        combination_metrics[combo]["retrieval_time"].append(result["retrieval_time_seconds"])
        combination_metrics[combo]["total_time"].append(result["total_time_seconds"])

    # Calculate means
    rows = []
    for combo, metrics in combination_metrics.items():
        row = {"combination": combo}

        for metric_name, values in metrics.items():
            if values:
                row[f"{metric_name}"] = sum(values) / len(values)

        # Calculate composite score
        row["composite_score"] = (
            row.get("precision@5", 0) * 0.2
            + row.get("recall@5", 0) * 0.2
            + row.get("f1@5", 0) * 0.15
            + row.get("mrr", 0) * 0.15
            + row.get("ndcg@5", 0) * 0.1
            + row.get("context_relevance", 0) * 0.05
            + row.get("answer_correctness", 0) * 0.1
            + row.get("faithfulness", 0) * 0.05
        )

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("composite_score", ascending=False)

    return df


def get_winner(df: pd.DataFrame) -> tuple[str, str]:
    """Get winning parser and storage combination.

    Args:
        df: Comparison DataFrame

    Returns:
        Tuple of (parser_name, storage_name)
    """
    if df.empty:
        return "LlamaParse", "Chroma"

    winner = df.iloc[0]["combination"]

    parts = winner.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]

    return winner, "Chroma"


def initialize_components(parser_name: str, storage_name: str):
    """Initialize parser, storage, embedder, and evaluator.

    Args:
        parser_name: Name of parser
        storage_name: Name of storage backend

    Returns:
        Tuple of (parser, storage, embedder, evaluator)
    """
    # Initialize parser
    parser_map = {
        "LlamaParse": LlamaParseParser,
        "Docling": DoclingParser,
        "PageIndex": PageIndexParser,
        "VertexDocAI": VertexDocAIParser,
    }

    parser_class = parser_map.get(parser_name, LlamaParseParser)
    parser = parser_class()

    # Initialize storage
    storage_map = {
        "Chroma": ChromaStore,
        "Weaviate": WeaviateStore,
        "FalkorDB": FalkorDBStore,
    }

    storage_class = storage_map.get(storage_name, ChromaStore)
    storage = storage_class()

    # Initialize embedder and evaluator
    embedder = UnifiedEmbedder()
    evaluator = Evaluator()

    return parser, storage, embedder, evaluator


async def query_rag_system(
    query: str,
    storage,
    embedder,
    evaluator,
    top_k: int = 5,
) -> tuple[str, list]:
    """Query the RAG system.

    Args:
        query: User query
        storage: Storage backend
        embedder: Embedder
        evaluator: Evaluator
        top_k: Number of results to retrieve

    Returns:
        Tuple of (answer, retrieved_results)
    """
    # Generate query embedding
    query_embedding = await embedder.embed_text(query)

    # Retrieve results
    retrieved = await storage.retrieve(
        query=query,
        query_embedding=query_embedding,
        top_k=top_k,
    )

    # Generate answer
    answer = await evaluator.generate_answer(query, retrieved)

    return answer, retrieved


def main():
    """Main Streamlit app."""
    st.title("🛢️ Petroleum RAG Benchmark Dashboard")

    # Sidebar
    st.sidebar.title("Configuration")

    # Check for results
    results_file = Path("data/results/raw_results.json")

    if not results_file.exists():
        st.error(
            "No benchmark results found. Please run `python benchmark.py` first "
            "to generate results."
        )
        st.stop()

    # Load results
    results_data = load_results(results_file)
    df = load_comparison_data(results_data)

    # Get winner
    winner_parser, winner_storage = get_winner(df)

    st.sidebar.success(f"Winner: {winner_parser} + {winner_storage}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Results", "💬 Chat Demo", "📈 Charts"])

    # Tab 1: Results
    with tab1:
        st.header("Benchmark Results")

        summary = results_data.get("summary", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Combinations",
                summary.get("total_combinations", "N/A"),
            )

        with col2:
            st.metric(
                "Total Queries",
                summary.get("total_queries", "N/A"),
            )

        with col3:
            st.metric(
                "Total Results",
                summary.get("total_results", "N/A"),
            )

        with col4:
            total_time = summary.get("total_time_seconds", 0)
            st.metric(
                "Total Time",
                f"{total_time / 60:.1f} min",
            )

        st.subheader("Winner")

        winner_row = df.iloc[0]

        col1, col2 = st.columns([2, 1])

        with col1:
            st.success(f"**{winner_row['combination']}**")
            st.write(f"**Composite Score:** {winner_row['composite_score']:.4f}")

            st.write("**Key Metrics:**")

            metrics_cols = st.columns(4)

            with metrics_cols[0]:
                st.metric("Precision@5", f"{winner_row.get('precision@5', 0):.3f}")
                st.metric("Recall@5", f"{winner_row.get('recall@5', 0):.3f}")

            with metrics_cols[1]:
                st.metric("F1@5", f"{winner_row.get('f1@5', 0):.3f}")
                st.metric("MRR", f"{winner_row.get('mrr', 0):.3f}")

            with metrics_cols[2]:
                st.metric("NDCG@5", f"{winner_row.get('ndcg@5', 0):.3f}")
                st.metric("Context Rel.", f"{winner_row.get('context_relevance', 0):.3f}")

            with metrics_cols[3]:
                st.metric("Answer Corr.", f"{winner_row.get('answer_correctness', 0):.3f}")
                st.metric("Faithfulness", f"{winner_row.get('faithfulness', 0):.3f}")

        with col2:
            st.write("**Timing:**")
            st.metric("Retrieval", f"{winner_row.get('retrieval_time', 0):.3f}s")
            st.metric("Generation", f"{winner_row.get('generation_time', 0):.3f}s")
            st.metric("Total", f"{winner_row.get('total_time', 0):.3f}s")

        st.subheader("All Results")

        # Display comparison table
        display_cols = [
            "combination",
            "composite_score",
            "precision@5",
            "recall@5",
            "f1@5",
            "mrr",
            "ndcg@5",
            "retrieval_time",
            "total_time",
        ]

        available_cols = [c for c in display_cols if c in df.columns]

        st.dataframe(
            df[available_cols].style.background_gradient(
                subset=["composite_score"],
                cmap="YlGn",
            ),
            use_container_width=True,
        )

    # Tab 2: Chat Demo
    with tab2:
        st.header("Interactive Chat Demo")

        st.info(
            f"This demo uses the winning combination: **{winner_parser} + {winner_storage}**"
        )

        # Note about storage initialization
        st.warning(
            "Note: The storage backend must be initialized and loaded with data. "
            "If you see errors, ensure the benchmark has been run and data is stored."
        )

        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the maximum operating pressure for 2-inch valves?",
        )

        top_k = st.slider("Number of results to retrieve", 1, 10, 5)

        if st.button("Ask Question", type="primary"):
            if not query:
                st.warning("Please enter a question")
            else:
                with st.spinner("Processing..."):
                    try:
                        # Initialize components (cached)
                        if "components" not in st.session_state:
                            parser, storage, embedder, evaluator = initialize_components(
                                winner_parser, winner_storage
                            )

                            # Initialize storage
                            asyncio.run(storage.initialize())

                            st.session_state.components = (parser, storage, embedder, evaluator)

                        parser, storage, embedder, evaluator = st.session_state.components

                        # Query
                        answer, retrieved = asyncio.run(
                            query_rag_system(
                                query,
                                storage,
                                embedder,
                                evaluator,
                                top_k,
                            )
                        )

                        # Display answer
                        st.subheader("Answer")
                        st.write(answer)

                        # Display sources
                        st.subheader(f"Sources (Top {len(retrieved)})")

                        for idx, result in enumerate(retrieved, start=1):
                            with st.expander(
                                f"Source {idx} - Score: {result.score:.3f}",
                                expanded=(idx <= 3),
                            ):
                                st.write(result.content)

                                if result.metadata:
                                    st.write("**Metadata:**")
                                    st.json(result.metadata)

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.exception(e)

    # Tab 3: Charts
    with tab3:
        st.header("Comparison Charts")

        charts_dir = Path("data/results/charts")

        if not charts_dir.exists():
            st.warning(
                "No charts found. Run `python analyze_results.py` to generate visualizations."
            )
        else:
            chart_files = {
                "Performance Heatmap": "heatmap_performance.png",
                "Metric Bars": "metric_bars.png",
                "Timing Comparison": "timing_comparison.png",
                "Top 3 Radar Chart": "radar_top3.png",
                "Precision-Recall": "precision_recall.png",
            }

            for chart_name, filename in chart_files.items():
                chart_path = charts_dir / filename

                if chart_path.exists():
                    st.subheader(chart_name)
                    image = Image.open(chart_path)
                    st.image(image, use_container_width=True)
                else:
                    st.warning(f"Chart not found: {filename}")

        # Show report
        report_file = Path("data/results/REPORT.md")

        if report_file.exists():
            st.subheader("Full Report")

            with open(report_file) as f:
                report_content = f.read()

            st.markdown(report_content)


if __name__ == "__main__":
    # Check API keys
    missing_keys = settings.validate_required_keys()

    if missing_keys:
        st.error(
            f"Missing required API keys: {', '.join(missing_keys)}. "
            "Please set them in your .env file."
        )
        st.stop()

    main()
