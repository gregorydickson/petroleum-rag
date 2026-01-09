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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Results", "💬 Chat Demo", "📈 Charts", "🔬 How It Works", "🏗️ Architecture"]
    )

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

    # Tab 4: How It Works
    with tab4:
        st.header("🔬 How the Benchmark Works")

        st.markdown("""
        Welcome! This page explains the benchmarking process in simple terms.
        """)

        # Step-by-step process
        st.subheader("The Benchmark Process (Step-by-Step)")

        # Step 1
        st.markdown("### 📄 Step 1: Upload Your Documents")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            You provide PDF documents (like technical handbooks, specifications, etc.)
            by placing them in the `data/input/` folder.

            **Example:** A petroleum refining handbook with tables, diagrams, and technical specs.
            """)
        with col2:
            st.info("**Input**\n\n📁 Your PDF files")

        st.divider()

        # Step 2
        st.markdown("### 🔄 Step 2: Document Parsing (4 Different Ways)")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            We test **4 different parsers** to see which one extracts information best:

            - **🦙 LlamaParse**: Cloud-based, great for complex tables
            - **🧠 Docling**: IBM's parser, good for structure preservation
            - **📄 PageIndex**: Semantic chunking approach
            - **☁️ Vertex AI**: Google's enterprise OCR

            Each parser reads your PDF differently and creates "chunks" of text.
            """)
        with col2:
            st.success("**Output**\n\n4 parsed versions\nof your document")

        st.divider()

        # Step 3
        st.markdown("### 💾 Step 3: Storage (3 Different Databases)")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            Each parsed version is stored in **3 different databases**:

            - **🎯 ChromaDB**: Fast vector similarity search
            - **🔀 Weaviate**: Hybrid search (keywords + meaning)
            - **🕸️ FalkorDB**: Graph database with relationships

            **Math:** 4 parsers × 3 databases = **12 combinations** to test!
            """)
        with col2:
            st.info("**Output**\n\n12 different\nRAG systems")

        st.divider()

        # Step 4
        st.markdown("### 🎯 Step 4: Testing with Real Questions")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            We ask **15 petroleum engineering questions** to each of the 12 combinations.

            **Example questions:**
            - "What are the pressure ratings for 2-inch valves?"
            - "What safety procedures are required for H2S?"
            - "Compare corrosion prevention methods"

            **Total tests:** 15 questions × 12 combinations = **180 tests**
            """)
        with col2:
            st.success("**Output**\n\n180 answers\nwith sources")

        st.divider()

        # Step 5
        st.markdown("### 📊 Step 5: Measuring Quality")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            We measure each answer using **multiple metrics**:

            **Accuracy Metrics:**
            - ✅ Precision: Are the results relevant?
            - ✅ Recall: Did we find all relevant information?
            - ✅ NDCG: Is the ranking correct?

            **Quality Metrics:**
            - ✨ Relevance: Does the answer match the question?
            - 🎯 Correctness: Is the answer accurate?
            - 🛡️ Faithfulness: Is it supported by the sources?
            """)
        with col2:
            st.info("**Output**\n\nQuality scores\nfor each combo")

        st.divider()

        # Step 6
        st.markdown("### 🏆 Step 6: Finding the Winner")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            We combine all the metrics into a **composite score** and rank all 12 combinations.

            The combination with the highest score wins!

            **Current winner:** {winner}
            """.format(winner=f"**{winner_parser} + {winner_storage}**" if not df.empty else "Not yet determined"))
        with col2:
            st.success("**Output**\n\n🏆 Best\nconfiguration!")

        st.divider()

        # Visual timeline
        st.subheader("⏱️ Typical Processing Time")

        timeline_data = {
            "Phase": ["📄 Parsing", "💾 Storage", "🎯 Testing", "📊 Analysis"],
            "Time": [22, 17, 12, 1],
            "Description": [
                "4 parsers process your PDF",
                "Store in 3 databases (12 combinations)",
                "Run 15 queries × 12 combos = 180 tests",
                "Calculate metrics and generate charts"
            ]
        }

        timeline_df = pd.DataFrame(timeline_data)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(
                timeline_df,
                use_container_width=True,
                hide_index=True
            )
        with col2:
            st.metric("Total Time (First Run)", "~52 minutes", help="For an 11MB PDF with 15 queries")
            st.metric("Total Time (Cached)", "~15 minutes", delta="-37 min", help="Thanks to 97% cache hit rate!")

        st.divider()

        # What happens next
        st.subheader("💡 What Happens Next?")

        st.markdown("""
        After the benchmark completes, you can:

        1. **📊 View Results** - See which combination won and compare all 12
        2. **💬 Chat** - Ask questions using the winning configuration
        3. **📈 Analyze Charts** - Explore detailed visualizations
        4. **🚀 Deploy** - Use the winning combination in production

        The winning combination is automatically used in the **Chat Demo** tab!
        """)

    # Tab 5: Architecture
    with tab5:
        st.header("🏗️ System Architecture")

        st.markdown("""
        This page shows you the components of the application and how they work together.
        """)

        # High-level architecture
        st.subheader("🎯 High-Level Overview")

        st.markdown("""
        ```
        ┌─────────────────────────────────────────────────────────────┐
        │                   PETROLEUM RAG BENCHMARK                    │
        │                                                              │
        │  You upload PDFs → We test 12 configurations → Find winner  │
        └─────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
              ┌─────────┐     ┌─────────┐     ┌─────────┐
              │ PARSERS │     │ STORAGE │     │  EVAL   │
              │  (4)    │     │  (3)    │     │ METRICS │
              └─────────┘     └─────────┘     └─────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                            🏆 WINNING COMBO
        ```
        """)

        st.divider()

        # Component details
        st.subheader("🔍 Component Details")

        # Parsers
        with st.expander("📄 **PARSERS** - Convert PDFs to Searchable Text", expanded=True):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("""
                **🦙 LlamaParse**

                *Cloud-based parser*

                ✅ Excellent table extraction
                ✅ Multi-column layouts
                ✅ Complex documents

                ⚡ Speed: Medium
                💰 Cost: API calls
                """)

            with col2:
                st.markdown("""
                **🧠 Docling**

                *IBM Research parser*

                ✅ Structure preservation
                ✅ Semantic chunking
                ✅ Local processing

                ⚡ Speed: Fast
                💰 Cost: Free
                """)

            with col3:
                st.markdown("""
                **📄 PageIndex**

                *Semantic approach*

                ✅ Context preservation
                ✅ Semantic boundaries
                ✅ Page relationships

                ⚡ Speed: Fast
                💰 Cost: Free
                """)

            with col4:
                st.markdown("""
                **☁️ Vertex AI**

                *Google Cloud parser*

                ✅ Enterprise OCR
                ✅ Form extraction
                ✅ High accuracy

                ⚡ Speed: Medium
                💰 Cost: API calls
                """)

        # Storage
        with st.expander("💾 **STORAGE** - Store and Retrieve Information", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                **🎯 ChromaDB**

                *Vector similarity search*

                **How it works:**
                - Converts text to numbers (embeddings)
                - Finds similar meanings
                - Pure semantic search

                **Best for:**
                - Fast semantic queries
                - Simple setup
                - Single-hop questions

                ⚡ Speed: Very Fast
                🎯 Accuracy: Good
                """)

            with col2:
                st.markdown("""
                **🔀 Weaviate**

                *Hybrid search engine*

                **How it works:**
                - Combines semantic + keywords
                - BM25 keyword matching
                - Vector similarity

                **Best for:**
                - Mixed query types
                - Exact + semantic matches
                - Production systems

                ⚡ Speed: Fast
                🎯 Accuracy: Excellent
                """)

            with col3:
                st.markdown("""
                **🕸️ FalkorDB**

                *Graph database*

                **How it works:**
                - Stores relationships
                - Graph traversal
                - Multi-hop queries

                **Best for:**
                - Connected information
                - Complex relationships
                - Multi-step reasoning

                ⚡ Speed: Medium
                🎯 Accuracy: Very Good
                """)

        # Evaluation
        with st.expander("📊 **EVALUATION** - Measure Quality", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                **📈 Traditional Metrics**

                *Mathematical precision*

                - **Precision@K**: % of results that are relevant
                - **Recall@K**: % of all relevant docs found
                - **F1 Score**: Balance of precision & recall
                - **NDCG**: Ranking quality (0-1)
                - **MRR**: Mean reciprocal rank
                - **MAP**: Mean average precision

                These metrics are objective and mathematical.
                """)

            with col2:
                st.markdown("""
                **🤖 LLM-Based Metrics**

                *AI-powered evaluation*

                - **Context Relevance**: Are sources relevant?
                - **Answer Correctness**: Is answer accurate?
                - **Faithfulness**: Supported by sources?
                - **Semantic Similarity**: Matches intent?
                - **Completeness**: Full answer?
                - **Hallucination Check**: Made up info?

                These metrics use Claude to judge quality.
                """)

        st.divider()

        # Data flow
        st.subheader("🔄 Data Flow: From PDF to Answer")

        st.markdown("""
        ```
        1️⃣  PDF Document
              │
              ▼
        2️⃣  Parser extracts text & tables
              │
              ▼
        3️⃣  Text split into chunks (with overlap)
              │
              ▼
        4️⃣  Chunks converted to embeddings (vectors)
              │
              ▼
        5️⃣  Embeddings stored in database
              │
              ▼
        6️⃣  User asks a question
              │
              ▼
        7️⃣  Question converted to embedding
              │
              ▼
        8️⃣  Database finds similar chunks
              │
              ▼
        9️⃣  LLM generates answer from chunks
              │
              ▼
        🔟 Answer + sources returned to user
        ```
        """)

        st.divider()

        # Technologies used
        st.subheader("🛠️ Technologies & APIs")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **Parsers**
            - LlamaParse API
            - Docling (IBM)
            - Custom PageIndex
            - Google Vertex AI
            """)

        with col2:
            st.markdown("""
            **Storage**
            - ChromaDB
            - Weaviate
            - FalkorDB (Redis)
            - Docker containers
            """)

        with col3:
            st.markdown("""
            **AI & Processing**
            - OpenAI embeddings
            - Claude (Anthropic)
            - Python/asyncio
            - Streamlit UI
            """)

        st.divider()

        # Why this matters
        st.subheader("💡 Why Test All These Combinations?")

        st.markdown("""
        Different documents need different approaches! Here's why we test everything:

        **📊 Tables & Data**
        - Some parsers extract tables better than others
        - LlamaParse excels at complex tables

        **🔍 Search Types**
        - Keyword search: Weaviate's BM25
        - Semantic search: ChromaDB's vectors
        - Relationships: FalkorDB's graphs

        **⚡ Speed vs Accuracy**
        - ChromaDB is fastest
        - Weaviate balances speed & accuracy
        - FalkorDB handles complex queries

        **💰 Cost**
        - Local parsers are free
        - Cloud APIs cost money
        - We help you find the best value!

        **By testing all 12 combinations, we find the BEST setup for YOUR specific documents!**
        """)

        st.success("""
        🏆 **The Result?** You get a production-ready RAG system optimized for
        petroleum engineering documents, with proof that it works!
        """)


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
