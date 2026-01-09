"""Generate markdown reports with comparison tables and analysis."""

from datetime import datetime
from typing import Any

from petroleum_rag.testing.models import CombinationSummary, TestCase, TestResult


class ReportGenerator:
    """Generate comprehensive markdown reports from test results."""

    def generate_report(
        self,
        test_results: list[TestResult],
        test_cases: list[TestCase],
        combination_summaries: dict[str, CombinationSummary],
        summary_stats: dict[str, Any],
    ) -> str:
        """Generate complete markdown report.

        Args:
            test_results: All test results
            test_cases: All test case definitions
            combination_summaries: Aggregated results per combination
            summary_stats: Overall summary statistics

        Returns:
            Markdown formatted report string
        """
        sections = []

        # Title and metadata
        sections.append(self._generate_header(summary_stats))

        # Summary
        sections.append(self._generate_summary(combination_summaries, summary_stats))

        # Winner announcement
        sections.append(self._generate_winner(combination_summaries))

        # Comparison table (EMPHASIZED)
        sections.append(self._generate_comparison_table(combination_summaries))

        # Per-test-case breakdown
        sections.append(self._generate_test_case_breakdown(test_cases, test_results))

        # Failure mode analysis
        sections.append(self._generate_failure_analysis(test_results, test_cases))

        # Detailed results
        sections.append(self._generate_detailed_results(combination_summaries))

        return "\n\n".join(sections)

    def _generate_header(self, summary_stats: dict[str, Any]) -> str:
        """Generate report header."""
        timestamp = datetime.fromisoformat(summary_stats["timestamp"])
        return f"""# 🧪 Petroleum RAG Test Results

**Generated**: {timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}

**Document**: Handbook of Petroleum Refining Processes (McGraw-Hill, 2nd Edition)"""

    def _generate_summary(
        self,
        combination_summaries: dict[str, CombinationSummary],
        summary_stats: dict[str, Any],
    ) -> str:
        """Generate summary section."""
        total_passed = sum(s.passed_tests for s in combination_summaries.values())
        total_tests = summary_stats["total_tests_run"]
        overall_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        total_time_minutes = summary_stats["total_time_seconds"] / 60

        return f"""## 📊 Summary

- **Total Combinations**: {summary_stats["total_combinations"]} (4 parsers × 3 storage backends)
- **Total Test Cases**: {summary_stats["total_test_cases"]}
- **Total Tests Run**: {total_tests} ({summary_stats["total_combinations"]} × {summary_stats["total_test_cases"]})
- **Overall Pass Rate**: {overall_pass_rate:.1f}% ({total_passed}/{total_tests})
- **Total Execution Time**: {total_time_minutes:.1f} minutes"""

    def _generate_winner(self, combination_summaries: dict[str, CombinationSummary]) -> str:
        """Generate winner announcement."""
        # Sort by pass rate, then by average score
        sorted_combos = sorted(
            combination_summaries.values(),
            key=lambda s: (s.pass_rate, s.avg_score),
            reverse=True,
        )

        if not sorted_combos:
            return "## 🏆 Winner\n\nNo valid results"

        winner = sorted_combos[0]

        return f"""## 🏆 Winner: {winner.combination}

**{winner.parser_name}** + **{winner.storage_backend}**

- **Pass Rate**: {winner.pass_rate * 100:.1f}% ({winner.passed_tests}/{winner.total_tests})
- **Average Score**: {winner.avg_score:.3f}
- **Average Retrieval Time**: {winner.avg_retrieval_time:.2f}s
- **Average Generation Time**: {winner.avg_generation_time:.2f}s
- **Total Time**: {winner.total_time:.1f}s"""

    def _generate_comparison_table(
        self,
        combination_summaries: dict[str, CombinationSummary],
    ) -> str:
        """Generate side-by-side comparison table."""
        # Sort by pass rate descending
        sorted_combos = sorted(
            combination_summaries.values(),
            key=lambda s: (s.pass_rate, s.avg_score),
            reverse=True,
        )

        rows = ["## 📈 Comparison Table\n"]
        rows.append(
            "| Rank | Combination | Pass Rate | Avg Score | Passed | Failed | Avg Retrieval (s) | Avg Generation (s) | Total Time (s) |"
        )
        rows.append(
            "|------|-------------|-----------|-----------|--------|--------|--------------------|--------------------|--------------------|"
        )

        rank_emojis = [
            "1️⃣",
            "2️⃣",
            "3️⃣",
            "4️⃣",
            "5️⃣",
            "6️⃣",
            "7️⃣",
            "8️⃣",
            "9️⃣",
            "🔟",
            "1️⃣1️⃣",
            "1️⃣2️⃣",
        ]

        for i, combo in enumerate(sorted_combos):
            rank_emoji = rank_emojis[i] if i < len(rank_emojis) else f"{i+1}"
            pass_rate_pct = combo.pass_rate * 100

            row = f"| {rank_emoji} | {combo.combination} | {pass_rate_pct:.1f}% | {combo.avg_score:.3f} | {combo.passed_tests}/{combo.total_tests} | {combo.failed_tests}/{combo.total_tests} | {combo.avg_retrieval_time:.2f} | {combo.avg_generation_time:.2f} | {combo.total_time:.1f} |"
            rows.append(row)

        return "\n".join(rows)

    def _generate_test_case_breakdown(
        self,
        test_cases: list[TestCase],
        test_results: list[TestResult],
    ) -> str:
        """Generate per-test-case breakdown."""
        sections = ["## 📝 Test Case Results\n"]

        for test_case in test_cases:
            # Get all results for this test case
            tc_results = [r for r in test_results if r.test_id == test_case.test_id]

            if not tc_results:
                continue

            passed_count = sum(1 for r in tc_results if r.passed)
            failed_count = len(tc_results) - passed_count

            # Find best and worst
            best = max(tc_results, key=lambda r: r.score)
            worst = min(tc_results, key=lambda r: r.score)

            sections.append(f"### {test_case.test_id}")
            sections.append(f"**Question**: {test_case.question}")
            sections.append(
                f"**Difficulty**: {test_case.difficulty} | **Scoring**: {test_case.scoring_type}"
            )
            sections.append(
                f"**Results**: {'✅' if passed_count > failed_count else '❌'} {passed_count}/{len(tc_results)} combinations passed\n"
            )
            sections.append(f"**Best**: {best.combination} (score: {best.score:.3f})")
            sections.append(f"**Worst**: {worst.combination} (score: {worst.score:.3f})")

            # Show common failures
            if failed_count > 0:
                failed_results = [r for r in tc_results if not r.passed]
                common_missing = self._find_common_missing_terms(failed_results)
                common_hallucinations = self._find_common_hallucinations(failed_results)

                if common_missing:
                    sections.append(f"\n**Common Missing Terms**: {', '.join(common_missing)}")
                if common_hallucinations:
                    sections.append(
                        f"**Common Hallucinations**: {', '.join(common_hallucinations)}"
                    )

            sections.append("\n---\n")

        return "\n".join(sections)

    def _generate_failure_analysis(
        self,
        test_results: list[TestResult],
        test_cases: list[TestCase],
    ) -> str:
        """Generate failure mode analysis."""
        sections = ["## 🔍 Failure Mode Analysis\n"]

        failed_results = [r for r in test_results if not r.passed]

        if not failed_results:
            sections.append(
                "✅ **No failures detected! All tests passed across all combinations.**"
            )
            return "\n".join(sections)

        # Analyze by failure mode
        failure_modes: dict[str, list[TestResult]] = {}
        for result in failed_results:
            # Find test case to get failure mode
            tc = next((tc for tc in test_cases if tc.test_id == result.test_id), None)
            if tc:
                mode = tc.failure_mode_tested or "Unknown"
                if mode not in failure_modes:
                    failure_modes[mode] = []
                failure_modes[mode].append(result)

        # Sort by frequency
        sorted_modes = sorted(failure_modes.items(), key=lambda x: len(x[1]), reverse=True)

        sections.append(
            f"**Total Failures**: {len(failed_results)}/{len(test_results)} ({len(failed_results)/len(test_results)*100:.1f}%)\n"
        )

        for mode, failures in sorted_modes:
            sections.append(f"### {mode}")
            sections.append(f"**Frequency**: {len(failures)} failures")

            # Most affected combinations
            combo_failures: dict[str, int] = {}
            for f in failures:
                combo_failures[f.combination] = combo_failures.get(f.combination, 0) + 1

            worst_combos = sorted(combo_failures.items(), key=lambda x: x[1], reverse=True)[:3]
            sections.append(
                f"**Most Affected**: {', '.join(f'{c} ({n}x)' for c, n in worst_combos)}"
            )
            sections.append("")

        return "\n".join(sections)

    def _generate_detailed_results(
        self,
        combination_summaries: dict[str, CombinationSummary],
    ) -> str:
        """Generate detailed results per combination."""
        sections = ["## 📋 Detailed Results by Combination\n"]

        # Sort by pass rate
        sorted_combos = sorted(
            combination_summaries.values(),
            key=lambda s: (s.pass_rate, s.avg_score),
            reverse=True,
        )

        for combo in sorted_combos:
            sections.append(f"### {combo.combination}")
            sections.append(
                f"**Parser**: {combo.parser_name} | **Storage**: {combo.storage_backend}"
            )
            sections.append(
                f"**Pass Rate**: {combo.pass_rate * 100:.1f}% | **Average Score**: {combo.avg_score:.3f}\n"
            )

            # Test results table
            sections.append(
                "| Test ID | Passed | Score | Required Coverage | Hallucinations |"
            )
            sections.append(
                "|---------|--------|-------|-------------------|----------------|"
            )

            for result in combo.test_results:
                passed_icon = "✅" if result.passed else "❌"
                hall_count = len(result.forbidden_terms_found)
                hall_text = f"⚠️ {hall_count}" if hall_count > 0 else "✓"

                row = f"| {result.test_id} | {passed_icon} | {result.score:.3f} | {result.required_term_coverage*100:.0f}% | {hall_text} |"
                sections.append(row)

            sections.append("\n---\n")

        return "\n".join(sections)

    def _find_common_missing_terms(self, failed_results: list[TestResult]) -> list[str]:
        """Find terms commonly missing across failed results."""
        if not failed_results:
            return []

        # Count missing terms
        term_counts: dict[str, int] = {}
        for result in failed_results:
            for term in result.required_terms_missing:
                term_counts[term] = term_counts.get(term, 0) + 1

        # Return terms missing in >50% of failures
        threshold = len(failed_results) * 0.5
        common = [term for term, count in term_counts.items() if count >= threshold]
        return sorted(common, key=lambda t: term_counts[t], reverse=True)[:5]

    def _find_common_hallucinations(self, failed_results: list[TestResult]) -> list[str]:
        """Find hallucinations commonly found across failed results."""
        if not failed_results:
            return []

        # Count hallucinations
        term_counts: dict[str, int] = {}
        for result in failed_results:
            for term in result.forbidden_terms_found:
                term_counts[term] = term_counts.get(term, 0) + 1

        # Return terms found in >30% of failures
        threshold = len(failed_results) * 0.3
        common = [term for term, count in term_counts.items() if count >= threshold]
        return sorted(common, key=lambda t: term_counts[t], reverse=True)[:5]
