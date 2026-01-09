# RAG System Test Cases: Handbook of Petroleum Refining Processes

## Overview

This document provides test cases for validating a RAG (Retrieval-Augmented Generation) system processing the **McGraw-Hill Handbook of Petroleum Refining Processes** (2nd Edition, Robert A. Meyers, Editor).

**Source Document**: `Handbook of Petroleum Refining Processes.pdf`  
**Document Size**: ~800+ pages, 56 chapters, 14 parts  
**Content Types**: Technical prose, process flow diagrams, reaction equations, data tables, economic analyses

---

## Coding Agent Prompt

```
You are building test cases for a RAG system that processes petroleum refinery technical manuals. 

For each test case below, implement:
1. A function that queries the RAG system with the provided question
2. A validation function that compares the RAG response against the ground truth
3. Scoring logic that accounts for:
   - Exact numerical matches (when applicable)
   - Semantic similarity for descriptive answers
   - Presence of required key terms
   - Absence of hallucinated information

Use the following test case structure:

class RAGTestCase:
    id: str
    question: str
    ground_truth: str
    required_terms: list[str]
    forbidden_terms: list[str]  # Terms that indicate hallucination
    source_location: str
    difficulty: str  # "easy", "medium", "hard", "expert"
    failure_mode_tested: str
    scoring_type: str  # "exact", "semantic", "numerical", "multi-part"
    
Implement pytest fixtures and parametrized tests for batch execution.
```

---

## Test Cases

### Test Case 1: Specific Numerical Data Retrieval

**ID**: `TC001_ALKYLATION_ECONOMICS`

**Question**: 
> What is the estimated capital cost range per barrel per day for a 7,500 BPSD Exxon stirred autorefrigerated alkylation plant on the U.S. Gulf Coast?

**Ground Truth**:
> The total capital required for a plant erected in 1995 on the U.S. Gulf Coast varies from $60 to $85 per metric ton per year ($2,500 to $3,500 per barrel per day) of C5+ polymer gasoline.

**Required Terms**: `["$2,500", "$3,500", "barrel per day", "U.S. Gulf Coast"]`

**Forbidden Terms**: `["$1,000", "$5,000", "HF alkylation"]` 

**Source Location**: Chapter 1.1 - Exxon Sulfuric Acid Alkylation Technology, Economics section

**Difficulty**: `medium`

**Failure Mode Tested**: Numerical precision retrieval - tests if system can extract specific dollar figures from economic tables

**Scoring Type**: `numerical`

---

### Test Case 2: Process Chemistry Understanding

**ID**: `TC002_ALKYLATION_REACTIONS`

**Question**:
> What are the four types of secondary reactions that reduce octane in alkylation, and what is an example equation for the polymerization reaction?

**Ground Truth**:
> The four secondary reactions are:
> 1. Polymerization (e.g., 2C₄H₈ → C₈H₁₆)
> 2. Hydrogen transfer (e.g., 2C₄H₁₀ + C₃H₆ → C₈H₁₈ + C₃H₈)
> 3. Disproportionation (e.g., 2C₈H₁₈ → C₇H₁₆ + C₉H₂₀)
> 4. Cracking (e.g., C₁₂H₂₆ → C₄H₈ + C₈H₁₈)
>
> These secondary reactions produce a wide spectrum of compounds that tend to reduce the octane to about the 96 octane level from the theoretical ~100 octane of pure trimethylpentane.

**Required Terms**: `["polymerization", "hydrogen transfer", "disproportionation", "cracking", "96"]`

**Forbidden Terms**: `["isomerization as secondary"]` (isomerization is a separate category, not secondary)

**Source Location**: Chapter 1.1 - Exxon Sulfuric Acid Alkylation Technology, Chemistry Overview section

**Difficulty**: `hard`

**Failure Mode Tested**: Multi-part answer synthesis - tests if system can enumerate a complete list and provide supporting detail

**Scoring Type**: `multi-part`

---

### Test Case 3: Comparative Analysis

**ID**: `TC003_REACTOR_COOLING_COMPARISON`

**Question**:
> Why is autorefrigeration more energy-efficient than indirect effluent refrigeration for alkylation reactor cooling?

**Ground Truth**:
> Autorefrigeration is more efficient because:
> 1. Zero temperature difference between reaction mass and refrigerant (direct vaporization from reaction mass)
> 2. Indirect effluent refrigeration requires a finite temperature difference for heat transfer across tubes, necessitating lower refrigeration temperatures
> 3. Lower refrigerant temperature requires lower compressor suction pressure, resulting in higher compressor energy requirements
> 4. Effluent refrigeration requires additional mixing power to overcome pressure drop across heat-transfer tubes
> 5. The additional heat of mixing must also be removed, increasing refrigeration duty

**Required Terms**: `["temperature difference", "compressor", "mixing power", "pressure drop"]`

**Forbidden Terms**: `["autorefrigeration is less efficient", "effluent refrigeration preferred"]`

**Source Location**: Chapter 1.1 - "Reactor Cooling via Autorefrigeration Is More Efficient Than Effluent Refrigeration" section

**Difficulty**: `hard`

**Failure Mode Tested**: Reasoning chain retrieval - tests if system retrieves complete causal explanation rather than partial answer

**Scoring Type**: `semantic`

---

### Test Case 4: Process Parameter Lookup

**ID**: `TC004_CUMENE_PRODUCT_SPECS`

**Question**:
> What are the product purity specifications for cumene produced by the Dow-Kellogg Cumene Process?

**Ground Truth**:
> - Cumene purity: minimum 99.97 wt%
> - Bromine index: maximum 5
> - Ethylbenzene: maximum 100 ppm
> - n-Propylbenzene: maximum 200 ppm
> - Butylbenzene: maximum 100 ppm

**Required Terms**: `["99.97", "bromine index", "5", "ppm"]`

**Forbidden Terms**: `["99.5%", "99.0%"]` (these would indicate retrieval from wrong process)

**Source Location**: Chapter 1.2 - The Dow-Kellogg Cumene Process, Table 1.2.2

**Difficulty**: `medium`

**Failure Mode Tested**: Table data extraction - tests if system correctly parses tabular specifications

**Scoring Type**: `exact`

---

### Test Case 5: Process Flow Understanding

**ID**: `TC005_CATALYTIC_CONDENSATION_FLOW`

**Question**:
> In the UOP Catalytic Condensation process for gasoline production, what is the purpose of the depropanizer and what happens to the propane overhead stream?

**Ground Truth**:
> The depropanizer removes propane that enters the unit with the feed to prevent propane buildup in the alkylation plant. A small slipstream of refrigerant is depropanized after being caustic- and water-washed to remove any SO₂. The propane overhead is sent to storage, while the isobutane-rich bottoms are returned to the process.

**Required Terms**: `["propane", "buildup", "caustic", "water-washed", "SO₂", "storage", "isobutane", "bottoms"]`

**Forbidden Terms**: `["propane recycled to reactor", "propane flared"]`

**Source Location**: Chapter 1.3 - UOP Catalytic Condensation Process, Process Description section

**Difficulty**: `medium`

**Failure Mode Tested**: Process flow comprehension - tests if system understands material flow and purpose

**Scoring Type**: `semantic`

---

### Test Case 6: Cross-Reference Question

**ID**: `TC006_FEED_CONTAMINANTS`

**Question**:
> How much additional makeup sulfuric acid is required per pound of mercaptan sulfur in the alkylation feed, and why is feed pretreatment necessary?

**Ground Truth**:
> Roughly 40 pounds of additional makeup acid are needed for each pound of mercaptan sulfur that enters the plant with the feed. This is because mercaptan sulfur reacts with sulfuric acid to form sulfonic acid and water (acid-soluble compounds) plus sulfur dioxide: RSH + 3H₂SO₄ → RSO₃H + 3H₂O + 3SO₂. The acid-soluble compounds must be purged from the plant with spent acid, increasing makeup requirements. Caustic treating facilities are always provided to remove sulfur from the olefin feed.

**Required Terms**: `["40 pounds", "mercaptan", "sulfonic acid", "caustic treating"]`

**Forbidden Terms**: `["10 pounds", "100 pounds"]`

**Source Location**: Chapter 1.1 - Exxon Sulfuric Acid Alkylation Technology, "Feed Impurities Form Acid-Soluble Compounds" section

**Difficulty**: `hard`

**Failure Mode Tested**: Quantitative relationship with chemical explanation - tests combined retrieval of number + mechanism

**Scoring Type**: `numerical` + `semantic`

---

### Test Case 7: Equipment Design Specifics

**ID**: `TC007_REACTOR_DIFFERENCES`

**Question**:
> What are the key design differences between modern ER&E-designed autorefrigeration reactors and older industry designs, specifically regarding acid-to-hydrocarbon ratio and space velocity?

**Ground Truth**:
> | Parameter | Older Industry Design | Modern ER&E Design |
> |-----------|----------------------|-------------------|
> | Acid/hydrocarbon mixing | Submerged pumps | Special mixers |
> | Isobutane/olefin premixed | No | Yes |
> | Olefin feed injectors | Open pipe | Special nozzles |
> | Acid-to-hydrocarbon ratio | <0.5 | ~1.5 |
> | Space velocity | 0.3 | 0.1 |
> | Pressure controllers per reactor | Up to 10 | 2 |

**Required Terms**: `["0.1", "0.3", "1.5", "0.5", "special mixers", "special nozzles"]`

**Forbidden Terms**: `["identical design", "no difference"]`

**Source Location**: Chapter 1.1 - Table 1.1.1, "Modern ER&E Reactor Is a Vast Improvement over Older System" section

**Difficulty**: `expert`

**Failure Mode Tested**: Comparative table extraction - tests if system can retrieve and contrast values from comparison tables

**Scoring Type**: `exact`

---

### Test Case 8: Yield Data Extraction

**ID**: `TC008_DISTILLATE_YIELDS`

**Question**:
> For the UOP Catalytic Condensation process in distillate production mode using FCC C₃-C₄ LPG feed with 59 LV% olefin content, what are the product yields in liquid volume percent?

**Ground Truth**:
> - C₃-C₄ LPG: 41.2 LV%
> - Polymer gasoline: 8.3 LV%
> - Distillate: 34.0 LV%
> - Heavy polymer: 0.8 LV%
> - Total: 84.3 LV%
>
> Note: Overall liquid yield is 84.3 LV% because net volume loss occurs as a result of the oligomerization of olefins.

**Required Terms**: `["41.2", "8.3", "34.0", "0.8", "84.3", "volume loss"]`

**Forbidden Terms**: `["100%", "95%"]`

**Source Location**: Chapter 1.3 - Table 1.3.4, "Yields and Product Properties" section

**Difficulty**: `medium`

**Failure Mode Tested**: Multi-row table extraction with context - tests if system retrieves complete yield breakdown

**Scoring Type**: `numerical`

---

### Test Case 9: Process Integration Question

**ID**: `TC009_HF_ALKYLATION_FLOW`

**Question**:
> In a UOP HF Alkylation unit processing C₃-C₄ olefins, describe the path of the isostripper overhead stream and explain the purpose of the HF stripper.

**Ground Truth**:
> The isostripper overhead consists mainly of isobutane, propane, and HF acid. A drag stream of overhead material is charged to the HF stripper to strip the acid. The overhead from the HF stripper is returned to the isostripper overhead system to recover acid and isobutane. A portion of the HF stripper bottoms is used as flushing material. A net bottom stream is withdrawn, defluorinated, and charged to the gas-concentration section (C₃-C₄ splitter) to prevent buildup of propane in the HF Alkylation unit.

**Required Terms**: `["isobutane", "propane", "HF acid", "drag stream", "defluorinated", "flushing"]`

**Forbidden Terms**: `["HF vented", "acid disposal"]`

**Source Location**: Chapter 1.4 - UOP HF Alkylation Technology, Process Description section

**Difficulty**: `hard`

**Failure Mode Tested**: Sequential process flow - tests if system can trace material through multiple unit operations

**Scoring Type**: `semantic`

---

### Test Case 10: Negative/Contradictory Test

**ID**: `TC010_CATALYST_DISPOSAL`

**Question**:
> Does the Dow-Kellogg Cumene Process produce environmentally difficult effluents, and what are the catalyst disposal requirements?

**Ground Truth**:
> No, the Dow-Kellogg Cumene Process produces no environmentally difficult effluents. The zeolite catalyst (3DDM) is very sturdy and regenerable, so there is no catalyst disposal concern as in other commercial processes. Spent catalyst is benign and requires no special disposal considerations other than normal landfill. The only emissions are normal stack emissions from heaters (hot-oil system), BFW blowdown from steam generators, and vacuum jet or vacuum pump vents.

**Required Terms**: `["no environmentally difficult effluents", "benign", "normal landfill", "regenerable"]`

**Forbidden Terms**: `["hazardous waste", "special disposal required", "acid disposal"]`

**Source Location**: Chapter 1.2 - "Wastes and Emissions" section and "Process Features" section

**Difficulty**: `medium`

**Failure Mode Tested**: Negative assertion retrieval - tests if system correctly identifies what a process does NOT produce

**Scoring Type**: `semantic`

---

## Test Case Implementation Template

```python
import pytest
from dataclasses import dataclass
from typing import List, Optional
import re

@dataclass
class RAGTestCase:
    id: str
    question: str
    ground_truth: str
    required_terms: List[str]
    forbidden_terms: List[str]
    source_location: str
    difficulty: str
    failure_mode_tested: str
    scoring_type: str
    
    def validate_response(self, response: str) -> dict:
        """Validate RAG response against ground truth."""
        results = {
            "id": self.id,
            "passed": False,
            "required_terms_found": [],
            "required_terms_missing": [],
            "forbidden_terms_found": [],
            "scores": {}
        }
        
        response_lower = response.lower()
        
        # Check required terms
        for term in self.required_terms:
            if term.lower() in response_lower:
                results["required_terms_found"].append(term)
            else:
                results["required_terms_missing"].append(term)
        
        # Check forbidden terms (hallucination detection)
        for term in self.forbidden_terms:
            if term.lower() in response_lower:
                results["forbidden_terms_found"].append(term)
        
        # Calculate scores
        required_score = len(results["required_terms_found"]) / len(self.required_terms) if self.required_terms else 1.0
        hallucination_penalty = len(results["forbidden_terms_found"]) * 0.2
        
        results["scores"]["required_term_coverage"] = required_score
        results["scores"]["hallucination_penalty"] = hallucination_penalty
        results["scores"]["final_score"] = max(0, required_score - hallucination_penalty)
        
        # Pass threshold
        results["passed"] = (
            results["scores"]["final_score"] >= 0.7 and
            len(results["forbidden_terms_found"]) == 0
        )
        
        return results


# Test cases data
TEST_CASES = [
    RAGTestCase(
        id="TC001_ALKYLATION_ECONOMICS",
        question="What is the estimated capital cost range per barrel per day for a 7,500 BPSD Exxon stirred autorefrigerated alkylation plant on the U.S. Gulf Coast?",
        ground_truth="The total capital required varies from $2,500 to $3,500 per barrel per day of C5+ polymer gasoline (or $60 to $85 per metric ton per year), based on 1995 U.S. Gulf Coast construction.",
        required_terms=["$2,500", "$3,500", "barrel per day"],
        forbidden_terms=["$1,000", "$5,000", "HF alkylation"],
        source_location="Chapter 1.1 - Economics section",
        difficulty="medium",
        failure_mode_tested="Numerical precision retrieval",
        scoring_type="numerical"
    ),
    # Add remaining test cases...
]


@pytest.fixture
def rag_system():
    """Initialize your RAG system here."""
    # Replace with your actual RAG system initialization
    from your_rag_module import RAGSystem
    return RAGSystem(index_path="petroleum_refining_index")


@pytest.mark.parametrize("test_case", TEST_CASES, ids=[tc.id for tc in TEST_CASES])
def test_rag_accuracy(rag_system, test_case: RAGTestCase):
    """Test RAG system accuracy against ground truth."""
    response = rag_system.query(test_case.question)
    results = test_case.validate_response(response)
    
    assert results["passed"], (
        f"Test {test_case.id} failed:\n"
        f"  Missing terms: {results['required_terms_missing']}\n"
        f"  Hallucinations: {results['forbidden_terms_found']}\n"
        f"  Score: {results['scores']['final_score']:.2f}"
    )


def test_numerical_extraction(rag_system):
    """Specific test for numerical data extraction accuracy."""
    numerical_cases = [tc for tc in TEST_CASES if tc.scoring_type == "numerical"]
    
    for tc in numerical_cases:
        response = rag_system.query(tc.question)
        # Extract numbers from response
        numbers_in_response = re.findall(r'\$?[\d,]+\.?\d*', response)
        numbers_in_ground_truth = re.findall(r'\$?[\d,]+\.?\d*', tc.ground_truth)
        
        # Check if key numbers are present
        for num in numbers_in_ground_truth:
            assert num in numbers_in_response, f"Missing number {num} in response for {tc.id}"
```

---

## Evaluation Metrics

### Scoring Rubric

| Metric | Weight | Description |
|--------|--------|-------------|
| Required Term Coverage | 40% | Percentage of required terms present in response |
| Numerical Accuracy | 25% | Exact match of numerical values (±5% tolerance for calculations) |
| Hallucination Penalty | -20% per | Deduction for each forbidden term found |
| Completeness | 15% | Multi-part answers: percentage of parts addressed |

### Pass/Fail Criteria

- **Pass**: Final score ≥ 0.70 AND zero forbidden terms
- **Marginal**: Final score 0.50-0.69 OR one forbidden term
- **Fail**: Final score < 0.50 OR multiple forbidden terms

### Difficulty-Weighted Scoring

```python
DIFFICULTY_WEIGHTS = {
    "easy": 1.0,
    "medium": 1.5,
    "hard": 2.0,
    "expert": 3.0
}

def calculate_weighted_score(test_results: List[dict], test_cases: List[RAGTestCase]) -> float:
    total_weighted = 0
    max_weighted = 0
    
    for result, tc in zip(test_results, test_cases):
        weight = DIFFICULTY_WEIGHTS[tc.difficulty]
        total_weighted += result["scores"]["final_score"] * weight
        max_weighted += weight
    
    return total_weighted / max_weighted if max_weighted > 0 else 0
```

---

## Expected Failure Modes

Document these when tests fail to diagnose RAG system issues:

1. **Chunk boundary problems**: Answer spans multiple chunks, retriever only gets partial info
2. **Table parsing failures**: Numerical data in tables not correctly extracted
3. **Similar content confusion**: Wrong chapter retrieved due to similar terminology
4. **Negation mishandling**: System misses "no" or "not" qualifiers
5. **Unit conversion errors**: System confuses $/BBL with $/MTA
6. **Cross-reference gaps**: Information requiring synthesis from multiple sections

---

## Notes for Test Execution

1. **Chunk Size Sensitivity**: Run tests with different chunk sizes (512, 1024, 2048 tokens) to identify optimal settings
2. **Embedding Model Comparison**: Test with different embedding models to compare retrieval accuracy
3. **Top-K Variation**: Test with k=3, k=5, k=10 retrieved chunks
4. **Reranking Impact**: Compare results with and without reranking step