# Golden Evaluation Dataset v1

## Purpose

The golden evaluation dataset provides 30 hand-curated questions and reference answers to evaluate the quality and coverage of a retrieval-augmented generation (RAG) system against NIST security and privacy standards. The corpus comprises 1,003 chunks extracted from five authoritative NIST documents totaling over 2,000 pages.

The dataset is designed to test both the retrieval capability (can the system find relevant chunks?) and the generation capability (can it synthesize an accurate answer?). Questions span single-fact lookups, multi-section synthesis, cross-document reasoning, and intentional near-misses.

## Schema

Each line in `evals/golden/v1.jsonl` is a JSON object with these exact fields:

- **id** (string): Question identifier in format `v1-qNN` (v1-q01 through v1-q30)
- **question** (string): Natural language question an analyst might ask about NIST requirements
- **reference_answer** (string): 1–4 sentence answer grounded entirely in cited chunks
- **source_citations** (array of objects): Citation objects with `doc` (document ID) and `section` (section identifier)
  - For lookup/synthesis/multihop: ≥1 citation
  - For unanswerable: empty array
- **difficulty** (string): `"easy"` | `"medium"` | `"hard"`
- **type** (string): `"lookup"` | `"synthesis"` | `"multihop"` | `"unanswerable"`
- **held_out** (boolean, optional, default false): marks items used verbatim as few-shot
  examples inside pipeline prompts (see "Held-out planner few-shot items" below).
  Generation evals exclude held-out items; retrieval evals keep them.

## Authoring Criteria

### Grounding Rule
Every question and reference answer is derived from the actual chunk text in `data/corpus/chunks.jsonl`. Before writing each Q/A pair, the author reads the supporting chunks (via grep + Read on the jsonl) and ensures:
- The question is answerable from the cited sections
- The reference answer quotes or closely paraphrases the chunk text
- No general knowledge is introduced that doesn't appear in the corpus

### Question Naturalness
Questions are framed as real analyst inquiries:
- "What does SP 800-53 control AC-2 require?" (lookup)
- "How do AC-2 and AC-10 work together?" (synthesis)
- "How does FIPS 199 categorization inform SP 800-53 baseline selection?" (multihop)

Not quiz-style text lookups ("What does the third sentence of …").

### Near-Miss Rule for Unanswerable (v1-q26..v1-q30)
Unanswerable questions are topically adjacent but outside the corpus:
- Specific password length thresholds (corpus addresses strength of mechanism, not numbers)
- Commercial product recommendations (corpus describes requirements, not tools)
- Cost estimates (corpus addresses controls, not budgets)
- Implementation timelines (corpus addresses requirements, not deadlines)
- International standard mappings (corpus does not include ISO/IEC 27001 cross-references)

These are plausible requests an analyst *could* make, but the corpus cannot answer them.

## Composition

- **v1-q01..v1-q10** (Lookup, 10 questions): Single-fact questions answerable from one section.
  - Mostly "easy" difficulty
  - Typically 1 citation per question
  - Example: "What does AC-2 require regarding account types?"

- **v1-q11..v1-q20** (Synthesis, 10 questions): Require combining 2–3 sections *within the same document*.
  - "Medium" difficulty
  - 2–3 citations per question
  - Example: "How do AC-1, AC-2, and AC-10 establish comprehensive access control?"

- **v1-q21..v1-q25** (Multihop, 5 questions): Require sections from *at least two different documents*.
  - "Hard" difficulty
  - 2–4 citations spanning 2+ documents
  - Example: "How does FIPS 199 categorization inform SP 800-53 baseline selection?"

- **v1-q26..v1-q30** (Unanswerable, 5 questions): Topically adjacent but the corpus cannot answer.
  - "Hard" difficulty
  - Zero citations
  - Reference answer states why the corpus cannot answer and what would be needed

## Coverage Matrix

Citation counts by document and question type (v1 = 30 questions, 41 total citations):

| Document   | Lookup | Synthesis | Multihop | Unanswerable | Total |
|------------|--------|-----------|----------|--------------|-------|
| sp800-53r5 |    8   |    17     |    4     |      0       |  29   |
| sp800-171r3|    1   |     2     |    1     |      0       |   4   |
| fips-199   |    1   |     0     |    2     |      0       |   3   |
| fips-200   |    0   |     0     |    2     |      0       |   2   |
| ai-rmf     |    0   |     2     |    1     |      0       |   3   |
| **Total**  | **10** |  **21**   |  **10**  |     **0**    | **41**|

All 5 documents are cited at least twice, meeting the minimum coverage requirement.

## Citation Matching

Citations are matched at the **section level** against chunk `section_ids`:
- A question with citation `{"doc": "sp800-53r5", "section": "AC-2"}` resolves to any chunk where `doc_id == "sp800-53r5"` and `"AC-2"` appears in the chunk's `section_ids` array.
- Some chunks span multiple sections (e.g., `section_ids: ["AC-2", "AC-3"]`); any matching section counts.
- For NIST documents with hierarchical sections (e.g., SP 800-171: "03.01.01"), section IDs are exact matches.

## Versioning

### v1 (baseline) = 30 questions, 41 total citations
- Baseline coverage across all five documents
- Lookup (10), Synthesis (10), Multihop (5), Unanswerable (5)

### v2 (current) = 50 questions, 69 total citations
- Expands v1 baseline with 20 new items targeting underrepresented documents and question types
- **ID Stability**: First 30 items (v1-q01 through v1-q30) are byte-for-byte identical to v1.jsonl; new items use v2-q31 through v2-q50 IDs, enabling v1 results to remain comparable across versions
- Lookup (16), Synthesis (14), Multihop (12), Unanswerable (8)

#### v2 Expansion Rationale

The 20 new items fill coverage gaps identified in v1:

1. **AI RMF expansion** (5 items: v2-q31 to v2-q35)
   - v1 had only 3 AI RMF citations; v2 adds 9 total (9 citations across 5 questions)
   - 2 lookup (MAP 1.1 intended purposes, MANAGE 1.3 risk responses)
   - 2 synthesis (GOVERN 1.4 + GOVERN 1.6, MAP vs MEASURE functions)
   - 1 multihop (AI RMF MANAGE + SP 800-53 AC-2)

2. **FIPS expansion** (4 items: v2-q36 to v2-q39)
   - v1 had 5 FIPS citations; v2 brings the total to 11
   - 3 lookup (FIPS-199 Integrity, FIPS-200 Accreditation, FIPS-200 Adequate Security)
   - 0 synthesis (the corpus's FIPS chunks concentrate in the appendix-a glossaries — too few
     distinct sections for within-document synthesis)
   - 1 multihop (FIPS-199 security objectives + SP 800-53 AC-2)

3. **SP 800-171r3 expansion** (4 items: v2-q40 to v2-q43)
   - v1 had 4 SP 800-171r3 citations; v2 brings the total to 12
   - 1 lookup (Requirement 3.2.1 security literacy training)
   - 2 synthesis (3.2.1 + 3.3.1 logging, 3.1.8 logon + 3.2.1 training)
   - 1 multihop (3.1.8 logon attempts + SP 800-53 AC-11 device lock)

4. **Cross-document multihop** (4 items: v2-q44 to v2-q47)
   - Expands multihop reasoning across document pairs not covered in v1
   - FIPS-200 + SP 800-171r3
   - AI RMF GOVERN + SP 800-53 CP-1
   - FIPS-199 + SP 800-171r3 training
   - AI RMF MAP + SP 800-53 SC-7 boundary protection

5. **Unanswerable items** (3 items: v2-q48 to v2-q50)
   - Complements v1's 5 unanswerable items with 3 additional near-miss questions
   - AI bias/fairness metrics (corpus covers evaluation but not specific thresholds)
   - Control baseline cost-benefit framework (corpus covers controls, not economics)
   - SP 800-53 revision timeline (corpus covers current version, not deprecation schedules)

#### v2 Coverage Matrix

Citation counts by document and question type (v2 = 50 questions, 69 total citations;
unanswerable questions carry no citations and are counted separately):

| Document   | Lookup | Synthesis | Multihop | Total |
|------------|--------|-----------|----------|-------|
| sp800-53r5 |    8   |    17     |    9     |  34   |
| sp800-171r3|    2   |     6     |    4     |  12   |
| fips-199   |    2   |     0     |    4     |   6   |
| fips-200   |    2   |     0     |    3     |   5   |
| ai-rmf     |    2   |     6     |    4     |  12   |
| **Total**  | **16** |  **29**   | **24**   | **69**|

Question counts by type: 16 lookup, 14 synthesis, 12 multihop, 8 unanswerable. All five
documents appear in both single-document and cross-document questions, with AI RMF and
SP 800-171r3 coverage brought closer to parity with SP 800-53r5.

#### Answerable vs. Unanswerable Split

- **v1**: 25 answerable, 5 unanswerable (83% / 17%)
- **v2**: 42 answerable, 8 unanswerable (84% / 16%)

The slight increase in unanswerable items maintains representation of near-miss queries an analyst might pose but the corpus cannot fully address.

#### Held-out planner few-shot items (week 5)

The agentic planner prompt (`planner.v1`) uses four golden questions verbatim as
few-shot examples: v1-q02 and v1-q06 (lookup, shown as `direct`) and v1-q21 and v1-q22
(multihop, shown decomposed as `multi_hop`). Scoring the pipeline on questions its own
prompt contains would be training on the test, so these four carry `"held_out": true`
and are excluded from generation eval runs — the week-5 comparative benchmark therefore
runs on 46 items (14 lookup, 14 synthesis, 10 multihop, 8 unanswerable). Retrieval
evals still include them: no retrieval prompt ever sees golden questions. Question
text, answers, and IDs are unchanged, so pre-week-5 results on the full 50 remain
interpretable; cross-week comparisons should use the 46-item intersection.

### Future Versions

Future versions will:
- Increase question density further, reaching 75–100 items
- Add domain-specific question types (e.g., compliance gap analysis, control interaction graphs, scenario-based reasoning)
- Expand coverage of emerging documents (e.g., newer NIST AI RMF guidance, NIST Cybersecurity Framework alignment)

## How to Use

For retrieval evaluation:
```python
# Load dataset (use v2.jsonl for latest, or v1.jsonl for baseline)
with open("evals/golden/v2.jsonl") as f:
    questions = [json.loads(line) for line in f]

# For each question, retrieve chunks and evaluate:
# - Does the system find all chunks in source_citations?
# - Does the reference answer text appear in or derive from those chunks?
```

For generation evaluation:
```python
# Compare model output against reference_answer
# Metrics: ROUGE, BERTScore, semantic similarity, factual grounding
```

For end-to-end RAG evaluation:
```python
# Query the full RAG system
# Check both retrieval (does it find the right chunks?) 
# and generation (does it synthesize an accurate answer?)
```

## Stability and Versioning for Continuous Evaluation

To support continuous evaluation while tracking improvements:

- **v1 and v2 are both stable datasets**: v2 appends 20 new items to the 30 v1 items without modification
- **ID Scheme preserves comparability**: v1-q01 through v1-q30 remain unchanged (byte-for-byte identical in v2.jsonl), so evaluation results on v1 questions are directly comparable across model versions
- **Upgrade path**: Metrics computed on v1 (30 questions) remain valid; results on v2 (50 questions) extend the evaluation with greater coverage
- **Example**: A model that scores 80% on v1-q01:q30 in one evaluation run can be compared directly to a later run; separately, aggregate scores on v2-q01:q50 track improvement over a larger question set
