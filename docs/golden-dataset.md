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

**v1** (current) = 30 questions, 41 total citations
- Baseline coverage across all five documents
- Planned expansion to **v2** (50 questions) in week 4 of development

Future versions will increase question density per document and add domain-specific question types (e.g., compliance gap analysis, control interaction graphs).

## How to Use

For retrieval evaluation:
```python
# Load dataset
with open("evals/golden/v1.jsonl") as f:
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
