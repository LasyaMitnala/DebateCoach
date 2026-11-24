# DebateCoach — Project Specification

**Version:** 0.1.0  
**Development Phase:** Phase 0 — Foundations  
**Status:** Initial specification  
**Project Type:** AI / Machine Learning / Natural Language Processing

---

## 1. Project Overview

### 1.1 Project Description

DebateCoach is an evidence-grounded, AI-powered debate coaching platform designed to help users develop stronger arguments, recognize logical weaknesses, construct effective rebuttals, and improve their debating skills.

Unlike conventional AI chatbots that generate counterarguments primarily from learned knowledge, DebateCoach combines specialized machine-learning models with information retrieval, evidence verification, and structured feedback.

The system will analyze arguments, retrieve relevant counterarguments and research, generate evidence-supported rebuttals, and track user improvement.

### 1.2 Problem Statement

Existing AI-based debate assistance tools may suffer from several limitations:

- **Hallucinated evidence:** Generated rebuttals may contain fabricated studies, statistics, or citations.
- **Insufficient argument analysis:** Systems may generate opposing viewpoints without identifying the underlying argumentative structure.
- **Inconsistent feedback:** General-purpose language models may assign subjective or inconsistent argument scores.
- **Limited transparency:** Users may not understand why their argument is weak or how a rebuttal was constructed.
- **Limited progress tracking:** Users may lack measurable feedback on their development over time.

### 1.3 Proposed Solution

DebateCoach will use a modular NLP architecture consisting of specialized models for argument parsing, fallacy detection, counterargument retrieval, evidence verification, rebuttal generation, and quality assessment.

Each component will be evaluated independently against baseline approaches.

**Core principle:** Factual claims in generated rebuttals must be grounded in retrieved, verifiable evidence rather than unsupported model-generated assertions.

---

## 2. Target Users

### Primary Users

- High school and college competitive debaters
- Students studying argumentation, rhetoric, and critical thinking
- Speech and debate club members
- Students preparing argumentative essays or presentations

### Secondary Users

- Debate coaches and educators
- Researchers studying computational argumentation
- Individuals seeking structured practice in evaluating arguments

### Primary Use Cases

**UC-01: Argument Analysis**

A user submits a written argument and receives an explanation of its claims, premises, logical weaknesses, and argumentative structure.

**UC-02: Rebuttal Generation**

A user submits an argument and requests a counterargument supported by verifiable research.

**UC-03: Interactive Debate Practice**

A user selects a topic and stance, participates in multiple debate rounds, and receives feedback.

**UC-04: Argument Improvement**

A user receives suggestions for strengthening clarity, reasoning, evidence, and responsiveness.

**UC-05: Progress Tracking**

A user reviews previous debate sessions, scores, frequently identified weaknesses, and improvements over time.

---

## 3. Functional Requirements

### FR-01: Argument Parsing

The system must identify argumentative components, including claims and premises, and determine their relationships.

**Input:**
- Argument text
- Optional debate topic
- Optional stance

**Output:**
- Identified claims
- Identified premises
- Text spans or sentence positions
- Support or attack relationships
- Model confidence scores

**Proposed model:** Fine-tuned DeBERTa-v3 or RoBERTa encoder.

**Training resources:** Persuasive Essays corpus and IBM Project Debater datasets.

### FR-02: Logical Fallacy Detection

The system must identify potential logical fallacies and explain how they affect the argument.

**Input:**
- Argument text
- Parsed claims and premises
- Optional topic context

**Output:**
- Fallacy category
- Relevant text span
- Explanation
- Confidence score
- Suggested improvement

**Initial proposed taxonomy:**

1. Ad hominem
2. Straw man
3. False dilemma
4. Hasty generalization
5. Slippery slope
6. Circular reasoning
7. Appeal to authority
8. False cause
9. Appeal to emotion
10. Red herring

A separate `none/uncertain` outcome must be supported. The taxonomy will be revised after examining available dataset labels.

**Proposed model:** Fine-tuned DeBERTa-v3 classifier.

**Training resources:** LOGIC, LOGIC Climate, MAFALDA, and Argotario.

### FR-03: Counterargument Retrieval

The system must retrieve relevant opposing arguments from indexed argument corpora.

**Input:**
- User argument
- Topic
- Stance
- Optional extracted claims

**Output:**
- Ranked counterarguments
- Source identifiers
- Relevance scores
- Opposing stance labels

**Proposed architecture:**
- Dense semantic embeddings
- BM25 keyword retrieval
- Hybrid ranking
- Cross-encoder reranking

**Training resources:** ArguAna, Args.me, and Touché benchmarks.

### FR-04: Evidence Retrieval

The system must retrieve scholarly research and public statistical evidence relevant to a claim.

**Input:**
- Claim text
- Topic
- Search query or generated query set

**Output:**
- Relevant research documents
- Evidence passages
- Titles and authors
- Publication dates
- DOI or source URL
- Retrieval relevance scores

**Planned sources:**
- OpenAlex
- Semantic Scholar
- PubMed
- Crossref
- Government statistical databases

Evidence retrieval must respect source licensing, API rate limits, and redistribution requirements.

### FR-05: Evidence Verification

The system must evaluate whether retrieved evidence supports, refutes, or provides insufficient information about a claim.

**Input:**
- Claim
- Retrieved evidence passage
- Source metadata

**Output:**
- `SUPPORTS`
- `REFUTES`
- `NOT_ENOUGH_INFO`
- Confidence score
- Supporting or contradicting rationale
- Citation verification status

**Proposed model:** Natural language inference encoder, initially pretrained and subsequently fine-tuned using SciFact and FEVER.

**Requirements:**
- Verify citation identifiers and source URLs.
- Check that cited passages exist in retrieved material.
- Distinguish source existence from actual evidentiary support.
- Flag uncertain or unverifiable evidence.
- Avoid presenting unverified evidence as established fact.

### FR-06: Rebuttal Generation

The system must generate a coherent counterargument grounded in retrieved and verified evidence.

**Input:**
- Original argument
- Parsed claims and premises
- Identified weaknesses
- Retrieved counterarguments
- Verified evidence passages
- Requested stance

**Output:**
- Main counterclaim
- Supporting reasoning
- Evidence references
- Anticipated response
- Confidence or uncertainty indicators

**Development stages:**

**Tier A — Baseline:** Hosted language model using retrieval-augmented generation and structured output.

**Tier B — Fine-tuned model:** Open-weight language model fine-tuned using LoRA or QLoRA.

**Tier C — Preference optimization:** Direct Preference Optimization using appropriately collected preference data.

Every generated citation must correspond to a retrieved source identifier. Unsupported factual assertions must be removed, qualified, or regenerated.

### FR-07: Argument Quality Scoring

The system must assess arguments across defined dimensions and generate actionable feedback.

**Input:**
- Argument text
- Argument structure
- Evidence assessment
- Debate context
- Optional opposing argument

**Output:**
- Dimension scores
- Overall score
- Strengths
- Weaknesses
- Improvement recommendations

**Initial proposed scoring rubric:**

| Dimension | Weight |
|---|---:|
| Clarity | 20% |
| Logical reasoning | 25% |
| Evidence quality | 25% |
| Responsiveness | 20% |
| Argument structure | 10% |
| **Total** | **100%** |

Each dimension will be scored from 0–10. The overall score will be a weighted average converted to a 0–100 scale.

These weights are provisional and will require calibration against human judgments. Responsiveness will be marked not applicable when no opposing argument is supplied, with remaining weights normalized.

**Training resources:** Dagstuhl ArgQuality corpus and IBM argument quality datasets.

### FR-08: Debate History and Progress Tracking

The system should store user-approved debate sessions and feedback for longitudinal analysis.

**Input:**
- Session identifier
- User identifier, when applicable
- Argument and rebuttal records
- Feedback scores
- Timestamp

**Output:**
- Previous sessions
- Score trends
- Fallacy frequency
- Improvement summaries

Users must be informed about data retention and have a mechanism to delete stored debate records.

---

## 4. System Architecture

### 4.1 Processing Pipeline

```text
User Argument
     |
     v
Argument Parsing
     |
     v
Weakness / Fallacy Detection
     |
     v
Counterargument Retrieval
     |
     v
Evidence Retrieval
     |
     v
Evidence Verification
     |
     v
Rebuttal Generation
     |
     v
Argument Scoring & Feedback
     |
     v
User Interface / Saved History
```

Each stage must expose a defined input/output interface to support independent testing, model replacement, and error analysis.

### 4.2 Proposed Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| ML framework | PyTorch |
| NLP models | Hugging Face Transformers |
| Embeddings | sentence-transformers |
| Fine-tuning | PEFT / TRL |
| Experiment tracking | MLflow or Weights & Biases |
| Backend | FastAPI |
| Frontend | React |
| Relational storage | PostgreSQL |
| Vector search | pgvector or FAISS |
| Testing | pytest |
| Version control | Git / GitHub |
| Deployment | Docker and cloud hosting |

Technology selections may change based on experimental performance, hardware requirements, and deployment cost.

---

## 5. Data Specifications

### 5.1 Standardized Argument Record

Proposed JSON representation:

```json
{
  "argument_id": "arg_001",
  "topic": "School homework policies",
  "stance": "opposition",
  "text": "Homework should be banned
