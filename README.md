# RAG Retrieval Failure Analysis using SQuAD v2

A Retrieval-Augmented Generation (RAG) project developed as part of the CortiqoLabs Research & Solution Development Internship Take-Home Assignment.

The objective of this project is to build a simple end-to-end RAG pipeline and investigate how retrieval parameters such as **Chunk Size** and **Top-k Retrieval** influence answer quality.

---

# Project Overview

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline using:

- SQuAD v2 Dataset
- FAISS Vector Database
- Sentence Transformers (MiniLM)
- Groq Llama 3.1 8B Instant
- LangChain

Beyond building the baseline pipeline, experiments were performed to analyze retrieval failures and evaluate the impact of different retrieval configurations.

---

# Problem Statement

Large Language Models often hallucinate when answering questions outside their context.

Retrieval-Augmented Generation (RAG) reduces hallucination by retrieving relevant documents before generating an answer.

This project studies how different retrieval settings affect the quality of generated answers.

---

# Dataset

**Dataset:** SQuAD v2 (Stanford Question Answering Dataset)

The dataset consists of:

- Context passages
- Questions
- Ground truth answers

It is widely used for Question Answering and Retrieval research.

---

# Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python |
| Dataset | SQuAD v2 |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| LLM | Groq Llama 3.1 8B Instant |
| Framework | LangChain |

---

# Project Structure

```
.
├── src
│   ├── data_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── generator.py
│   ├── rag_pipeline.py
│   ├── evaluator.py
│   └── experiments.py
│
├── vector_db/
├── results/
├── report.pdf
├── requirements.txt
├── main.py
└── README.md
```

---

# Pipeline

```
SQuAD v2 Dataset
        │
        ▼
Load Context Documents
        │
        ▼
Text Chunking
(RecursiveCharacterTextSplitter)
        │
        ▼
Sentence Embeddings
(all-MiniLM-L6-v2)
        │
        ▼
FAISS Vector Database
        │
        ▼
User Question
        │
        ▼
Top-k Retrieval
        │
        ▼
Groq Llama 3.1
        │
        ▼
Generated Answer
```

---

# Features

- End-to-End RAG Pipeline
- FAISS Vector Search
- Semantic Retrieval
- LLM-based Answer Generation
- Automatic Evaluation
- Chunk Size Experiments
- Top-k Retrieval Experiments
- Retrieval Failure Analysis

---

# Installation

Clone the repository

```bash
git clone <your-repository-url>

cd <repository-name>
```

Create Virtual Environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env`

```env
GROQ_API_KEY=YOUR_API_KEY
```

---

# Running the Project

Build the Vector Database

```bash
python main.py
```

Choose the required option from the menu.

Example:

```
1 -> Interactive RAG

2 -> Baseline Evaluation

3 -> Chunk Size Experiment

4 -> Top-k Experiment
```

---

# Experiments

## Experiment 1 – Chunk Size

Chunk sizes evaluated:

- 300
- 500
- 800

Sample Results

| Chunk Size | Questions | Accuracy |
|------------|----------:|----------:|
| 300 | 20 | 60% |
| 500 | 20 | 70% |
| 800 | 20 | 50% |

### Observation

A chunk size of **500** produced the best retrieval quality by balancing context completeness and retrieval precision.

---

## Experiment 2 – Top-k Retrieval

Top-k values evaluated:

- Top-1
- Top-3
- Top-5

Sample Results

| Top-k | Questions | Accuracy |
|-------:|----------:|----------:|
| 1 | 20 | 50% |
| 3 | 20 | 70% |
| 5 | 20 | 60% |

### Observation

Retrieving the **Top-3** documents produced the best results. Increasing Top-k beyond three introduced additional irrelevant context.

---

# Example Outputs

### Successful Retrieval

**Question**

```
Who is Beyonce married to?
```

**Retrieved Context**

```
The couple are known for their private relationship...

Beyoncé and Jay Z traveled to Paris...
```

**Generated Answer**

```
Jay Z
```

---

### Retrieval Failure

**Question**

```
Who were the Normans?
```

**Retrieved Context**

```
Native American, French, Cajun...

Native American, French, Cajun...
```

**Generated Answer**

```
I don't know.
```

### Analysis

The retriever returned unrelated passages instead of relevant Norman history, demonstrating that retrieval quality is the primary bottleneck in the pipeline.

---

# Key Findings

- Retrieval quality has a greater impact on answer quality than the language model itself.
- Chunk Size significantly affects retrieval effectiveness.
- Top-k retrieval requires balancing relevance and context coverage.
- Retrieval failures were the main cause of incorrect answers.

---

# Limitations

- Limited evaluation set (20 questions).
- Simple string matching used for evaluation.
- Single embedding model.
- Single LLM configuration.

---

# Future Work

- Evaluate BGE and E5 embedding models.
- Compare Gemini, Mistral, and Qwen models.
- Introduce Hybrid Search (BM25 + Dense Retrieval).
- Add Cross-Encoder Reranking.
- Experiment with Semantic Chunking.
- Use Exact Match and F1 metrics for evaluation.

---

# Acknowledgement

For transparency, AI-assisted tools were used to optimize portions of the implementation and assist with documentation formatting. All experiments, implementation decisions, debugging, and validation were performed manually to ensure correctness and understanding.

---

# Author

**Roshan**

CortiqoLabs Research & Solution Development Internship Assignment