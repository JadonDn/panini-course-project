# RUNME.md

# ECE 232E Project 3 — Reproduction Instructions

This document describes how to reproduce the final experiment in Google Colab.

## Environment

The final run was performed on:

- Python: 3.13.15
- OS: Linux 6.6.122+, x86\_64, glibc 2.35
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- GPU: NVIDIA A100-SXM4-40GB
- Transformers: 5.15.0
- Accelerate: 1.14.0
- bitsandbytes: 0.50.1
- scikit-learn: 1.6.1
- FAISS CPU: 1.15.0
- rank-bm25: 0.2.2
- pandas: 2.2.3
- numpy: 2.1.3

The exact package environment is recorded separately in `environment.txt`.

## Repository

Clone the project repository:

```bash
git clone https://github.com/JadonDn/panini-course-project.git
cd panini-course-project
```

Record the commit used for the final submission:

```bash
git rev-parse HEAD
```

The final submission commit hash is:

```text
5023d75471fd5690c88e7afbebaa56957ecdf9a0
```

Before running the notebook, verify that the working tree is clean:

```bash
git status
```

The reproduction should use the exact commit recorded above.

## Google Colab setup

Open the project notebook in Google Colab and select a GPU runtime.

The final run used an NVIDIA A100-SXM4-40GB.

## Execution controls

The notebook separates the neural stages so that two Qwen models are never kept in GPU memory simultaneously.

During development, use:

```python
QUESTION_LIMIT = 2
```

For the final experiment, use:

```python
QUESTION_LIMIT = None
```

The final run uses both datasets:

```python
DATASETS = ("2wiki", "musique")
```

The final frozen configuration is:

```text
BEAM_WIDTH = 5
CANDIDATES_PER_HOP = 15
RETRIEVAL_POOL = 60
RRF_CONSTANT = 60.0
MULTI_PARENT_THRESHOLD = 0.3
retrieval_backend = dual_hybrid
```

These settings were selected using the 2Wiki development data and then frozen before the final run.

## Models

The final model identifiers are:

```text
Decomposer:
yigitturali/GSW-QA-Decomposer-Qwen3-4B

Reranker:
Qwen/Qwen3-Reranker-8B

Answerer:
Qwen/Qwen3-4B
```

The supplied Qwen3-Embedding-8B corpus embeddings and query vectors are used as provided. Embeddings are **not regenerated** during reproduction.

## Restart points

The experiment is designed to survive Colab disconnections.

Do **not** delete the persistent cache/output directory when restarting.

Restart according to the last completed stage:

```text
Stage A incomplete
    -> rerun Stage A

Stage A complete, Stage B incomplete
    -> disable Stage A
    -> rerun Stage B

Stage B complete, Stage C incomplete
    -> disable Stages A/B
    -> rerun Stage C

Q9 RICR traces complete, Q9 answers incomplete
    -> rerun the Q9 answer-generation portion
```

The answer-generation code checks for completed `(configuration, question_id)` pairs and skips answers that already exist.

Therefore, an interrupted answer-generation run does not require recomputing completed answers.

## Seeds

The fixed seed in the project configuration is:

```text
42

Used in the betweenness_centrality audit in 3c and the 
fixed-seed merge audit in 2d.
```

## Expected runtime

Runtime depends strongly on the Colab GPU, model-loading time, cache state, and whether a stage is being run from scratch.

The final environment used an A100-SXM4-40GB. 

Record the observed times below after the final run:

```text
Stage A — decomposition:       30 minutes
Stage B — retrieval/RICR:      1 Hour
Stage C — answer generation:   45 minutes
Stage D — Q9 ablations:        1 hour
Total:                         3 hours 15 minutes

```

Expect an estimated runtime of 4-5+ hours on the T4 GPU. Environment is designed in a way that stage reruns can be easily resumed in the event of a runtime expiration.

Cached reruns are substantially faster because completed intermediate results and reranker scores are reused.

**Final commit hash:**

```text
5023d75471fd5690c88e7afbebaa56957ecdf9a0
```
