# Graph-Based Context-Aware Reranking for RAG

This repository contains a research-style Jupyter notebook for exploring a graph-based reranking strategy for retrieval-augmented generation (RAG) systems.

The notebook evaluates multiple retrieval signals and combines them into a reranking pipeline that is designed to improve context selection for question answering tasks. In particular, it experiments with:

- semantic similarity between query and chunks
- cross-encoder relevance scores
- NLI-based evidence scoring
- a graph-inspired reranking step using similarity and evidence signals
- a logistic-regression ensemble over multiple features

## Project goal

The aim is to improve the quality of retrieved context chunks before they are passed to a downstream LLM or QA model. Rather than relying on a single retrieval signal, the notebook builds a richer reranking model that integrates multiple signals and scores them jointly.

## Files

- `Graph_Based_Reranking.ipynb` — the main experiment and analysis notebook

## Requirements

Python 3.10+ is recommended.

Install the dependencies with:

```bash
pip install numpy pandas scikit-learn datasets sentence-transformers transformers torch tqdm
```

Depending on your environment, you may also need:

```bash
pip install ipykernel jupyter
```

## Notebook workflow

The notebook does the following:

1. Loads a QA dataset (including SQuAD-style examples)
2. Splits context into chunks
3. Computes retrieval signals such as:
   - sentence similarity
   - cross-encoder relevance
   - NLI evidence
4. Builds candidate reranking features
5. Applies graph-based scoring / context-aware reranking logic
6. Trains and evaluates ensemble models for final ranking quality
7. Reports benchmark-style metrics such as R@1, R@3, R@5, MRR, and average rank

## How to run

Open the notebook in Jupyter or VS Code and run all cells:

```bash
jupyter notebook Graph_Based_Reranking.ipynb
```

Or use VS Code's notebook runner if you prefer an IDE-based workflow.

## Typical experiment structure

The code in the notebook includes functions for:

- chunking text
- encoding chunks and queries
- computing similarity matrices
- applying cross-encoder and NLI-based evidence scores
- building graph-style reranking data
- training logistic-regression models over feature sets
- evaluating retrieval metrics across several reranking configurations

## Research focus

This project is best understood as a prototype or experimental notebook for investigating how graph-based context-aware reranking can improve ranking quality in RAG pipelines. It is especially relevant for:

- knowledge-intensive QA tasks
- long-context retrieval
- improving the quality of retrieved evidence before generation
- combining multiple signals into a more robust reranker

## Notes

- The repository currently contains the notebook as the core artifact; there is no packaged library or app yet.
- The notebook is research-oriented and intended for experimentation rather than production deployment.
- Outputs in the notebook include sample metrics and a comparison of different reranking strategies.

## License

This repository does not currently include an explicit license file. If you plan to reuse or publish the code, add a license before distribution.

## Contributing

This project is currently a single-notebook research prototype. Contributions are welcome if you want to:

- modularize the code into reusable Python scripts
- add dependencies and environment setup files
- document the methodology more formally
- improve benchmarking and evaluation scripts

