# Reliability-Guided Cross-Scene RGB-D Reconstruction

This repository contains the reproducibility materials for the manuscript
“Reliability-Guided Cross-Scene RGB-D Reconstruction with Source-Conditioned Depth Filtering”.

## Contents

- `src/core/`: RGB-D fusion implementation and local smoke-test utilities
- `src/analysis/`: experiment runners, analysis scripts, figure-generation source, and derived summaries
- `src/tests/`: local tests for the core implementation
- `results/figures/`: publication figures in vector and preview formats
- `results/summaries/`: derived experiment summaries in JSON format
- `docs/`: reproducibility notes and experiment protocol

## Data

The experiments use the public TUM RGB-D and NeRF Synthetic datasets. The
original third-party datasets are not included in this repository. Download
the datasets from their original public sources and provide the local paths
required by the scripts.

## Reproducibility

The experiment protocol, frame splits, metrics, baseline definitions, and
negative-result boundaries are documented in `docs/`. Generated summaries are
derived outputs and do not replace the original public datasets.

## License

The repository license and any dataset-specific terms should be added before
public release. Dataset terms remain governed by the original dataset owners.
