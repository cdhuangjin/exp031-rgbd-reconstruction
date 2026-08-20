# Reproducibility notes

The experiments use public TUM RGB-D and NeRF Synthetic data. The original
archives are excluded from this repository. Set local dataset paths when
running the scripts.

The protocol includes source-only threshold calibration, held-out frame
evaluation, three expanded sampling windows, pose perturbation stress tests,
TSDF references, and Gaussian-splatting references. The JSON summaries in
`results/summaries/` are derived outputs from the verified runs.

The repository contains the implementation, analysis scripts, plotting source,
figures, configurations, and derived summaries. A persistent repository DOI or
accession identifier should be added after the public repository is created.
