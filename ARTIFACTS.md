# Experiment artifacts

This repository contains selected artifacts from the completed cluster runs.

## Included

The repository keeps the compact reference weights, benchmark CSV files, and figures needed to inspect the main experiments without rerunning training.

Reference weights:

```text
checkpoints/reference/dyn_tcn.pt
checkpoints/reference/ext_force_balance.pt
checkpoints/reference/ext_corruption_trained.pt
checkpoints/reference/tcn_full_novis.pt
```

Primary benchmark table:

```text
results/tables/loso_all.csv
```

Architecture and robustness tables:

```text
results/tables/design_comparison.csv
results/tables/extensions_clean.csv
results/tables/extensions_occlusion.csv
```

## Intentionally excluded

`groundlink_id_v4.pt` is a large processed cache derived from licensed GroundLink and SMPL-H assets. It is kept outside Git and should be copied to `data/processed/groundlink_id_v4.pt` for local reproduction.

The raw GroundLink recordings, MoSh++ fits, and SMPL-H model files are not redistributed.

The optional image demonstration inputs are also excluded from the public repository because they are not part of the validated dynamics benchmark and may contain third-party image content.
