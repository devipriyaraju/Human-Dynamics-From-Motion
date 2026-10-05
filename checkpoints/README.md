# Checkpoints

`reference/` contains selected model weights from the completed Compute Canada runs.

The included files are raw PyTorch `state_dict` objects from the original experiments:

- `dyn_tcn.pt`, temporal dynamics model trained on the fixed development split
- `ext_force_balance.pt`, dynamics model trained with the force-balance objective
- `ext_corruption_trained.pt`, dynamics model trained with motion corruption
- `tcn_full_novis.pt`, four-channel temporal contact model

The dynamics weights use 132 input channels, four 256-channel temporal blocks, and a 24-channel output head. The contact model uses the same temporal backbone and a four-channel output head.

The raw dynamics weights do not store target mean and standard deviation. Recompute those statistics from the training subjects before using the weights for physical-unit inference. New training runs can use the structured checkpoint helper in `motion_dynamics.training.engine` to save weights and normalization together.

Fold-specific LOSO weights were not present in the uploaded result bundle, so the repository includes their exact metrics in `results/tables/loso_all.csv` rather than inventing checkpoint files.
