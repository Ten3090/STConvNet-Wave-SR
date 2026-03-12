# Dataset placeholder

Place the processed training data in this directory before training.

Expected files:

- `train_lr.nc`
- `train_hr.nc`
- `val_lr.nc`
- `val_hr.nc`
- `test_lr.nc`
- `test_hr.nc`
- `stats.json`

Expected variables:

- low-resolution NetCDF: `swh`, `period`, `dir_sin`, `dir_cos`
- high-resolution NetCDF: `swh`

These data files are intentionally not included in the public repository.
