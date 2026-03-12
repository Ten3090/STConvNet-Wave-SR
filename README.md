# STConvNet-Wave-SR

Code release for the spatiotemporal wave super-resolution model used in the manuscript.

## Overview

This repository contains the training code for a spatiotemporal super-resolution model for significant wave height (SWH) reconstruction. The method combines:

- ConvLSTM-based temporal modeling over consecutive low-resolution frames
- Swin Transformer based spatial super-resolution
- selective multi-scale feature fusion
- physics-informed loss constraints

The current public release is configured for the manuscript model setting:

- temporal input length: `T = 5`
- low-resolution input: `8 x 8`
- high-resolution target: `40 x 40`
- channels per low-resolution frame: `swh`, `period`, `dir_sin`, `dir_cos`
- default upscale factor: `5x`

## Repository structure

```text
STConvNet-Wave-SR/
├── train_stconvnet.py
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── models/
│   ├── __init__.py
│   ├── swin_transformer.py
│   ├── partial_fusion_swin_sr.py
│   └── spatiotemporal_partial_fusion_sr.py
├── data/
│   ├── __init__.py
│   ├── spatiotemporal_dataset.py
│   └── yearly_hr_0p1/
│       └── README.md
└── utils/
    ├── __init__.py
    ├── metrics.py
    ├── physics_loss.py
    └── visualization.py
```

## Environment

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data preparation

The training code expects the following directory structure:

```text
data/yearly_hr_0p1/
├── train_lr.nc
├── train_hr.nc
├── val_lr.nc
├── val_hr.nc
├── test_lr.nc
├── test_hr.nc
└── stats.json
```

Expected contents:

- `*_lr.nc`: low-resolution inputs containing `swh`, `period`, `dir_sin`, `dir_cos`
- `*_hr.nc`: high-resolution target containing `swh`
- `stats.json`: normalization statistics

### Data Availability

The training data used in this study is derived from the **Copernicus Marine Service**:

- **Product**: Global Ocean Waves Reanalysis (WAVERYS)
- **Source**: [https://marine.copernicus.eu/](https://marine.copernicus.eu/)
- **Variables**: Significant wave height (swh), wave period, wave direction
- **Spatial domain**: Black Sea region
- **Temporal resolution**: Hourly

Due to Copernicus data redistribution policies, the processed dataset is **not included** in this repository. Users can:

1. Register for free access at [Copernicus Marine Service](https://marine.copernicus.eu/)
2. Download the raw wave reanalysis data for the Black Sea region
3. Use the data loader in `data/spatiotemporal_dataset.py` as a reference for preprocessing
4. Ensure the processed files match the expected format described above

For questions about data access, please refer to the Copernicus Marine Service documentation or contact the repository maintainer.

## Training

Run the main training script:

```bash
python train_stconvnet.py --name stconvnet_release
```

You can also specify a custom dataset path:

```bash
python train_stconvnet.py --name stconvnet_release --data_dir path/to/data/yearly_hr_0p1
```

By default, outputs are written to:

```text
experiments/exp13/<experiment_name>/
```

including:

- `config.json`
- `history.json`
- `training_curves.png`
- `checkpoints/best_model.pth`
- `checkpoints/last.pth`

## Notes for reproducibility

- Set the same random seed as used in the paper when reproducing results.
- Ensure that the low-resolution and high-resolution files are temporally aligned.
- The code uses normalized inputs if `stats.json` is present.
- Reported metrics depend on the exact data split and preprocessing pipeline.

## Quick test

Run the quick model check:

```bash
python quick_test.py
```

This test verifies that the model can be constructed and that an input with
shape `[2, 5, 4, 8, 8]` produces an output with shape `[2, 1, 40, 40]`.

## License

This repository is released under the MIT License.
