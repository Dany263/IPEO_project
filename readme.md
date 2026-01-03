# Mapping Swiss Ecosystems from Aerial Images and Environmental Variables

**Authors:** Dany Montandon, Loïc Trochen  
**Section:** SIE  
**Course:** Image Processing for Earth Observation  
**Date:** November–December 2025 & January 2026  

---

## 1. Overview

This repository contains the code, data preparation steps, and trained models required to reproduce our analysis of **Swiss ecosystems using aerial imagery and environmental (tabular) variables**.

The complete methodology, experiments, and discussion of results are presented in the accompanying **`report.pdf`**.

---

## 2. Repository Structure

```text
.
├── data/
│   ├── .gitkeep
│   └── dataset_split.csv
├── models/
│   ├── .gitkeep
│   └── TabularStandard_0.pt
├── figures/
│   └── .gitkeep
├── utils/
│   ├── eunis_label.py
│   ├── scripts.py
│   └── sweco_group_of_variables.py
├── ipeo_project.ipynb
├── inference.ipynb
├── environment.yml
├── experiment_log.csv
├── report.pdf
└── README.md

```

### Main files

- **`ipeo_project.ipynb`**  
  Main notebook containing the full pipeline: data preparation, model training, hyperparameter selection, and analysis.

- **`inference.ipynb`**  
  Lightweight notebook performing inference on a small set of test samples using the best trained model.

- **`report.pdf`**  
  Final report describing the methodology, experiments, and results.

---

## 3. Environment Setup

Create and activate the Conda environment using:

```bash
conda env create -f environment.yml
conda activate ecosystem_project
```

## 4. Data Preparation

### 4.1 Required Data

- Place the archive `images.zip` inside the `data/` folder:
    - data/images.zip

- The file `dataset_split.csv` is already provided and must remain in `data/`.


- The file `dataset_split.csv` is already provided and **must remain** in the `data/` directory.

### 4.2 Image Preprocessing (Run Once)

In `ipeo_project.ipynb`, run **only the first two cells** to:

- Unzip `images.zip`
- Convert the original `.tif` images to `.png` for faster loading

⚠️ **Important**  
This preprocessing step needs to be executed **only once**.  
After the images are extracted and converted to `.png`, these cells can be skipped in all subsequent runs.

---

## 5. Model Training and Experiments

After preprocessing, run the remainder of `ipeo_project.ipynb` normally.

The notebook performs:

- Tabular data processing  
- Image processing  
- Training of **Tabular**, **Image**, and **Combined** models  
- Model selection using **validation macro-F1 score**  
- Evaluation:
- Metrics
- Confusion matrices
- Permutation importance

Random seeds are fixed to ensure reproducibility when notebooks are run in the same order.

---

## 6. Inference

The `inference.ipynb` notebook:

- Loads the trained parameters of the best model  
- Runs inference on selected test samples  
- Displays predicted ecosystem classes  

### Model File

The trained model `TabularStandard_0.pt` is already included in:
- models/TabularStandard_0.pt


⚠️ **Note**  
Re-running `ipeo_project.ipynb` with the same architecture and hyperparameters will overwrite this file.  
If this happens, the original model used in the report can be downloaded here:

👉 **ADD LINK**

### Test Samples Used

The inference notebook evaluates the following randomly selected sample IDs:

```python
[
  '2743707_1218749',
  '2708579_1271348',
  '2748834_1205701',
  '2556236_1203945',
  '2496165_1116505'
]
```

7. Experiment Order (Summary)

Recommended execution order:

1. Data preparation
    - Unzip images
    - Convert `.tif` → `.png`
2. Model training
    - Run `ipeo_project.ipynb`
3. Evaluation
    - Metrics, confusion matrices, importance analysis
4. Inference
    - Run `inference.ipynb` on test samples

8. Reproducibility

- Random seeds are fixed.
- All results reported in report.pdf are fully reproducible using this repository.
- Specific files (dataset_split.csv, TabularStandard_0.pt) are version-controlled to ensure consistent behavior across runs.