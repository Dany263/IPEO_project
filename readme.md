Mapping Swiss Ecosystems from Aerial Images and Environmental Variables

Authors: Dany Montandon, Loïc Trochen
Section: SIE
Course: Image Processing for Earthi Observation
Date: November/December 2025 & January 2026

1. Overview

This repository contains the code, and models to reproduce the analysis of Swiss ecosystems using aerial images and environmental variables.
The full analysis and discussion are presented in the accompanying report.

2. Environment Setup

Create the Python environment using:

conda env create -f environment.yml
conda activate ecosystem_project

3. Dataset Preparation

EXPLAIN THE DOWLOAD OF GITHUB AND WHAT THEY NEED TO ADD IN ORDER THAT EVERYTHING WORKS

Before running the notebook, the aerial images must be prepared. This involves two steps:

- Rename and place the images archive: Ensure the provided images archive is named images.zip and located in the data/ folder.
- Extract and convert images: The notebook includes cells that will (a) unzip the archive and (b) convert the original .tif images into .png format for faster loading.

⚠️ Important: This preprocessing only needs to be run once. After the images are extracted and converted to .png, you can skip this step in all subsequent runs.
 
 - Additonally the dataset_split.csv need to be added in the data/ folder
 

4. Experiment Order

- Data preparation: preprocess tabular data and images.
- Model training: run the notebook training.ipynb to train Tabular, Image, or Combined models.
- Evaluation: compute metrics, permutation importances, and confusion matrices.
- Inference: run inference.ipynb on a test sample.

Place the best models (ADD THE NAME OF IT .pt) in the models/ folder before running inference.ipynb.

5. Trained Model

Download the best-performing model from

ADD A LINK TO DOWNLOAD THE BEST MODEL

6. Running Inference

The notebook inference.ipynb:

- Loads a sample image from the test set.
- Loads the trained parameters of the best model.
- Runs inference on the test sample.
- Displays the predicted ecosystem class.

7. Additional Notes

Random seeds are set for reproducibility if notebooks are run in the same order.

All results shown in the report are fully reproducible using this repository.