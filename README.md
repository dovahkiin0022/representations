# Investigating representation schemes for surrogate modeling of High Entropy Alloys

by Arindam Debnath, Wesley F. Reinhart.

This repository is the official implementation of [Investigating representation schemes for surrogate modeling of High Entropy Alloys](https://arxiv.org/pdf/2301.00179.pdf), which has been submitted for publication in *Computational Materials Science*.

In this paper, we have systematically compared several representation schemes for atomic fractions of alloys on the basis of their performance in single-task deep learning models and in transfer learning scenarios.

<img width="2088" alt="image" src="https://user-images.githubusercontent.com/64245681/230444106-e58c25b2-e130-429a-b5ba-b3dea68da774.png">

# Getting the code

You can download a copy of all the files in this repository by cloning the git repository:

```
git clone https://github.com/dovahkiin0022/representations.git
```

A copy of the repository is also archived at 

# Installation

Run the following command to install the required dependencies specified in the file `environment.yml`
```
conda env create -f environment.yml
```

# Implementation

The repository contains the following directories:
```
.
├── dataset
│   ├── alternate_orders.pkl
│   ├── hardness.csv
│   ├── periodic_table.csv
│   ├── yield_strength.csv
│   └── yield_strength_original.csv
├── environment.yml
├── figures
├── Files_from_GTDL_paper
│   ├── element_property.txt
│   ├── gao_data.txt
│   ├── gfa_dataset.txt
│   └── Z_row_column.txt
├── misc
├── modules
│   ├── encoder.py
│   ├── function.py
│   ├── __init__.py
│   ├── model_select.py
│   ├── plotting_functions.py
│   ├── __pycache__
│   └── representation_schemes.py
├── notebooks
│   ├── 0_encoder_training.ipynb
│   ├── 1a_transfer_learning_phase.ipynb
│   ├── 1b_transfer_learning_hardness.ipynb
│   ├── 1c_transfer_learning_yield_strength.ipynb
│   ├── 2a_generalizability_phase.ipynb
│   ├── 2b_generalizability_hardness.ipynb
│   ├── 2c_generalizability_ys.ipynb
│   └── visualization.ipynb
├── LICENSE
├── README.md
├── results
└── saved_models
            
```

## Dataset 

The `dataset` folder contains the following files - 

* alternate_orders.pkl - A pickle file containing the 1D Pettifor and Modified Pettifor ordering
* hardness.csv - The hardness dataset from the paper [A machine learning-based alloy
design system to facilitate the rational design of high entropy alloys with enhanced hardness](https://doi.org/10.1016/j.actamat.2021.117431)
* periodic_table.csv - A csv file containing properties for the elements of the periodic table, from [this repository](https://gist.github.com/GoodmanSciences/c2dd862cd38f21b0ad36b8f96b4bf1ee)
* yield_strength_original.csv - The yield strength dataset from the paper [Comprehensive data compilation on the mechanical properties of refractory high-entropy alloys](https://doi.org/10.1016/j.dib.2018.10.071)
* yield_strength.csv - The yield strength dataset obtained after removal of less frequently occuring elements from yield_strength_original.csv 

## Figures

All the figures depicting the results from the various tests performed are contained in this folder.

## Files from GTDL paper

This folder contains the necessary files needed for the 2D Periodic Table Representation as well as the datasets from the paper [A general and transferable deep learning framework for predicting phase formation in materials](https://doi.org/10.1038/s41524-020-00488-z) (the github repository can be found [here](https://github.com/sf254/glass-froming-ability-prediction.git)). These include - 

* element_property.txt - Necessary file for the 2D Periodic Table Representation.
* gao_data.txt - The High Entropy Alloy phase dataset. 
* gfa_dataset.txt - The Glass Formation Ability dataset.
* Z_row_column.txt - Necessary file for the 2D Periodic Table Representation.


## Miscellaneous 

The information necessary for reproducing the results in the paper (like the dataset splits used for the the 10-fold cross validation) have been stored in the `misc` folder as pickle or json files.

## Modules

The `modules` folder contains several .py files with necessary functions and codes for the project.

* encoder.py - Contains the Dense Neural Networks and the Convolutional Neural Networks for the single-task predictions
* function.py - Contains some general functions as well as functions for converting compositions into the PTR image.
* model_select.py - Contains code to evaluate several out-of-the-box `scikit-learn` regression models for the regression task using Root Mean Squared Error and Pearson's Correlation Coefficient.
* plotting_functions.py - Contains code for generating the periodic table heatmap.
* representation_schemes.py - Contains code for converting compositions into 1D representations as well as obtaining the latent codes from the trained single-task models.

## Notebooks

The `notebook` folder contains the Jupyter notebooks used for running the codes for the different tests. 

* 0_encoder_training.ipynb - This notebook contains the code for training the different single-task models using the 0D, 1D, 2D representations on the task of predicting the Glass Formation Ability of a given alloy.
* 1a_transfer_learning_phase.ipynb - This notebook contains the code for training transfer learning models using the latent codes from the trained single-task models for the task of High Entropy Alloy phase prediction. 
* 1b_transfer_learning_hardness.ipynb - This notebook contains the code for training transfer learning models using the latent codes from the trained single-task models for the task of High Entropy Alloy hardness prediction. 
* 1c_transfer_learning_yield_strength.ipynb - This notebook contains the code for training transfer learning models using the latent codes from the trained single-task models for the task of High Entropy Alloy yield strength prediction. 
* 2a_generalizability_phase.ipynb - This notebook contains the code for training the transfer learning models on different ratios of splitting the High Entropy Alloy phase dataset to evaluate the transfer learning models' ability to generalize to new data.
* 2b_generalizability_hardness.ipynb - This notebook contains the code for training the transfer learning models on different ratios of splitting the High Entropy Alloy hardness dataset to evaluate the transfer learning models' ability to generalize to new data.
* 2c_generalizability_ys.ipynb - This notebook contains the code for training the transfer learning models on different ratios of splitting the High Entropy Alloy yield strength dataset to evaluate the transfer learning models' ability to generalize to new data.
* visualization.ipynb - This notebook contains code to generate the figures used in the paper.

## Results

The results from the different Jupyter notebooks are stored as json objects in the `results` folder.

## Saved models

The single-task models are stored as `pytorch` objects in the `saved_models` folder.





