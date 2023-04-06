# Investigating representation schemes for surrogate modeling of High Entropy Alloys

by Arindam Debnath, Wesley F. Reinhart.

This repository is the official implementation of '[Investigating representation schemes for surrogate modeling of High Entropy Alloys]'(https://arxiv.org/pdf/2301.00179.pdf), which has been submitted for publication in *Computational Materials Science*.

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
├── README.md
├── results
└── saved_models
    ├── best_models
    └── Encoders
        ├── atomic
        ├── dense
        ├── mod_pettifor
        ├── pettifor
        ├── PTR
        ├── random
        └── random-tr
            
```

## Dataset 

The `dataset` folder contains the following files - 

* alternate_orders.pkl - A pickle file containing the 1D Pettifor and Modified Pettifor ordering
* hardness.csv - The hardness dataset from the paper '[A machine learning-based alloy
design system to facilitate the rational design of high entropy alloys with enhanced hardness]'(https://doi.org/10.1016/j.actamat.2021.117431)


## Files from GTDL paper

This folder contains 

## Figures

All the figures depicting the results from the various tests performed are contained in this folder.

## Modules

Contains the 

## Miscellaneous 



