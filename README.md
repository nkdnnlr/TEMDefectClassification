# TEM-Defect-Classification
Hi! You've reached the code repository corresponding to the paper [Learning-based Defect Recognition for Quasi-Periodic HRTEM Images](https://doi.org/10.1016/j.micron.2021.103069), authored by Nik Dennler, Antonio Foncubierta-Rodriguez, Titus Neupert and Marilyne Sousa. In case of questions regarding the paper, please [write us](mailto:nik.dennler@posteo.de). If you have an issue relating the code, please use the GitHub Issues function.

## Before you start
Clone the repository, and set up the [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment using
```
conda env create -f environment.yml
```

## Data
The data used for this project is publicly available at [Zenodo](https://dx.doi.org/10.5281/zenodo.4739588). You can either download and use the data provided, or use your own data for re-traing the model and/or for testing the algorithm. 

## Training Model
In order to re-train the model, you have to adjust the parameters in `setup.py` to suit your data directories, then simply run 
```
python setup.py
```

## Running and Evaluating Algorithm
For running the algorithm, adjust the parameters in '' to suit your data directory, then run
```
python run_segmentation.py
```
If you would like to check how well the algorithm has performed, you will need to specify in 'evaluate_run.py' a directory where you provide the ground truth labels for your symmetry and blurry-region defects (as defined in the paper). Then you can run
```
python evaluate_segmentation.py
``` 
