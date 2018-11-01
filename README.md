# Coalescent Simulations

For a general introduction, see [presentation](presentation/main.ipynb), prepared using [RISE](https://github.com/damianavila/RISE).

## Introduction

### [Classifier](analyze/Classifier.py)
SVM (default kernel: linear) that accepts features as data and is trained to distinguish Kingman from Bolthausen-Sznitman.

### [Experiment](experiment/Experiment.py)
Sets up and executes simulation experiments with various combinations of hyper-parameters. Mutation is applied here while computing tree asymmetry measures. 

### [Controller](experiment/Controller.py)
A need to save / repeat same experiments led to a Controller serving as a handle to experiments. In other words, [main](main.py) executes Experiments through Controller.

### [Models](models/)
Defines Kingman and Bolthausen-Sznitman.

### [Utils](utils/)
Helper functions.

### [config.py](config.py)
Argparser

## Run
To run with default settings:
```
python main.py
```

