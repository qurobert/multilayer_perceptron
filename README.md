Breast Cancer Classification with a Custom MLP
==============================================

This repository contains a Python implementation of a Multi-layer Perceptron (MLP) for binary classification of breast tumors (malignant vs. benign). The model is built from scratch with configurable parameters and training options, and it also includes a comparison with scikit-learn's `MLPClassifier`.

Overview
--------

This project provides functionality to:

-   **Data Splitting and Preprocessing:**

    -   Standardizes the dataset.
    -   Converts tumor labels (`M` for malignant and `B` for benign) into binary values (1 and 0).
    -   Splits the data into 75% training and 25% validation sets.
-   **Training a Custom Neural Network:**

    -   Configurable multi-layer perceptron with:
        -   **Hidden Layers:** Default is 2 (configurable).
        -   **Hidden Nodes:** Default is 20 per layer.
        -   **Output Nodes:** 2 for binary classification.
        -   **Activation Functions:** ReLU or Sigmoid.
        -   **Weight Initialization:** Xavier or HeUniform.
        -   **Loss Function:** Categorical Cross Entropy.
        -   **Optimizer:** Adam with L2 regularization.
-   **Prediction and Evaluation:**

    -   The trained model classifies tumors and evaluates performance using key metrics:
        -   Loss
        -   Accuracy
        -   Precision
        -   Recall
        -   F1-score
-   **Comparison with scikit-learn:**

    -   Uses `MLPClassifier` from scikit-learn for performance benchmarking.

Dataset
-------

The dataset consists of **567 tumor samples**, each represented by multiple feature measurements. The labels are:

-   **M:** Malignant
-   **B:** Benign

Requirements
------------

-   Python 3.6+
-   Required dependencies (install via `pip`):
    -   `numpy`
    -   `pandas`
    -   `scikit-learn`
    -   `matplotlib`
    -   `tqdm`

To install all requirements, run:

```
pip install -r requirements.txt

```

Usage
-----

The program supports several modes of operation via command-line arguments:

### split

Splits and preprocesses the dataset into training and validation sets.

```
python3 ./src/main.py split

```

### train

Trains the custom neural network with the following default parameters:

-   **Hidden Layers:** 2
-   **Hidden Nodes:** 20 (configurable)
-   **Weight Initialization:** He (default) or Xavier
-   **Activation Function:** ReLU (default) or Sigmoid
-   **Optimizer:** Adam with L2 regularization

```
python3 ./src/main.py train

```

Additional options, such as disabling the progress bar or logging, are available via command-line flags.

### predict

Uses a trained model (saved weights and biases) to classify validation data and evaluate performance.

```
python3 ./src/main.py predict

```

### sklearn

Trains and evaluates an `MLPClassifier` from scikit-learn for performance comparison.

```
python3 ./src/main.py sklearn

```

Configuration
-------------

You can modify the following parameters directly in the code or extend the argument parser for runtime adjustments:

### Network Architecture

-   **Hidden Layers:** Default is 2.
-   **Hidden Nodes:** Default is 20.
-   **Output Nodes:** Default is 2.

### Activation Functions

-   Options: `relu` or `sigmoid`

### Weight Initialization

-   Options: `xavier` or `he`

### Training Parameters

-   Epochs, batch size, learning rate, early stopping patience, and L2 regularization strength.

Logging and Plotting
--------------------

### Logging

-   Training and evaluation details are logged in a file (`.log`).
-   Use `--disable-logs` to disable logging.

### Plotting

-   Training and validation loss/accuracy curves are plotted.
-   Use `--disable-plot` to disable plotting.

License
-------

This project is licensed under the MIT License.

Acknowledgements
----------------

-   The dataset used in this project is assumed to be derived from a publicly available breast cancer dataset.
-   This project compares the custom MLP model with scikit-learn's `MLPClassifier` to evaluate performance and flexibility.
