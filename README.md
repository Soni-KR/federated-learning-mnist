
# Federated Learning on MNIST: FedAvg vs FedProx under Non-IID Data

## Overview

This project explores the impact of data heterogeneity on Federated Learning using the MNIST handwritten digit dataset.

The study progressively evaluates:

* A centralized CNN baseline
* Federated Averaging (FedAvg) under IID conditions
* FedAvg under extreme Non-IID conditions
* FedProx under the same Non-IID setting

The objective is to understand how client data distributions affect federated optimization and how FedProx can mitigate the effects of client drift.

---

## Results

| Method            | Accuracy (%) |
| ----------------- | -----------: |
| Centralized CNN   |        98.93 |
| FedAvg IID        |    **99.05** |
| FedAvg Non-IID    |        71.50 |
| FedProx μ = 0.001 |        82.98 |
| FedProx μ = 0.01  |    **85.60** |
| FedProx μ = 0.1   |        73.88 |

---

## Key Findings

### Federated Learning can match centralized performance

| Method          | Accuracy |
| --------------- | -------: |
| Centralized CNN |   98.93% |
| FedAvg IID      |   99.05% |

Under IID client distributions, Federated Learning achieved performance comparable to centralized training.

### Non-IID data significantly degrades FedAvg

| Method         | Accuracy |
| -------------- | -------: |
| FedAvg IID     |   99.05% |
| FedAvg Non-IID |   71.50% |

When clients possess highly heterogeneous data, local objectives diverge and global performance decreases substantially.

### FedProx improves performance under Non-IID conditions

| Method           | Accuracy |
| ---------------- | -------: |
| FedAvg Non-IID   |   71.50% |
| FedProx μ = 0.01 |   85.60% |

FedProx recovered **14.10 percentage points** compared to standard FedAvg.

---

## Visualizations

### Overall Comparison

![Comparison](final_comparison.png)

### Impact of Non-IID Data

![Non-IID Impact](noniid_impact.png)

### FedProx Recovery

![FedProx Recovery](fedprox_improvement.png)

---

## Experimental Setup

### Dataset

MNIST handwritten digit dataset:

* 60,000 training images
* 10,000 test images
* 10 classes
* 28 × 28 grayscale images

### Model Architecture

A Convolutional Neural Network (CNN) was used throughout all experiments:

* Conv2D (32 filters)
* Conv2D (64 filters)
* MaxPooling
* Fully Connected Layer (128 neurons)
* Output Layer (10 classes)

### Federated Learning Configuration

* 5 Clients
* 5 Communication Rounds
* 1 Local Epoch
* Adam Optimizer
* Learning Rate = 0.001

---

## Repository Contents

```text
.
├── centralized.ipynb
├── fedavg_iid.ipynb
├── fedavg_non_iid.ipynb
├── fedprox_non_iid.ipynb
├── figures.ipynb
├── fedavg_non_iid_accuracy.png
├── fedprox_non_iid_accuracy.png
├── requirements.txt
└── README.md
```

---

## Notebooks

### centralized.ipynb

Centralized CNN training on MNIST used as a baseline for comparison.

### fedavg_iid.ipynb

Implementation of Federated Averaging (FedAvg) under IID client distributions.

### fedavg_non_iid.ipynb

Investigation of FedAvg performance under extreme Non-IID client data distributions.

### fedprox_non_iid.ipynb

Implementation of FedProx to mitigate client drift under Non-IID conditions.

### figures.ipynb

Generation of result visualizations and comparative analysis figures.

---

## Discussion

These experiments demonstrate one of the key challenges of Federated Learning: data heterogeneity.

While FedAvg performs similarly to centralized learning under IID conditions, performance drops significantly when clients hold highly heterogeneous data. FedProx introduces a proximal regularization term that stabilizes local optimization and substantially improves performance in Non-IID environments.

Among the tested configurations, **μ = 0.01** achieved the best results.

---

## Future Work

Potential extensions include:

* Flower framework implementation
* FedNova
* SCAFFOLD
* Differential Privacy
* Secure Aggregation
* CIFAR-10 experiments
* Larger client populations

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Author

**Mourad Kraiem**

Engineering Student – ENSI

Interests:

* Federated Learning
* Machine Learning
* Distributed AI Systems

