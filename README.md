# CIFAR-10 CNN Architecture Search

Automated CNN architecture and hyperparameter search for CIFAR-10 using **KerasTuner RandomSearch**.

Instead of manually tuning the CNN architecture, the project searches over different network configurations and training hyperparameters to find a promising model.

## Overview

The project uses the CIFAR-10 dataset:

* 50,000 training images
* 10,000 test images
* 32×32 RGB images
* 10 classes

The workflow is:

```text
CIFAR-10
   ↓
Train / Validation Split
   ↓
CNN Architecture Search
   ↓
KerasTuner RandomSearch
   ↓
Best Hyperparameters
   ↓
Retrain on Full Training Set
   ↓
Final Test Evaluation
```

## Search Space

The tuner searches over:

| Hyperparameter      | Values     |
| ------------------- | ---------- |
| Conv Blocks         | 1–4        |
| Kernel Size         | 3×3 / 5×5  |
| Batch Normalization | On / Off   |
| Dropout             | 0.0–0.5    |
| Dense Layers        | 0–2        |
| Dense Units         | 64–256     |
| Optimizer           | Adam / SGD |
| Learning Rate       | 1e-4–1e-2  |

The convolutional filters use a fixed schedule:

```python
[32, 64, 128, 256]
```

This keeps the search space manageable.

## Best Configuration

The best configuration from the current 8-trial search was:

```text
Conv Blocks:      4
Filters:          32 → 64 → 128 → 256
Kernel Size:      3×3
Batch Norm:       False
Dropout:          0.3
Dense Layers:     2
Dense Units:      64
Optimizer:        Adam
Learning Rate:    0.000349
```

Best validation accuracy during the search:

```text
69.7%
```

## Training

The selected architecture is rebuilt from scratch and retrained using the full 50,000-image CIFAR-10 training set.

The final model is then evaluated on the untouched 10,000-image test set.

Early stopping is used during both the search and final training stages.

## Project Structure

```text
cifar10-cnn-architecture-search/
│
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── cifar10_cnn_architecture_search.ipynb
├── models/
└── outputs/
    ├── best_hyperparameters.json
    ├── training_curves.png
    └── kerastuner_logs/
```

## Technologies

* Python
* TensorFlow / Keras
* KerasTuner
* NumPy
* Matplotlib
* Google Colab

## Future Improvements

* Increase RandomSearch trials to 20–40+
* Compare with KerasTuner Hyperband
* Tune filters independently per block
* Add learning-rate scheduling
* Explore residual and depthwise-separable CNN architectures
