# Robustness Experiments on CIFAR-10, Fashion-MNIST, and MNIST

This repository contains experiments and models for evaluating the robustness of machine learning models under various adversarial conditions. The project focuses on three datasets: CIFAR-10, Fashion-MNIST, and MNIST. It includes scripts, notebooks, and pre-trained models for reproducing the experiments.

## Project Structure

```
DL/
├── cifar10/
│   ├── cifar10.ipynb                # Notebook for CIFAR-10 experiments
│   ├── cifar10_results_table.csv    # Results summary for CIFAR-10
│   ├── nc_fld_experiment_report.txt # Experiment report
│   ├── models/                      # Pre-trained models for CIFAR-10
│   └── data/                        # Dataset folder
├── fashionmnist/
│   ├── fashion.ipynb                # Notebook for Fashion-MNIST experiments
│   ├── saved_models/                # Pre-trained models for Fashion-MNIST
├── mnist/
│   ├── mnist_training.ipynb         # Notebook for MNIST experiments
│   ├── models/                      # Pre-trained models for MNIST
```

## How Models Are Developed and Saved

### 1. **Model Development**
- **Libraries Used**: PyTorch, NumPy, Scikit-learn, and others.
- **Model Architectures**:
  - Simple Convolutional Neural Networks (CNNs) for classification tasks.
  - Defense mechanisms include K-Nearest Neighbors (KNN), Agglomerative Clustering, and One-Class SVM (OCSVM).
- **Adversarial Scenarios**:
  - Label flipping, sign flipping, and LIE (Label Injection Errors) with varying malicious ratios (e.g., 0.1, 0.2, 0.3, 0.4).

### 2. **Training and Evaluation**
- Models are trained on clean and adversarially perturbed datasets.
- Evaluation metrics include accuracy, robustness scores, and heatmaps.

### 3. **Model Saving**
- Models are saved in `.pth` format using PyTorch's `torch.save()` function.
- Naming convention:
  - `<dataset>_<attack_type>_mr<malicious_ratio>_<defense_method>_final.pth`
  - Example: `cifar10_label_flip_mr0.1_knn_final.pth`

### 4. **Pre-trained Models**
- **CIFAR-10**: Located in `cifar10/models/`
- **Fashion-MNIST**: Located in `fashionmnist/saved_models/`
- **MNIST**: Located in `mnist/models/`

## Requirements

To run the notebooks and scripts, install the required Python packages:

```bash
pip install -r requirements.txt
```

## Requirements File

```plaintext
numpy
pandas
matplotlib
seaborn
torch
torchvision
scikit-learn
tqdm
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/AyaanKhan0111/FLDefense.git
   cd FLDefense
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the desired notebook:
   - `cifar10/cifar10.ipynb` for CIFAR-10 experiments.
   - `fashionmnist/fashion.ipynb` for Fashion-MNIST experiments.
   - `mnist/mnist_training.ipynb` for MNIST experiments.

4. Run the cells to reproduce the experiments.

## Results

- Results are saved in CSV files and visualized in the notebooks.
- Pre-trained models can be loaded for evaluation or further training.

