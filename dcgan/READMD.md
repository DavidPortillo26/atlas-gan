# 🧠 DCGAN MNIST Project

Welcome to my Deep Convolutional Generative Adversarial Network (DCGAN) project!  
This repository demonstrates how to generate handwritten digit images similar to those in the MNIST dataset using PyTorch and Weights & Biases for experiment tracking.

---

## 📚 Project Objectives

- Build and train a baseline DCGAN architecture.
- Experiment with model architectures, hyperparameters, and precision settings.
- Evaluate image quality, training stability, and convergence across runs.
- Track all experiments and metrics using [Weights & Biases](https://wandb.ai).
- Record a video walkthrough of key learnings.

---

## 🗂️ Repository Structure

```bash
dcgan/
├── data/            # MNIST dataset and processed versions
├── models/          # Generator and Discriminator architectures
├── utils/           # Training scripts and helper functions
├── experiments/     # Jupyter Notebooks for each experiment
│   ├── baseline_dcgan.ipynb
│   ├── arch_variations.ipynb
│   ├── hyperparam_tuning.ipynb
│   └── precision_experiment.ipynb
├── configs/         # YAML files with hyperparameter settings
├── logs/            # Generated images and training logs
├── report.md        # Final project summary and insights
├── README.md        # You’re reading this!


# Create a virtual environment
python -m venv dcgan-env
source dcgan-env/bin/activate  # Use 'dcgan-env\\Scripts\\activate' on Windows

# Install dependencies
pip install torch torchvision wandb matplotlib pyyaml


python utils/train_baseline.py


python utils/train_arch_variant.py


python utils/train_tuning.py --config configs/tuning_config.yaml


python utils/train_precision.py