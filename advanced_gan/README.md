# Advanced GAN: Celebrity Face Generation with PyTorch 🎭🧠

This project builds on the foundational DCGAN model trained on MNIST and applies the same principles to a more complex image generation task — synthesizing celebrity faces using the CelebA dataset. It explores architectural enhancements, hyperparameter tuning, and leverages Weights & Biases (W&B) for experiment tracking and visualization.

## 🔍 Project Objectives

- Generate realistic 64×64 celebrity faces using a Deep Convolutional GAN
- Experiment with architecture variants and hyperparameter settings
- Apply lessons learned from the MNIST GAN project to a higher-complexity dataset
- Track and compare model performance using W&B
- Maintain modular code structure for reproducibility

## 📁 Directory Structure

```plaintext
advanced_gan/
├── configs/        # YAML files for baseline and experimental settings
├── data/           # CelebA dataset and preprocessed versions
├── models/         # Generator and Discriminator architectures
├── utils/          # Data loader and training logic
├── logs/           # Sample images and training logs
├── experiments/    # Jupyter notebooks for each experiment
└── README.md       # Project overview and usage

git clone https://github.com/davidportillo26/atlas-gan.git
cd atlas-gan/advanced_gan

python3 -m venv venv
source venv/bin/activate

pip install torch torchvision wandb pyyaml

python3 utils/train_baseline.py
