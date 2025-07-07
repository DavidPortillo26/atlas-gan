# Final Report: DCGAN on MNIST

## 📌 Project Overview
This project involved training a Deep Convolutional Generative Adversarial Network (DCGAN) to generate digit images similar to those in the MNIST dataset. The goal was to gain practical experience with GANs and explore how architectural choices, hyperparameter settings, and precision affect model performance and image generation quality.

We tracked each experiment using [Weights & Biases](https://wandb.ai), maintained a modular repository structure, and reflected on observations at each stage.

---

## 🧱 Baseline DCGAN

**Setup:**
- Generator and Discriminator implemented with 2 layers each.
- Adam optimizer with learning rate `0.0002`, batch size `64`.
- 25 training epochs.

**Observations:**
- Generator started producing digit-like images around epoch 15.
- Loss curves were somewhat stable, but image sharpness varied across batches.
- Solid foundation for experimentation.

---

## 🧪 Experiment 1: Architecture Variations

**Changes Made:**
- Increased number of layers and filter depth in both Generator and Discriminator.
- Added an extra convolutional layer to Generator to increase expressiveness.

**Results:**
- Generated images had improved detail and edge clarity.
- Training was slightly slower but more stable overall.
- Generator loss decreased more steadily than in baseline.

**Takeaway:**
Architectural depth helped the model capture more visual features, resulting in cleaner digit outlines.

---

## ⚙️ Experiment 2: Hyperparameter Tuning

**Hyperparameters Tested:**
- Learning rates: `0.0001`, `0.0003`, `0.0005`
- Batch sizes: `64`, `128`
- Optimizers: `Adam`, `RMSprop`

**Results:**
- Lower learning rates produced smoother image transitions.
- Larger batch sizes led to more stable training dynamics.
- RMSprop introduced instability and noisier images compared to Adam.

**Takeaway:**
Using `Adam` with a learning rate around `0.0002–0.0003` and batch size of `128` provided best balance between stability and speed.

---

## 🔬 Experiment 3: Precision Changes

**Methods:**
- Float32 vs. Mixed Precision (using `torch.cuda.amp`)

**Results:**
- Mixed precision reduced training time by ~30%.
- GPU memory usage significantly decreased.
- Image quality stayed consistent with float32 results.

**Takeaway:**
Mixed precision training is ideal for faster experimentation without sacrificing visual quality—especially useful on GPU-limited setups.

---

## 📊 Weights & Biases Tracking

Throughout the project, we tracked:
- Generator and Discriminator losses per epoch
- Sample generated images
- Hyperparameters for each experiment
- Model comparisons in a unified dashboard

Link to W&B Dashboard → _[Insert your W&B link here]_

---

## 🧠 Summary & Learnings

- DCGAN is highly sensitive to architecture and hyperparameters.
- More layers and careful tuning lead to better image fidelity.
- Weights & Biases was essential for monitoring and comparing runs.
- Using mixed precision unlocked faster training with no downside.

---

## 🎥 Video Submission

Link to Loom/OBS demo → _[Insert your video link here]_

In the demo, you'll find:
- Project walkthrough and code structure
- W&B dashboard tour with experiment results
- Final thoughts and takeaways

---

## 📁 Repo Structure


---

## ✅ Conclusion

This project offered practical exposure to GAN training and evaluation. Each experiment demonstrated how small changes can greatly affect output. Future extensions might include training on more complex datasets (e.g. CelebA) or implementing Conditional GANs.