# ATML Programming Assignment 0

A comprehensive implementation of deep learning fundamentals covering computer vision architectures, vision transformers, generative models, and multimodal learning with CLIP.

## Overview

This project demonstrates practical implementations of key deep learning concepts through five core assignments:

1. **Baseline CNN Training & ResNet Analysis** - Understanding residual connections and transfer learning
2. **Vision Transformers (ViT)** - Exploring attention mechanisms and patch-based image processing
3. **Generative Adversarial Networks (GANs)** - Implementing and analyzing training dynamics
4. **CLIP Multimodal Learning** - Zero-shot classification and modality gap analysis
5. **Variational Autoencoders (VAEs)** - Generative modeling with latent variable approaches

## Requirements

Install dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── Task1.py               # CNN baseline and ResNet analysis
├── Task2.py               # Vision Transformers implementation
├── Task3.py               # GAN implementation and training analysis
├── Task4.py               # CLIP multimodal experiments
├── Task5.py               # VAE implementation
├── architecture.py        # VAE model architecture
├── requirements.txt       # Project dependencies

```

## Tasks

### Task 1: CNN Baseline & ResNet Analysis

Implements comprehensive analysis of ResNet architectures including:
- Baseline ResNet-152 training on CIFAR-10
- Residual connection ablation studies
- Feature hierarchy visualization using t-SNE
- Transfer learning experiments (CIFAR-10 to CIFAR-100)
- Comparison of fine-tuning strategies

**Key Features:**
- Weight initialization and training loop implementation
- Feature extraction from multiple network layers
- Transfer learning performance comparison
- Visualization of training dynamics

### Task 2: Vision Transformers

Explores Vision Transformer architecture and attention mechanisms:
- Pre-trained ViT inference on CIFAR-10
- Attention weight visualization and interpretation
- Patch masking robustness analysis
- CLS token vs mean pooling comparison for feature extraction

**Key Features:**
- Attention map visualization and overlay generation
- Robustness testing with random and systematic patch masking
- Linear probe evaluation of different pooling strategies

### Task 3: Generative Adversarial Networks

Comprehensive GAN implementation with training issue analysis:
- Vanilla GAN implementation for MNIST generation
- Training dynamics monitoring and visualization
- Common training problems investigation:
  - Gradient vanishing with strong discriminator
  - Mode collapse scenarios
  - Discriminator overfitting on limited data
- Mitigation strategies and their effectiveness

**Key Features:**
- Loss tracking and convergence analysis
- Sample quality assessment
- Training stability experiments

### Task 4: Variational Autoencoders

VAE implementation focusing on generative modeling:
- Encoder-decoder architecture for MNIST
- Latent space analysis and interpolation
- Reconstruction quality assessment
- Generative sampling from learned distributions

### Task 5: CLIP Multimodal Learning

Advanced experiments with CLIP for multimodal understanding:
- Zero-shot classification on STL-10 with various prompting strategies
- Modality gap analysis and visualization
- Procrustes alignment for bridging image-text embedding spaces
- Performance evaluation of alignment techniques

**Key Features:**
- Multiple prompting strategy comparison
- t-SNE visualization of embedding spaces
- Orthogonal Procrustes analysis for alignment
- Classification accuracy improvements through alignment



## Usage

Execute individual tasks:

```bash
python Task1.py    # CNN and ResNet experiments
python Task2.py    # Vision Transformer analysis
python Task3.py    # GAN implementation
python Task4.py    # CLIP multimodal experiments
python Task5.py    # VAE experiments
```

Each script is self-contained and will:
- Download required datasets automatically
- Execute all experiments for the given task
- Generate visualizations and save results
- Display performance metrics and analysis

## Results

These contain:
- Performance metrics and accuracy scores
- Training progress logs
- Experimental observations
- Key findings and analysis



## Hardware Requirements

- CUDA-compatible GPU recommended for efficient training
- Minimum 8GB GPU memory for larger models
- Automatic fallback to CPU if CUDA unavailable

## License

This project is part of an academic programming assignment for educational purposes.