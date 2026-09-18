# Defense-Against-Adversarial-Attacks

# Defense Against Adversarial Attacks on Image Recognition

An adversarially robust image classification framework that combines PGD-based adversarial training, Exponential Particle Swarm Optimization (ExPSO) for structured pruning, Gumbel-Softmax Stochastic Multi-Expert (SME) routing, and Multi-Version Compressed Neural Network Training (MVC-NNT).

## Overview

Deep neural networks used for image classification are vulnerable to adversarial perturbations. While adversarial training improves robustness, it can result in computationally expensive and dense models.

This project explores a defense framework that combines model compression with stochastic multi-expert routing and data diversity to improve adversarial robustness while reducing model complexity.

The framework is implemented using **ResNet-18 on CIFAR-10** under an L-infinity threat model with `ε = 8/255`.

## Key Components

- **PGD Adversarial Training** for improving robustness against adversarial perturbations
- **ExPSO-Guided Structured Pruning** for producing compressed expert models
- **Gumbel-Softmax Stochastic Multi-Expert (SME) Routing** for stochastic expert selection
- **Multi-Version Compressed Neural Network Training (MVC-NNT)** using disjoint data partitions
- Evaluation against **FGSM, PGD-20, PGD-100, and AutoAttack**

## Pipeline

```text
CIFAR-10
   │
   ▼
ResNet-18 + PGD Adversarial Training
   │
   ▼
ExPSO-Guided Structured Pruning
   │
   ├────────────┬────────────┐
   ▼            ▼            ▼
30% Expert   50% Expert   70% Expert
   │            │            │
   └────────────┴────────────┘
                │
                ▼
       Gumbel-Softmax SME
                │
                ▼
             MVC-NNT
                │
                ▼
        Hybrid / Ultimate Defense
                │
                ▼
   FGSM / PGD-20 / PGD-100 / AutoAttack

Experimental Results
Model	Clean Accuracy	FGSM	PGD-20	PGD-100	AutoAttack	Sparsity
ExPSO-30	71.46%	45.05%	39.28%	39.04%	34.90%	30.3%
ExPSO-50	69.05%	44.14%	39.00%	38.77%	35.00%	53.7%
ExPSO-70	65.99%	41.72%	37.41%	37.21%	33.50%	73.7%
SME	71.36%	51.33%	47.75%	47.46%	38.40%	52.5%
MVC-NNT	68.60%	47.36%	45.30%	—	45.20%	52.5%
Ultimate Defense	71.36%	50.74%	47.26%	46.99%	38.60%	52.5%


Inference Performance
- ExPSO-50 achieved a 1.24× inference speedup over the baseline.
- ExPSO-70 achieved a 1.22× inference speedup.
- The SME ensemble improved robustness but introduced additional inference cost because all three experts are evaluated for each input.
Ablation Study
The project evaluates the contribution of individual components by comparing single experts, averaging, stochastic SME routing, MVC-NNT, and the hybrid defense.
The SME approach achieved 47.75% PGD-20 robust accuracy, while MVC-NNT achieved 45.20% AutoAttack robust accuracy.
Gradient Diversity
Gradient coherence between MVC-trained expert pairs remained below 0.14, indicating diverse gradient landscapes across the expert models.
Technologies
- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Jupyter Notebook
- ResNet-18
- CIFAR-10
- Adversarial Training
- Exponential Particle Swarm Optimization
- Gumbel-Softmax
