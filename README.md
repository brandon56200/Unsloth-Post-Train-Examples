# 🔧 Post-Training Pipelines with Unsloth

### **Supervised Fine-Tuning (SFT), QLoRA, RLVR, and GRPO on Qwen Models**

Modern LLMs rely heavily on *post-training* techniques such as
**Supervised Fine-Tuning (SFT)**, **parameter-efficient adapters
(LoRA/QLoRA)**, and increasingly, **reinforcement learning with
verifiable rewards (RLVR)**. These methods are at the core of how models
learn structured behavior, reasoning skills, and domain-specific
capabilities.

This repository contains two hands-on Colab notebooks that demonstrate
these concepts using the **Unsloth** training framework and the **Qwen**
family of models. The goal is to provide clear, reproducible examples of
how to build practical post-training pipelines that mirror how modern AI
companies train production-grade models.

------------------------------------------------------------------------

# 📘 Notebook 1 --- Supervised Fine-Tuning with QLoRA

### `SFT_QLoRA_Ex.ipynb`

This notebook walks through how to fine-tune **Qwen-14B (4-bit
quantized)** using **Supervised Fine-Tuning (SFT)** with **QLoRA**
adapters.

## 🧠 What SFT Does

Supervised Fine-Tuning teaches a model to mimic desirable human
instruction-following behavior.\
All instruct or reasoning models start as base models and are shaped
using SFT.

-   **Base Model:** Qwen-14B\
-   **Dataset:** FineTome-100k\
-   **Goal:** Build an instruction-following model

------------------------------------------------------------------------

## 🛠 What QLoRA Does

QLoRA extends LoRA by:

-   Freezing base model weights\
-   Injecting low-rank trainable adapters\
-   Targeting key attention tensors (Q, K, V)\
-   Quantizing the base weights to 4-bit

This dramatically reduces compute and memory while still enabling
high‑quality behavioral training.

------------------------------------------------------------------------

# 📗 Notebook 2 --- RLVR + GRPO for Skill-Based Training

### `RLVR_GRPO_Ex.ipynb`

This notebook demonstrates a full two‑phase pipeline:

1.  **SFT + LoRA** to teach structured reasoning\
2.  **RLVR via GRPO** to teach domain-specific *skills* using verifiable
    feedback

------------------------------------------------------------------------

## Phase 1 --- LoRA/SFT ("Teach the Model to Think")

-   **Base Model:** Qwen-4B\
-   **Dataset:** OpenMathReasoning-mini\
-   **Goal:** Establish reasoning structure + chain‑of‑thought patterns

------------------------------------------------------------------------

## Phase 2 --- RLVR + GRPO ("Teach the Model a Skill")

### 🧮 What RLVR Does

RLVR uses objective, verifiable signals (right/wrong answers) to teach
*skills*, not just behavior.

-   **Dataset:** DAPO-Math-17k-Processed\
-   **Rewards:** Formatting correctness + Numerical accuracy\
-   **Goal:** Improve math reasoning ability through verifiable reward
    functions

------------------------------------------------------------------------

### ⚙️ What GRPO Does

GRPO is DeepSeek's reinforcement learning algorithm designed to expand
PPO for LLMs.

It provides:

-   Better stability\
-   Support for multi‑objective reward signals\
-   Alignment with verifiable, domain-specific tasks

------------------------------------------------------------------------

# 📦 Summary

These notebooks provide practical, end‑to‑end templates for:

-   SFT with QLoRA on quantized models\
-   Chain‑of‑thought behavior modeling\
-   RLVR‑based skill alignment\
-   GRPO-based reinforcement learning pipelines

Together, they represent the core components of modern LLM post‑training
workflows.
