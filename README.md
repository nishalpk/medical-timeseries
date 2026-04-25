# 🏥 Neurosymbolic MoE for Predictive ICU Monitoring

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![LLM: Gemini](https://img.shields.io/badge/LLM-Gemini--Embedding--2-green.svg)](https://ai.google.dev/)

A hybrid AI architecture that combines **GraphRAG-derived medical knowledge** with **deep temporal encoding** to predict patient mortality risk and forecast vital signs.

## 🌟 The Innovation: "Doctor-in-the-loop" AI
Standard ICU models treat every patient as a set of numbers. This project introduces a **Neurosymbolic Mixture of Experts (MoE)**:
- **Symbolic Logic:** Uses GraphRAG context (from the Gemini LLM) to understand the "why" (e.g., *Sepsis caused by Norepinephrine resistance*).
- **Neural Power:** Uses a Bi-GRU with Multi-Head Attention to process 48 hours of raw vitals.
- **Dynamic Routing:** The LLM acts as a "Triage Nurse," steering the patient's data to the clinical expert (Hemodynamic, Biochemical, or Generalist) best suited for that specific diagnosis.



## 🏗️ System Architecture
The pipeline is split into three distinct modules:

1. **Contextual Embedding (LLM):** Converts GraphRAG text into a high-dimensional (3072-dim) vector using `gemini-embedding-2`.
2. **Temporal Encoder:** A 2-layer Bi-Directional GRU that extracts features from 48-hour vital sign windows.
3. **MoE Gating Network:** A neural router that processes the LLM embedding to generate "specialization weights," determining the final output by fusing specialist opinions.

## 📂 Project Structure
```text
mira-moe-project/
├── models/
│   ├── encoder.py       # Bi-GRU & Attention for Vitals
│   ├── experts.py       # Clinical Specialist Layers (Hemo/Bio/Gen)
│   └── moe_system.py    # Gating Network & Weighted Fusion Logic
├── utils/
│   └── gemini_client.py # Gemini API & Embedding handlers
├── main1.py              # End-to-end inference pipeline
├── requirements.txt     # Dependency list
└── README.md            # You are here


🚀 Getting Started
1. Requirements
A Google AI Studio API Key.

Real ICU data (MIRA/MIMIC-IV format) or the provided mock data.

2. Installation
git clone [https://github.com/YOUR_GITHUB_USERNAME/mira-moe-project.git](https://github.com/YOUR_GITHUB_USERNAME/mira-moe-project.git)
cd mira-moe-project
pip install -r requirements.txt

3. Usage
Run the end-to-end inference test:

Python
python main.py
📊 Outputs & Explainability
Unlike "black box" models, this system provides Clinical Transparency:

Mortality Risk: Probability of clinical deterioration.

Vital Forecast: A 24-hour predictive window for 5 key vitals.

Routing Weights: See exactly which "Expert" the AI trusted (e.g., If Routing = 0.85 Hemodynamic, the model is focusing heavily on cardiovascular stability).

🛠️ Tech Stack
Deep Learning: PyTorch

LLM / NLP: Google Gemini (Generative AI SDK)

Knowledge Representation: GraphRAG (Medical Context)

Time-Series: Bi-GRU + Multi-Head Attention

⚠️ Security Note
Never commit your API_KEY to GitHub. Use an environment variable or a .env file to keep your credentials secure.

Developed for advanced clinical decision support in ICU environments.


---
