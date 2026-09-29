# VisualStep

**Fine-tuned Phi-3 Mini 3.8B for ADHD-adapted visual step card generation in primary classrooms.**

VisualStep converts spoken teacher instructions into structured, emoji-annotated step cards designed to support students with ADHD and executive function difficulties in Years 1–6 classrooms.

---

## Overview

Children with ADHD struggle to retain multi-step verbal instructions. VisualStep addresses this by automatically decomposing teacher speech into a maximum of five short, imperative steps — each paired with a visual emoji — formatted as a JSON card displayed on a classroom screen or printed card.

**Example:**

> *"Grab your maths book, turn to page 45, and write your answers in your exercise book."*

```json
{
  "steps": [
    {"id": 1, "emoji": "📖", "action": "Grab your maths book"},
    {"id": 2, "emoji": "📄", "action": "Turn to page 45"},
    {"id": 3, "emoji": "✏️", "action": "Write your answers"}
  ]
}
```

---

## Model

- **Base model:** [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- **Fine-tuning:** QLoRA (r=16, α=32), single NVIDIA T4 GPU (Google Colab free tier)
- **HuggingFace model:** [lsadouk1111/VisualStep](https://huggingface.co/lsadouk1111/VisualStep)
- **Live demo:** [lsadouk1111/VisualStep-app](https://huggingface.co/spaces/lsadouk1111/VisualStep-app)

---

## Dataset — VisualStep-2K

2,000 instruction–card pairs:
- 1,000 **real** transcripts from Oak National Academy lessons (Years 1–6, English/Maths/Science)
- 1,000 **synthetic** pairs generated with GPT-4o-mini

| Split | Size |
|-------|------|
| Train | 1,600 |
| Validation | 200 |
| Test | 200 |

Year distribution: Y1=421, Y2=517, Y3=217, Y4=357, Y5=239, Y6=249  
Subject distribution: English=816, Science=618, Maths=566  
Instruction types: Task=1,416, Routine=225, Discussion=178, Transition=173

---

## Results

Evaluated on the 200-item held-out test set against three zero-shot baselines.

| Model | BLEU-4 | ROUGE-1 | ROUGE-L | ACI% | Valid JSON |
|-------|--------|---------|---------|------|------------|
| **VisualStep (ours)** | **56.40** | **0.769** | **0.748** | **91.0%** | 199/200 |
| Zero-shot Gemma 2 2B | 15.71 | 0.471 | 0.427 | 68.5% | 200/200 |
| Zero-shot Phi-3 Mini | 8.99 | 0.437 | 0.401 | 21.5% | 196/200 |
| Zero-shot Qwen3.8 27B | 7.93 | 0.463 | 0.420 | 44.0% | 193/200 |

**ACI constraints:** max 5 steps (C), ≤6 words/action (A), imperative verb (I).

**Human evaluation (n=44, two domain experts):** Cohen's κ = 0.756 (substantial agreement), mean rating 1.75/2.

---

## Repository Structure

```
VisualStep/
├── data/
│   └── test_outputs_visualstep_corrected.csv   # Test set with model outputs
├── evaluation/
│   ├── human_evaluation.csv                    # Two-examiner annotation (κ=0.756)
│   └── run_metrics.py                          # BLEU-4, ROUGE, ACI metrics script
├── app/
│   └── app.py                                  # Gradio HuggingFace Spaces app
└── README.md
```

---

## Reproduce Metrics

```bash
pip install sacrebleu rouge-score pandas numpy
python evaluation/run_metrics.py
```

Place all model CSV files (`test_outputs_*.csv`) in the same directory as the script.

---

## Citation

> Sadouk, L. (2026). *VisualStep: Fine-tuned Phi-3 Mini for ADHD-adapted visual step card generation in primary classrooms.* EMSI Casablanca.

---

## Author

**Lamyaa Sadouk** — AI/ML Researcher & Professor, EMSI Casablanca  
GitHub: [@lsadouk](https://github.com/lsadouk) · HuggingFace: [lsadouk1111](https://huggingface.co/lsadouk1111)
