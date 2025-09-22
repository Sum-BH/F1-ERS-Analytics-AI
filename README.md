# F1 ERS Deployment Analytics — Portfolio Repo

![PPO Rewards](assets/screenshot.png)


Minimal, self-contained project showcasing:
- **DS**: synthetic telemetry generation, feature engineering, lap summaries.
- **ML**: basic data work and feature exports for modeling.
- **RL**: PPO agent learning per-lap ERS deployment (with reward shaping & logging).
- **RAG & GenAI**: FAISS-based retrieval + optional local LLM (ctransformers).
- **UI**: Gradio dashboard to explore outputs and query the assistant.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## What you'll get
- `src/` modular package with reproducible components:
  - `data_gen.py`, `features.py`, `env.py`, `train.py`, `rag.py`, `llm.py`
- `app.py` — launches the Gradio dashboard
- `outputs/` and `models/` (created on run)
- `notebooks/demo.ipynb` — a runnable notebook that walks through the pipeline
- `.github/workflows/ci.yml` — CI workflow (lint & tests)
- `tests/` — minimal tests for features

## Quick start (recommended)
1. Create virtualenv and activate:
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the pipeline step-by-step (recommended):
```bash
python -m src.data_gen
python -m src.features
python -m src.train     # may take time depending on timesteps
python -m src.rag
python app.py
```

## Notes
- LLM usage is optional. If you have a local GGUF model and `ctransformers` installed, place the model file under `models/local_llm/` and `src/llm.py` will use it.
- The repository is prepared for offline/local demos and to be uploaded to GitHub as a showcase repo.

---