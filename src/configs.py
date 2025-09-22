from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
RAG_DIR = BASE_DIR / "rag_index"
DATA_CSV = OUTPUT_DIR / "powerbi_export.csv"
MODELS_DIR = BASE_DIR / "models"
LLM_FOLDER = MODELS_DIR / "local_llm"   # place GGUF or ctransformers-compatible files here
LLM_FILE = "mistral-7b-instruct-v0.1.Q5_K_M.gguf"
OUTPUT_DIR.mkdir(exist_ok=True)
RAG_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
