# src/llm.py
import os, json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from .configs import RAG_DIR, LLM_FOLDER, LLM_FILE
try:
    from ctransformers import AutoModelForCausalLM
except Exception:
    AutoModelForCausalLM = None

emb = SentenceTransformer('all-MiniLM-L6-v2')

def local_llm_answer(question, use_local=True, max_tokens=120):
    idx_path = Path(RAG_DIR)/'index.faiss'
    docs_path = Path(RAG_DIR)/'docs.json'
    if not idx_path.exists() or not docs_path.exists():
        return '⚠️ RAG index missing. Build it with `python -m src.rag`.'
    index = faiss.read_index(str(idx_path))
    with open(docs_path,'r',encoding='utf-8') as f:
        docs = json.load(f)
    q_emb = emb.encode([question], convert_to_numpy=True).astype('float32')
    D,I = index.search(q_emb, k=min(3, len(docs)))
    retrieved = [docs[i] for i in I[0] if i < len(docs)]
    context = ' '.join(retrieved)
    prompt = f"You are an F1 race engineer. Context: {context}\nQuestion: {question}\nAnswer:"
    if use_local and AutoModelForCausalLM is not None:
        model_path = Path(LLM_FOLDER)/LLM_FILE
        if model_path.exists():
            try:
                llm = AutoModelForCausalLM.from_pretrained(str(model_path), model_type='mistral')
                out = llm(prompt, max_new_tokens=max_tokens)
                return str(out)
            except Exception as e:
                return f'⚠️ Local LLM failed: {e}'
        else:
            return '⚠️ Local LLM model file not found. Place your GGUF file in models/local_llm/'
    return '⚠️ Local LLM not available. Install ctransformers and provide a GGUF model.'
