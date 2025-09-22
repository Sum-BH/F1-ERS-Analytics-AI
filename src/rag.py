# src/rag.py
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from .configs import RAG_DIR, DATA_CSV

def build_rag():
    df = pd.read_csv(DATA_CSV)
    events = []
    grouped = df.groupby('lap').agg(total_deploy=('deployed','sum'), total_harvest=('harvested','sum')).reset_index()
    for _, r in grouped.iterrows():
        events.append(f"Lap {int(r['lap'])}: deploy {r['total_deploy']:.3f} MJ, harvest {r['total_harvest']:.3f} MJ.")
    if not events:
        events = ['No deployment events detected.']

    emb = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = emb.encode(events, convert_to_numpy=True).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    Path(RAG_DIR).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(Path(RAG_DIR)/'index.faiss'))
    with open(Path(RAG_DIR)/'docs.json','w',encoding='utf-8') as f:
        json.dump(events, f)
    print('Built FAISS RAG index in rag_index/')

if __name__ == '__main__':
    build_rag()
