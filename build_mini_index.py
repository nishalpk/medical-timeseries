import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

def build_mini_index(kg_path, out_prefix="primekg"):
    print("🏗️ Loading subset of PrimeKG...")
    df = pd.read_csv(kg_path, low_memory=False, nrows=1000) 
    
    nodes_raw = pd.concat([df['x_name'], df['y_name']]).unique()
    nodes = [str(n) for n in nodes_raw if pd.notna(n)]
    
    if "MAP" not in nodes: nodes.append("MAP")
    if "sepsis" not in nodes: nodes.append("sepsis")
    
    print(f"💎 Nodes to index: {len(nodes)}")
    
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    embeddings = model.encode(nodes, batch_size=32)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    faiss.write_index(index, f"{out_prefix}.index")
    
    with open(f"{out_prefix}_nodes.pkl", "wb") as f:
        pickle.dump(nodes, f)
        
    print(f"✅ Mini Index Saved.")

if __name__ == "__main__":
    build_mini_index("kg.csv")
