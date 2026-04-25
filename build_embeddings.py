import pandas as pd
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

def build_static_index(kg_path, out_prefix="primekg"):
    print("🏗️ Loading PrimeKG and generating embeddings...")
    df = pd.read_csv(kg_path, low_memory=False)
    
    # 1. Extract and Clean Nodes
    # Convert to standard Python strings to avoid the ArrowStringArray error
    nodes_raw = pd.concat([df['x_name'], df['y_name']]).unique()
    nodes = [str(n) for n in nodes_raw if pd.notna(n)]
    
    print(f"💎 Total unique nodes to index: {len(nodes)}")
    
    # 2. Generate Embeddings (Standard strings are now guaranteed)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(nodes, show_progress_bar=True, batch_size=128)
    
    # 3. Save the FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    faiss.write_index(index, f"{out_prefix}.index")
    
    # 4. Save the Node List
    with open(f"{out_prefix}_nodes.pkl", "wb") as f:
        pickle.dump(nodes, f)
        
    print(f"✅ Static Index Saved successfully.")

if __name__ == "__main__":
    build_static_index("kg.csv")