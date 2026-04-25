import pandas as pd
import networkx as nx
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class PrimeKGGraphRAG:
    """
    Optimized Graph-RAG using Pre-computed Semantic Embeddings.
    Instant-load version for high-frequency clinical inference.
    """
    def __init__(self, kg_path, index_path="primekg.index", nodes_path="primekg_nodes.pkl"):
        print("⚡ Loading Pre-computed Knowledge Index...")
        # 1. Load the Graph Topology (for the 'Graph' part of RAG)
        df = pd.read_csv(kg_path, low_memory=False)
        self.G = nx.from_pandas_edgelist(
            df, source='x_name', target='y_name', edge_attr=['display_relation']
        )
        
        # 2. Load Semantic Index (for the 'Retrieval' part of RAG)
        self.index = faiss.read_index(index_path)
        with open(nodes_path, "rb") as f:
            self.node_names = pickle.load(f)
            
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Graph-RAG Ready.")

    def semantic_lookup(self, query, top_k=2):
        """Instant semantic search across PrimeKG nodes."""
        query_vec = self.embedder.encode([query]).astype('float32')
        _, indices = self.index.search(query_vec, top_k)
        return [self.node_names[i] for i in indices[0]]

    def run_retrieval(self, vital_name, condition="Sepsis"):
        """
        Retrieves the local clinical subgraph.
        """
        # Find entry points semantically
        vital_nodes = self.semantic_lookup(vital_name, top_k=2)
        condition_nodes = self.semantic_lookup(condition, top_k=1)
        
        all_entry_points = vital_nodes + condition_nodes
        subgraph_triplets = []

        # Extract 1-hop neighborhoods to build context
        for node in all_entry_points:
            if node in self.G:
                for neighbor in self.G.neighbors(node):
                    rel = self.G[node][neighbor]['display_relation']
                    subgraph_triplets.append(f"{node} {rel} {neighbor}")
        
        # Linearize for the MoE Router
        verifiable_text = " . ".join(list(set(subgraph_triplets))[:8])
        return {
            "vital": vital_name,
            "context": condition,
            "verifiable_text": verifiable_text if verifiable_text else "General clinical monitoring."
        }
    
    def run_retrieval(self, vital_name, condition="Sepsis"):
        vital_nodes = self.semantic_lookup(vital_name, top_k=2)
        condition_nodes = self.semantic_lookup(condition, top_k=1)
        
        all_entry_points = vital_nodes + condition_nodes
        subgraph_triplets = []

        # Target Node Types for Clinical Relevance
        target_types = ['disease', 'phenotype', 'biological_process', 'pathway']

        for node in all_entry_points:
            if node in self.G:
                # Look at neighbors and filter for useful medical context
                for neighbor in self.G.neighbors(node):
                    # We can check node type if your kg.csv has an 'x_type' column
                    rel = self.G[node][neighbor]['display_relation']
                    subgraph_triplets.append(f"{node}-> {rel}-> {neighbor}")
        
        # Linearize and CLEAN: Remove environmental noise
        clean_triplets = [t for t in subgraph_triplets if "Nicotine" not in t and "Pollutants" not in t]
        verifiable_text = "\n".join(list(set(clean_triplets))[:10])
        
        return verifiable_text

# # --- TEST ---
# if __name__ == "__main__":
#     rag = PrimeKGGraphRAG("kg.csv")
#     res = rag.run_retrieval("mean arterial pressure", "Septic Shock")
#     print(f"\n🧬 RETRIEVED KNOWLEDGE:\n{res}")