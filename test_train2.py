from main import NeurosymbolicPipeline
import graph_rag

# Mock GraphRAG so it doesn't load 1GB CSV and FAISS
class MockGraphRAG:
    def __init__(self, *args, **kwargs):
        pass

graph_rag.PrimeKGGraphRAG = MockGraphRAG

import os

if __name__ == "__main__":
    PIPELINE = NeurosymbolicPipeline(
        mira_ckpt="mira/checkpoints",
        kg_path="dummy.csv",
        gemini_key="dummy_key"
    )
    print("Testing training MoE system...")
    PIPELINE.train_moe_system("train_mira_100.jsonl", "val_mira_20.jsonl", "embeddings", epochs=2)
    print("Training test completed successfully.")
