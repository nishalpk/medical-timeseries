import main

class MockGraphRAG:
    def __init__(self, *args, **kwargs):
        pass

main.PrimeKGGraphRAG = MockGraphRAG

if __name__ == "__main__":
    PIPELINE = main.NeurosymbolicPipeline(
        mira_ckpt="mira/checkpoints",
        kg_path="kg.csv",
        gemini_key="dummy_key"
    )
    print("Testing training MoE system...")
    PIPELINE.train_moe_system("train_mira_100.jsonl", "val_mira_20.jsonl", "embeddings", epochs=2)
    print("Training test completed successfully.")
