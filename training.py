
import os
from dotenv import load_dotenv
load_dotenv()

from main import NeurosymbolicPipeline


PIPELINE = NeurosymbolicPipeline(
        mira_ckpt="mira/checkpoints",
        kg_path="kg.csv",
        gemini_key=os.getenv("GEMINI_API")
    )

PIPELINE.train_moe_system(
    train_jsonl="train_mira_100.jsonl",
    val_jsonl="val_mira_20.jsonl",
    embed_dir="embeddings")