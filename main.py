import torch
import json
from mira_fm import MIRASystem
from graph_rag import PrimeKGGraphRAG
# from graph_rag import HierarchicalGraphRAG
from model.moe_system import FullSystemMoE
from utils.gemini_client import MedicalEmbeddingClient
from dotenv import load_dotenv
import os

load_dotenv()

class NeurosymbolicPipeline:
    def __init__(self, mira_ckpt, kg_path, gemini_key):
        print("🛠️ Initializing Full Neurosymbolic Pipeline...")
        # 1. Neural Component (Nishal's MIRA)
        self.mira = MIRASystem(mira_ckpt)
        
        # 2. Symbolic Component (Graph-RAG)
        # self.graph_rag = HierarchicalGraphRAG(kg_path)
        self.graph_rag = PrimeKGGraphRAG(kg_path)
        
        # 3. Decision Component (MoE System)
        self.gemini = MedicalEmbeddingClient(gemini_key)
        self.moe = FullSystemMoE(text_embed_dim=3072).to("cuda").eval()

    def run_inference(self, patient_jsonl_line):
        """
        Runs a single patient case through the entire Neurosymbolic chain.
        """
        data = json.loads(patient_jsonl_line)
        itemid = str(data['itemid'])
        
        # --- STEP 1: NEURAL ENCODING (MIRA) ---
        # Extract the high-dimensional latent representation and forecast
        seq = torch.tensor([data['sequence']], dtype=torch.float32).to("cuda")
        times = torch.tensor([data['time']], dtype=torch.float32).to("cuda")
        
        if seq.dim() == 2:
            seq = seq.unsqueeze(-1)

        # MIRA produces the 'Clinical State' [1, 384]
        latent_vector, mira_forecast = self.mira.get_mira_outputs(seq, times)

        # --- STEP 2: SYMBOLIC RETRIEVAL (Graph-RAG) ---
        # Get the 'Verifiable Medical Knowledge' path from PrimeKG
        rag_output = self.graph_rag.run_retrieval(itemid, "Patient suspected Sepsis")
        # clinical_path = " -> ".join(rag_output['subgraph'])
        clinical_path = rag_output
        
        # --- STEP 3: SEMANTIC EMBEDDING (Gemini) ---
        # Convert the clinical path into a 3072-dim text embedding for the Router
        # print(f"🧬 Retrieved Knowledge: {clinical_path}")
        text_emb = self.gemini.get_embedding(clinical_path)

        text_emb = text_emb.to("cuda")

        # --- STEP 4: NEURO-SYMBOLIC FUSION (MoE) ---
        # The MoE Router uses the text_emb to decide which expert processes the latent_vector
        with torch.no_grad():
            # We pass the latent_vector from MIRA directly into the MoE system
            risk_score, final_forecast, weights = self.moe(latent_vector, text_emb)


        return {
            "risk": risk_score.item(),
            "forecast": final_forecast,
            "time_series_forecast": mira_forecast.cpu().numpy(),
            "expert_weights": weights.tolist(),
            "clinical_justification": clinical_path
        }

# --- FINAL SUBMISSION EXECUTION ---
if __name__ == "__main__":
    # Parameters
    PIPELINE = NeurosymbolicPipeline(
        mira_ckpt="mira/checkpoints",
        kg_path="kg.csv",
        gemini_key=os.getenv("GEMINI_API")
    )

    # Test with a sample from your MIMIC-IV dataset
    sample_case = '{"vital_name": "MAP", "sequence": [70, 68, 65, 62], "time": [0, 1, 2, 3]}'
    
    result = PIPELINE.run_inference(sample_case)

    print("\n" + "="*50)
    print(f"🚀 FINAL SYSTEM OUTPUT")
    print(f"Clinical Risk: {result['risk']*100:.2f}%")
    print(f"Expert Routing: {result['expert_weights']}")
    print(f"Justification: {result['clinical_justification']}")
    print(f"Time-Series Forecast (MIRA): {result['time_series_forecast']}")
    print("="*50)

