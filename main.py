import torch
import json
from mira_fm import MIRASystem
from graph_rag import PrimeKGGraphRAG
# from graph_rag import HierarchicalGraphRAG
from model.moe_system import FullSystemMoE
from utils.gemini_client import MedicalEmbeddingClient
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
import os
import torch
import torch.nn as nn
import torch.optim as optim

load_dotenv()

class SymbolicLayer:
    def __init__(self):
        #qSOFA and Sepsis Logic Implementation 
        self.rules = {
                # Vitals (Standard Thresholds)
                "220052": {"name": "MAP", "min": 65},          # Sepsis-3 threshold [cite: 44]
                "220045": {"name": "Heart Rate", "max": 100},  # Tachycardia [cite: 39]
                "220210": {"name": "Resp Rate", "max": 22},    # qSOFA criteria [cite: 28]
                "220277": {"name": "SpO2", "min": 90},         # Hypoxia threshold
                "223762": {"name": "Temp (C)", "range": (36, 38)}, 
                
                # Labs (Organ Failure / SOFA markers) [cite: 69, 133]
                "50813": {"name": "Lactate", "max": 2.0},      # Septic Shock marker
                "50820": {"name": "pH", "min": 7.35},          # Acidosis
                "50912": {"name": "Creatinine", "max": 1.2},   # Renal SOFA
                "50885": {"name": "Bilirubin", "max": 1.2},    # Hepatic SOFA
                "51265": {"name": "Platelets", "min": 150},    # Coagulation SOFA
                "51301": {"name": "WBC", "range": (4, 12)},    # Infection marker
                "50931": {"name": "Glucose", "range": (70, 180)}
            }
            
            # Pressor IDs for Refractory Shock Logic [cite: 19, 44]
        self.pressors = ["221289", "221662", "221749", "221906", "222315"]

    @staticmethod
    def _to_scalar(value):
        """Convert nested/tensor prediction values to a single float."""
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            return float(value.reshape(-1)[0].item())

        while isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None
            value = value[0]

        try:
            return float(value)
        except (TypeError, ValueError):
            return None
        
    def check_violation(self, itemid, forecast_val, patient_meds=None):
        """Returns 1 if a clinical law is violated, else 0."""
        rule = self.rules.get(itemid)
        if not rule: return 0

        forecast_val = self._to_scalar(forecast_val)
        if forecast_val is None:
            return 0
        
        violation = 0
        # Basic threshold checks
        if "min" in rule and forecast_val < rule["min"]: violation = 1
        if "max" in rule and forecast_val > rule["max"]: violation = 1
        if "range" in rule:
            if forecast_val < rule["range"][0] or forecast_val > rule["range"][1]:
                violation = 1
                
        # Advanced Logic: MAP < 65 while on Pressors indicates shock [cite: 19, 44]
        if itemid == "220052" and forecast_val < 65:
            if any(m in (patient_meds or []) for m in self.pressors):
                violation = 1 # High-risk refractory state detected
                
        return violation

    def calculate_vcc(self, patient_results):
        """
        Automates Vcc calculation across N patients.
        Vcc = Total Violations / Total Predictions.
        """
        total_predictions = 0
        total_violations = 0
        
        for record in patient_results:
            itemid = str(record['itemid'])
            forecasts = record['forecast'] # MIRA's statistical output 
            meds = record.get('current_meds', [])
            
            for val in forecasts:
                total_predictions += 1
                total_violations += self.check_violation(itemid, val, meds)
                
        vcc_rate = total_violations / total_predictions if total_predictions > 0 else 0
        return vcc_rate, total_violations, total_predictions
    

class SepsisMoEDataset(Dataset):
    def __init__(self, jsonl_path, embedding_dir):
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.embedding_dir = embedding_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # Pre-computed Gemini Embeddings are essential for training speed
        emb_path = os.path.join(self.embedding_dir, f"{item['stay_id']}_{item['itemid']}.pt")
        # Fallback to random if file doesn't exist to prevent crash during testing
        if os.path.exists(emb_path):
            text_emb = torch.load(emb_path)
        else:
            text_emb = torch.randn(1, 3072)
            
        return {
            "vitals": torch.tensor(item['sequence'], dtype=torch.float32).unsqueeze(-1),
            "times": torch.tensor(item['time'], dtype=torch.float32),
            "text_emb": text_emb, # [3072]
            "label": torch.tensor([item['label']], dtype=torch.float32),
            "itemid": item['itemid']
        }

class NeurosymbolicPipeline:
    # Pillar IV: Expert Domain Mapping [cite: 81-82, 171]

    def __init__(self, mira_ckpt, kg_path, gemini_key):
        print("🛠️ Initializing Full Neurosymbolic Pipeline...")
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        # 1. Neural Component (Nishal's MIRA)
        self.mira = MIRASystem(mira_ckpt, device=self.device)
        
        # 2. Symbolic Component (Graph-RAG)
        # self.graph_rag = HierarchicalGraphRAG(kg_path)
        self.graph_rag = PrimeKGGraphRAG(kg_path)
        
        # 3. Decision Component (MoE System)
        self.gemini = MedicalEmbeddingClient(gemini_key)
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.moe = FullSystemMoE(text_embed_dim=3072).to(self.device).eval()
        self.expert_map = {
        # Hemodynamic Specialist (Expert 0)
        "220045": 0, "220052": 0, "220210": 0, "221906": 0, "221289": 0, "222315": 0, "221662": 0, "221749": 0,
        # Biochemical Specialist (Expert 1)
        "50813": 1, "50912": 1, "50885": 1, "51265": 1, "51301": 1, "50820": 1,
        # Generalist Specialist (Expert 2)
        "220277": 2, "223762": 2, "50931": 2
    }

    def run_inference(self, patient_jsonl_line):
        """
        Runs a single patient case through the entire Neurosymbolic chain.
        """
        data = json.loads(patient_jsonl_line)
        itemid = str(data['vital_name'])
        itemidtrue = str(data['item_id']) # Use original itemid if available for evaluation
        
        # --- STEP 1: NEURAL ENCODING (MIRA) ---
        # Extract the high-dimensional latent representation and forecast
        seq = torch.tensor([data['sequence']], dtype=torch.float32).to(self.device)
        times = torch.tensor([data['time']], dtype=torch.float32).to(self.device)
        
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

        text_emb = text_emb.to(self.device)

        # --- STEP 4: NEURO-SYMBOLIC FUSION (MoE) ---
        # The MoE Router uses the text_emb to decide which expert processes the vitals
        with torch.no_grad():
            # We pass the raw vitals directly into the MoE system which now has its own encoder
            risk_score, final_forecast, weights = self.moe(seq, text_emb)

        # itemidtrue = data.get('itemid', itemid)  # Use original itemid if available for evaluation 
        SymbolicEvaluator = SymbolicLayer()
        patient_results = [{
            "itemid": itemidtrue,
            "forecast": mira_forecast.cpu().numpy()[0].tolist(),
            "current_meds": data.get('current_meds', [])
        }]
        vcc_rate, total_violations, total_predictions = SymbolicEvaluator.calculate_vcc(patient_results)

        return {
            "risk": risk_score.item(),
            "forecast": final_forecast,
            "time_series_forecast": mira_forecast.cpu().numpy(),
            "expert_weights": weights.tolist(),
            "clinical_justification": clinical_path,
            "vcc_rate": vcc_rate,
            "total_violations": total_violations,
            "total_predictions": total_predictions
        }
    import torch.nn.utils.rnn as rnn_utils

    @staticmethod
    def sepsis_safe_collate(batch, context_window=64):
        """
        Ensures all tensors in a batch are the same size for Pillar IV training.
        Pads shorter sequences and truncates longer ones to the fixed Context Window (C=64).
        """
        vitals = [item['vitals'] for item in batch]
        times = [item['times'] for item in batch]
        # Make sure text_emb is 1D before stacking
        text_embs = torch.stack([item['text_emb'].view(-1) for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        itemids = [item['itemid'] for item in batch]

        # 1. Pad/Truncate Vitals to [Batch, 64, 1]
        # We use 'post' padding to keep the temporal order correct
        vitals_padded = []
        for v in vitals:
            if v.size(0) > context_window:
                vitals_padded.append(v[:context_window])
            else:
                padding = torch.zeros(context_window - v.size(0), v.size(1))
                vitals_padded.append(torch.cat([v, padding], dim=0))
        
        # 2. Pad/Truncate Times to [Batch, 64]
        times_padded = []
        for t in times:
            if t.size(0) > context_window:
                times_padded.append(t[:context_window])
            else:
                padding = torch.zeros(context_window - t.size(0))
                times_padded.append(torch.cat([t, padding], dim=0))

        return {
            "vitals": torch.stack(vitals_padded),
            "times": torch.stack(times_padded),
            "text_emb": text_embs,
            "label": labels,
            "itemid": itemids
        }
    
    def train_moe_system(self, train_jsonl, val_jsonl, embed_dir, epochs=10):
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.moe.to(device)
        optimizer = optim.AdamW(self.moe.parameters(), lr=5e-5, weight_decay=1e-2)
        
        criterion_risk = nn.BCELoss()
        criterion_forecast = nn.MSELoss()
        criterion_routing = nn.CrossEntropyLoss() # To force specialization

        train_loader = DataLoader(
            SepsisMoEDataset(train_jsonl, embed_dir),
            batch_size=32, shuffle=True, drop_last=True,
            collate_fn=self.sepsis_safe_collate,
        )
        
        val_loader = DataLoader(
            SepsisMoEDataset(val_jsonl, embed_dir),
            batch_size=32, shuffle=False, drop_last=False,
            collate_fn=self.sepsis_safe_collate,
        )

        for epoch in range(epochs):
            self.moe.train()
            epoch_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()
                vitals = batch['vitals'].to(device)
                
                # --- FIX 1: TARGET NORMALIZATION ---
                # Standardizing targets to Z-score space to stop high losses [cite: 58-60]
                mean = vitals.mean(dim=1, keepdim=True)
                std = vitals.std(dim=1, keepdim=True) + 1e-6
                vitals_norm = (vitals - mean) / std

                risk_pred, forecast_pred, weights = self.moe(vitals, batch['text_emb'].to(device))

                # --- FIX 2: SUPERVISED ROUTING ---
                # Assigning target experts based on clinical item ID [cite: 90, 156]
                target_experts = torch.tensor([self.expert_map.get(str(i), 2) for i in batch['itemid']]).to(device)
                loss_route = criterion_routing(weights, target_experts)

                # Task Losses
                loss_r = criterion_risk(risk_pred, batch['label'].to(device))
                loss_f = criterion_forecast(forecast_pred[:, :24, 0], vitals_norm[:, :24, 0])
                
                # Combine losses (Supervised Routing + Forecast + Risk)
                total_loss = loss_r + (0.5 * loss_f) + (0.2 * loss_route)
                
                total_loss.backward()
                
                # Fix Explosion of Gradients
                torch.nn.utils.clip_grad_norm_(self.moe.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += total_loss.item()

            train_loss = epoch_loss / len(train_loader)
            
            # --- EVALUATION METRICS ---
            self.moe.eval()
            val_loss = 0
            correct_preds = 0
            total_preds = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    vitals = batch['vitals'].to(device)
                    mean = vitals.mean(dim=1, keepdim=True)
                    std = vitals.std(dim=1, keepdim=True) + 1e-6
                    vitals_norm = (vitals - mean) / std
                    
                    risk_pred, forecast_pred, weights = self.moe(vitals, batch['text_emb'].to(device))
                    
                    target_experts = torch.tensor([self.expert_map.get(str(i), 2) for i in batch['itemid']]).to(device)
                    
                    loss_route = criterion_routing(weights, target_experts)
                    loss_r = criterion_risk(risk_pred, batch['label'].to(device))
                    loss_f = criterion_forecast(forecast_pred[:, :24, 0], vitals_norm[:, :24, 0])
                    
                    batch_loss = loss_r + (0.5 * loss_f) + (0.2 * loss_route)
                    val_loss += batch_loss.item()
                    
                    # Accuracy for risk prediction (0.5 threshold)
                    predictions = (risk_pred >= 0.5).float()
                    correct_preds += (predictions == batch['label'].to(device)).sum().item()
                    total_preds += batch['label'].size(0)
                    
            val_loss = val_loss / len(val_loader)
            val_acc = (correct_preds / total_preds) * 100 if total_preds > 0 else 0
            
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# --- FINAL SUBMISSION EXECUTION ---
if __name__ == "__main__":
    # Parameters
    PIPELINE = NeurosymbolicPipeline(
        mira_ckpt="mira/checkpoints",
        kg_path="kg.csv",
        gemini_key=os.getenv("GEMINI_API")
    )

    # Test with a sample from your MIMIC-IV dataset
    sample_case = '{"item_id":"220052","vital_name": "MAP", "sequence": [70, 68, 65, 62], "time": [0, 1, 2, 3]}'
    
    result = PIPELINE.run_inference(sample_case)

    print("\n" + "="*50)
    print(f"🚀 FINAL SYSTEM OUTPUT")
    print(f"Clinical Risk: {result['risk']*100:.2f}%")
    print(f"Expert Routing: {result['expert_weights']}")
    print(f"Justification: {result['clinical_justification']}")
    print(f"Time-Series Forecast (MIRA): {result['time_series_forecast']}")
    print("="*50)

