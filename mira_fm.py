import torch
import numpy as np
# from mira.mira.models.modeling_mira import MIRAForPrediction
# from mira.mira.models.utils_time_normalization import normalize_time_for_ctrope

class MIRASystem:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = device
        print("⚠️ USING MOCKED MIRA SYSTEM (No checkpoint loaded)")
        self.model = None

    def get_mira_outputs(self, vitals_tensor, times_tensor, steps=5):
        """
        The "All-in-One" method for the Pipeline.
        Returns:
            - latent_vector: [1, 384] (For the MoE Router)
            - forecasts_real: [1, steps, 5] (Real clinical units for the Report)
        """
        B, L, C = vitals_tensor.shape
        
        # 1. Capture Stats for Inverse-Normalization
        # We calculate mean/std per channel (vital) to ensure accuracy
        means = vitals_tensor.mean(dim=1, keepdim=True) # [1, 1, 5]
        stds = vitals_tensor.std(dim=1, keepdim=True) + 1e-6

        # 2. Channel-Independent Normalization & Prep
        v_norm = (vitals_tensor - means) / stds
        v_reshaped = v_norm.permute(0, 2, 1).reshape(B * C, L, 1) # [5, 48, 1]
        t_expanded = times_tensor.expand(B * C, L)

        with torch.no_grad():
            latent_vector = torch.randn(B, 384, device=self.device)
            preds_combined = torch.randn(C, steps, 1, device=self.device)
            forecasts_real = (preds_combined * stds) + means

        return latent_vector, forecasts_real

# --- FINAL INTEGRATION TEST ---
# if __name__ == "__main__":
#     mira = MIRASystem("mira/checkpoints")
#     labels = ["MAP", "HR", "RR", "SpO2", "Temp"]

#     # Mock Input: 1 Patient, 48 Hours, 5 Vitals
#     mock_in = torch.randn(1, 48, 5).to("cuda")
#     mock_t = torch.arange(48).float().unsqueeze(0).to("cuda")

#     latent, forecast = mira.get_mira_outputs(mock_in, mock_t, steps=5)

#     print(f"✅ MoE Router Input (Latent): {latent.shape}")
#     print(f"✅ Clinical Forecast (T+1 to T+5):\n{forecast[0]}") 
    # forecast[0] is [5, 5] -> 5 steps for 5 vitals

# import torch
# import json
# import numpy as np
# from mira.mira.models.modeling_mira import MIRAForPrediction
# from mira.mira.models.utils_time_normalization import normalize_time_for_ctrope

# class MIRASystem:
#     def __init__(self, checkpoint_path, device="cuda"):
#         self.device = device
#         print(f"🏥 Loading MIRA Checkpoints from {checkpoint_path}...")
#         self.model = MIRAForPrediction.from_pretrained(checkpoint_path).to(device)
#         self.model.eval()
        
#         self.item_map = {
#             "220052": "MAP", "220045": "Heart Rate", "220210": "Resp Rate",
#             "220277": "SpO2", "223762": "Temp (C)", "50813": "Lactate",
#             "50820": "pH", "50912": "Creatinine", "51301": "WBC"
#         }

#     def get_latent_for_router(self, seq, times):
#         """Used by the MoE Router (Returns the 384-dim vector)"""
#         mean, std = seq.mean(), seq.std() + 1e-6
#         seq_norm = (seq - mean) / std
        
#         with torch.no_grad():
#             outputs = self.model(
#                 input_ids=seq_norm.unsqueeze(-1), 
#                 time_values=times,
#                 output_hidden_states=True
#             )
#             # Patient latent: mean pool across the sequence last hidden state
#             return outputs.hidden_states[-1][:, -1, :]

#     def run_clinical_demo(self, data_path, steps=3):
#         """Processes real JSONL data and de-normalizes for the report."""
#         seen_items = set()
#         print(f"\n{'='*60}\nRAW CLINICAL FORECASTS (Inverse Normalized)\n{'='*60}")

#         with open(data_path, 'r') as f:
#             for line in f:
#                 data = json.loads(line)
#                 itemid = str(data.get('itemid'))
                
#                 if itemid in seen_items or itemid not in self.item_map:
#                     continue
#                 seen_items.add(itemid)
                
#                 # 1. Prepare Data
#                 seq = torch.tensor([data['sequence']], dtype=torch.float32).to(self.device)
#                 times = torch.tensor([data['time']], dtype=torch.float32).to(self.device)
                
#                 # 2. Instance Normalization (Save stats for later!)
#                 mean = seq.mean().item()
#                 std = (seq.std() + 1e-6).item()
#                 seq_norm = (seq - mean) / std

#                 # 3. MIRA Time Normalization (CT-RoPE)
#                 full_scaled_times, _, _ = normalize_time_for_ctrope(
#                     time_values=times,
#                     attention_mask=torch.ones_like(times),
#                     seq_length=seq.shape[1],
#                     alpha=1.0
#                 )

#                 # 4. Autoregressive Loop
#                 cur_vals = seq_norm.clone()
#                 cur_times = full_scaled_times.clone()
#                 preds_norm = []

#                 with torch.no_grad():
#                     for _ in range(steps):
#                         out = self.model(input_ids=cur_vals.unsqueeze(-1), time_values=cur_times)
#                         next_val_norm = out.logits[:, -1, :]
#                         preds_norm.append(next_val_norm.item())
                        
#                         # Update rolling window
#                         cur_vals = torch.cat([cur_vals, next_val_norm], dim=1)
#                         next_t = cur_times[:, -1:] + 1.0 
#                         cur_times = torch.cat([cur_times, next_t], dim=1)

#                 # 5. DE-NORMALIZE (Back to real clinical units)
#                 preds_real = [round((p * std) + mean, 2) for p in preds_norm]
#                 history = [round(x, 2) for x in data['sequence'][-3:]]

#                 # 6. OUTPUT FOR REPORT
#                 label = self.item_map[itemid]
#                 print(f"[{label:^12}] ID: {itemid}")
#                 print(f"  ∟ History (T-2, T-1, T=0): {history}")
#                 print(f"  ∟ MIRA Prediction (+1h, +2h, +3h): {preds_real}")
#                 print("-" * 60)

# # --- MAIN EXECUTION ---
# if __name__ == "__main__":
#     # Point this to your actual training data subset
#     mira_engine = MIRASystem("mira/checkpoints")
#     mira_engine.run_clinical_demo("mimic_datasets/train_mira.jsonl", steps=3)