# GraphRAG + Time-Series Neurosymbolic Pipeline

This repository implements a neurosymbolic clinical inference pipeline that combines:

1. Neural temporal modeling (MIRA) for patient state encoding and short-horizon forecasting.
2. Symbolic retrieval (PrimeKG Graph-RAG) for verifiable medical context.
3. LLM semantic embedding (Gemini Embeddings) to condition expert routing.
4. Mixture-of-Experts (MoE) fusion for risk scoring and downstream forecast output.

The entrypoint is `main.py`, which wires all major components together through `NeurosymbolicPipeline`.

## Project Goal

Given a patient time series and metadata, the system:

1. Encodes raw vitals into a dense clinical latent (`MIRASystem`).
2. Retrieves relevant biomedical context from PrimeKG (`PrimeKGGraphRAG`).
3. Converts retrieved context into a fixed text embedding (`MedicalEmbeddingClient`).
4. Uses embedding-conditioned expert routing to produce risk and forecast (`FullSystemMoE`).

This design aims to preserve both predictive performance (neural) and interpretable knowledge grounding (symbolic retrieval text).

## Pipeline Outline (as implemented in `main.py`)

`NeurosymbolicPipeline.__init__` initializes:

- `self.mira = MIRASystem(mira_ckpt)`
- `self.graph_rag = PrimeKGGraphRAG(kg_path)`
- `self.gemini = MedicalEmbeddingClient(gemini_key)`
- `self.moe = FullSystemMoE(text_embed_dim=3072).to("cuda").eval()`

`NeurosymbolicPipeline.run_inference` executes:

1. Parse one JSON line.
2. Build tensors from `sequence` and `time` and send to CUDA.
3. Call MIRA to get latent + neural forecast.
4. Call Graph-RAG retrieval for symbolic context.
5. Embed symbolic context with Gemini.
6. Route and fuse in MoE for risk + forecast + expert weights.
7. Return packaged outputs for downstream reporting.

## Module Reference and Significance

This section documents each module used directly or transitively from `main.py` and why it matters.

### 1) `main.py` (orchestration layer)

Role:
- Defines `NeurosymbolicPipeline`, the system integration surface for inference.

Why it is significant:
- This is the canonical control flow for production-style inference.
- Any future deployment adapter (API server, batch job, notebook runner) should call this class or preserve its data contract.

Exports used:
- `NeurosymbolicPipeline`

Key methods:
- `__init__(mira_ckpt, kg_path, gemini_key)`
- `run_inference(patient_jsonl_line)`

### 2) `mira_fm.py` (neural temporal foundation wrapper)

Role:
- Wraps pretrained MIRA checkpoint loading and exposes a pipeline-friendly method: `get_mira_outputs`.

Why it is significant:
- Converts multivariate vitals into a shared latent (`[B, 384]`) consumed by MoE.
- Produces autoregressive multi-step forecasts and inverse-normalizes to clinical units.

Core class:
- `MIRASystem`

Core method:
- `get_mira_outputs(vitals_tensor, times_tensor, steps=5)` returns:
  - `latent_vector`: `[B, 384]`
  - `forecasts_real`: `[B, steps, C]`

Dependencies inside module:
- `mira/mira/models/modeling_mira.py` (`MIRAForPrediction`)
- `mira/mira/models/utils_time_normalization.py` (imported, not currently used in active method)

### 3) `graph_rag.py` (symbolic retrieval engine)

Role:
- Implements PrimeKG-backed retrieval with semantic entry-point matching and local graph expansion.

Why it is significant:
- Provides verifiable biomedical context that conditions MoE expert selection.
- Bridges sparse symbolic knowledge and dense neural prediction.

Core class:
- `PrimeKGGraphRAG`

Key methods:
- `semantic_lookup(query, top_k=2)`
- `run_retrieval(vital_name, condition="Sepsis")`

Important implementation note:
- `run_retrieval` is defined twice in this file. In Python, the second definition overrides the first. The active behavior currently returns a newline-joined string of filtered triplets.

Required artifacts:
- `kg.csv` with at least `x_name`, `y_name`, `display_relation` columns.
- `primekg.index` (FAISS index)
- `primekg_nodes.pkl` (node list aligned to FAISS vectors)

### 4) `build_embeddings.py` (offline index builder for Graph-RAG)

Role:
- One-time or periodic preprocessing script that builds FAISS search artifacts from `kg.csv`.

Why it is significant:
- `graph_rag.py` depends on precomputed index artifacts for low-latency retrieval.
- Without this script output, retrieval initialization fails.

Primary function:
- `build_static_index(kg_path, out_prefix="primekg")`

Outputs:
- `primekg.index`
- `primekg_nodes.pkl`

### 5) `utils/gemini_client.py` (LLM embedding adapter)

Role:
- Wraps Google GenAI embedding API and returns a PyTorch tensor.

Why it is significant:
- Produces router conditioning vector for MoE gating (`text_embed_dim=3072`).
- Isolates vendor-specific API logic from model logic.

Core class:
- `MedicalEmbeddingClient`

Core method:
- `get_embedding(text)` returns `torch.Tensor` with shape `[1, 3072]` for `gemini-embedding-2`.

### 6) `model/moe_system.py` (fusion and decision layer)

Role:
- Defines embedding-conditioned expert routing and fused clinical prediction heads.

Why it is significant:
- Central decision module that maps MIRA latent + Graph-RAG embedding to final risk and forecast.
- Encodes the neurosymbolic fusion logic of the project.

Core classes:
- `LLMGatingNetwork` (softmax over experts)
- `FullSystemMoE` (experts + fusion + output heads)

Forward contract:
- Input:
  - `mira_latent`: `[B, 384]`
  - `graphrag_embeddings`: `[B, 3072]`
- Output:
  - `risk`: `[B, 1]` after sigmoid
  - `forecast`: `[B, 24, time_features]`
  - `weights`: `[B, 3]` routing probabilities

### 7) `model/experts.py` (expert subnetworks)

Role:
- Defines `ResidualClinicalExpert`, used three times in `FullSystemMoE`.

Why it is significant:
- Implements specialized nonlinear transformations while preserving latent identity via residual connections.
- Serves as the specialization basis for hemodynamic, biochemical, and generalist branches.

Core class:
- `ResidualClinicalExpert`

### 8) `model/encoder.py` (legacy/placeholder timeseries encoder)

Role:
- Contains `MIRATimeSeriesEncoder` (BiGRU + attention) placeholder implementation.

Why it is significant:
- Imported in `model/moe_system.py` but not used in the current active `forward` path.
- Represents an alternative architecture where MoE ingests raw time-series rather than MIRA latent.

Contributor note:
- Safe to treat as legacy unless you intentionally switch MoE back to raw-series encoding.

### 9) External package integration from `main.py`

`dotenv` (`python-dotenv`):
- `load_dotenv()` loads local environment variables before API-key usage.

`torch`:
- Tensor creation, device placement, and no-grad inference execution.

`json`:
- Parses line-level JSON input payloads.

`os`:
- Reads `GEMINI_API` environment variable.

## Data Contracts

### Input to `run_inference`

Expected JSON string should include at least:

- `itemid`: str or int (used by Graph-RAG retrieval)
- `sequence`: list of numeric values or list of lists (time-series)
- `time`: list of timestamps aligned with sequence length

Important:
- The demo `sample_case` in `main.py` currently omits `itemid`. Running exactly that sample will cause a key error at `data['itemid']`.

### Output from `run_inference`

Returns dict with keys:

- `risk`: scalar float
- `forecast`: PyTorch tensor from MoE forecast head
- `time_series_forecast`: NumPy array from MIRA forecast
- `expert_weights`: list of routing probabilities
- `clinical_justification`: retrieved symbolic context text

## Environment and Setup

### Python

- Project metadata (`pyproject.toml`) declares Python `>=3.12`.

### Install dependencies

You can use either:

1. `pip install -r requirements.txt`
2. `pip install -e .` (using `pyproject.toml`)

Recommendation for contributors:
- Keep `requirements.txt` and `pyproject.toml` aligned when updating dependencies.

### Required files before inference

1. MIRA checkpoint directory at `mira/checkpoints`.
2. PrimeKG CSV at `kg.csv`.
3. Retrieval index artifacts in repo root:
	- `primekg.index`
	- `primekg_nodes.pkl`
4. `.env` containing:
	- `GEMINI_API=<your_api_key>`

### Build Graph-RAG index (if missing)

Run:

`python build_embeddings.py`

This reads `kg.csv` and generates `primekg.index` and `primekg_nodes.pkl`.

## How to Run

Basic run:

`python main.py`

What happens at runtime:

1. Loads MIRA checkpoint on CUDA.
2. Loads PrimeKG graph + FAISS index.
3. Calls Gemini embedding API.
4. Runs MoE fusion and prints summary outputs.

## Extension Guide for Contributors

### Add a new expert branch

Where:
- `model/moe_system.py`

What to update:
- Increase `num_experts` in `LLMGatingNetwork`.
- Add new expert module instance in `FullSystemMoE.__init__`.
- Include expert output in `experts_out` stack.

Why:
- Lets you route to finer-grained physiologic specialization.

### Swap embedding model/provider

Where:
- `utils/gemini_client.py`

What to keep stable:
- Return tensor shape expected by MoE (`[B, text_embed_dim]`).

Why:
- Decouples vendor choice from model architecture.

### Change retrieval strategy

Where:
- `graph_rag.py`

Ideas:
- Multi-hop expansion.
- Relation-type weighting.
- Structured output dict with provenance metadata.

Why:
- Improves context relevance and interpretability.

### Enable CPU fallback

Where:
- `main.py`, `mira_fm.py`

What to update:
- Replace hardcoded `.to("cuda")` calls with configurable device selection.

Why:
- Supports development on non-GPU environments.

## Known Implementation Notes

1. Duplicate method in `graph_rag.py`:
	- Second `run_retrieval` overrides first.
2. Legacy import in `model/moe_system.py`:
	- `MIRATimeSeriesEncoder` imported but unused in active forward path.
3. Demo payload mismatch in `main.py`:
	- `sample_case` does not include `itemid` though runtime requires it.


## Suggested Repository Conventions

1. Keep all runtime tensor shapes documented near module interfaces.
2. Treat `main.py` as orchestration only; put modeling logic in module files.
3. Add small integration tests for tensor shape compatibility across:
	- `mira_fm.py` output latent dimension
	- `utils/gemini_client.py` embedding dimension
	- `model/moe_system.py` expected input dimensions

## Minimal Contributor Checklist

1. Confirm `.env` and checkpoint paths are valid.
2. Confirm `primekg.index` and `primekg_nodes.pkl` exist and match current `kg.csv`.
3. Validate one full forward pass through `NeurosymbolicPipeline.run_inference`.
4. If changing dimensions, update all affected modules in one commit.
5. Document any interface changes in this README.
