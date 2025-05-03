```markdown
# lncWrapper: Explainable Deep Reinforcement Learning for lncRNA–GWAS Integration

A complete, modular pipeline to predict and interpret regulatory hotspots in long non-coding RNAs (lncRNAs) by integrating sequence, structure, GWAS, conservation, motif and regulatory-feature data, and optimizing model attention via deep reinforcement learning.

---

## Repository Structure

```

lncWrapper/
├── data/
│   ├── gwas/
│   │   └── gwas\_catalog\_v1.0-associations\_…tsv
│   ├── preprocessed/              ← training dataset
│   │   ├── summary.csv
│   │   └── base\_arrays/
│   │       └── \<transcript\_id>.npz
│   └── preprocessed\_test/         ← held-out evaluation dataset
│       ├── summary.csv
│       └── base\_arrays/
├── preprocessing/
│   └── generate\_preprocessed.py   ← builds summary.csv and .npz from raw data
├── environment/
│   └── lncrna\_explain\_env.py      ← custom Gym environment
├── agent/
│   └── drl\_agent\_explain.py       ← DQN implementation
├── training/
│   └── train\_agent\_lncrna\_explain.py
├── evaluation/
│   └── evaluate\_agent\_lncrna\_explain.py
├── deployment/
│   └── lncrna\_api\_server.py       ← FastAPI inference service
├── ui/
│   └── streamlit\_app.py           ← Streamlit dashboard (Live API)
├── frontend-demo/
│   └── demo\_script.py             ← Static placeholder demo (no live model)
└── checkpoints/
└── final\_model.pth

```

---

## System Requirements

- **Operating System:** Linux (Ubuntu ≥18.04), macOS, or WSL2 on Windows  
- **Python:** ≥3.10  
- **Memory:** ≥32 GB RAM recommended for large per-base arrays  
- **GPU:** NVIDIA CUDA-capable GPU recommended for accelerating DQN training  

### Python Dependencies

Install via `pip` or `conda`:

```

numpy>=1.21
pandas>=1.3
scipy>=1.7
scikit-learn>=1.0
torch>=1.12
gym>=0.21
requests>=2.26
biopython>=1.79
viennarna                   # ViennaRNA Python bindings
matplotlib>=3.5
plotly>=5.6
fastapi>=0.75
uvicorn\[standard]>=0.17
langchain\_community
duckduckgo-search
streamlit>=1.15
typing\_extensions>=4.0
python-multipart>=0.0.5

````

---

## Installation and Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your_org/lncWrapper.git
   cd lncWrapper
````

2. **Create and activate a Conda environment**

   ```bash
   conda create -n lncwrapper python=3.10
   conda activate lncwrapper
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   # or install individual packages as listed above
   ```

4. **Install ViennaRNA**

   ```bash
   conda install -c bioconda viennarna
   ```

---

## Pipeline Workflow

### 1. Data Preprocessing

Parse raw FASTA and GWAS catalog, retrieve REST data, and build per-base arrays:

```bash
python preprocessing/generate_preprocessed.py \
  --fasta data/sequences/lncRNA_LncBookv2.1_GRCh38.fa.gz \
  --gwas data/gwas/gwas_catalog_v1.0-associations_…tsv \
  --out_dir data/preprocessed
```

This produces:

* `data/preprocessed/summary.csv`
* `data/preprocessed/base_arrays/*.npz`

### 2. Agent Training

Train the DQN agent to tune attention scaling and bias:

```bash
python training/train_agent_lncrna_explain.py \
  --summary_csv data/preprocessed/summary.csv \
  --base_array_dir data/preprocessed/base_arrays \
  --num_episodes 200
```

Outputs:

* `training_metrics.png`
* `episode_lengths_hist.png`
* `policy_net_checkpoint.pth`

### 3. Model Evaluation

Evaluate fixed-policy performance on held-out transcripts:

```bash
python evaluation/evaluate_agent_lncrna_explain.py \
  --checkpoint checkpoints/final_model.pth \
  --summary_csv data/preprocessed_test/summary.csv \
  --base_array_dir data/preprocessed_test/base_arrays
```

Generates:

* `evaluation_metrics_test.json`
* `evaluation_improvements_test.png`

### 4. Deployment (REST API)

Launch the FastAPI inference service:

```bash
uvicorn deployment.lncrna_api_server:app \
  --reload --host 0.0.0.0 --port 8000
```

Endpoints:

* `POST /infer` accepts `transcript_id` or FASTA upload and returns full JSON output (per-base arrays, metrics, hotspots, GO/KEGG, LLM narrative).

### 5. Interactive Dashboard

Run the Streamlit dashboard (connects to the live API):

```bash
streamlit run ui/streamlit_app.py
```

* Sidebar: select known transcript or upload FASTA
* Main panel: multi-track visualization, DRL metrics, enrichment tables, LLM insights

---

## Demo Placeholder

The `frontend-demo/` directory contains a static demonstration script (`frontend.py`) with synthetic data for illustrative purposes. This placeholder **does not** invoke the live model or API. For a fully interactive experience, use the dashboard in `ui/streamlit_app.py`.

---

## Notes

* Ensure both `data/preprocessed/` and `data/preprocessed_test/` directories are populated before training or evaluation (currently contain synthetic data due to size limtations).
* Hyperparameters (episodes, ε-decay schedule, reward weights) are configurable in the training script.
* Precompute and cache REST queries (GO counts, eQTLs) to reduce runtime.
* For production deployment, consider asynchronous workers or GPU acceleration for inference.

---

```
```
