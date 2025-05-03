# lncWrapper: Explainable Deep Reinforcement Learning for lncRNA–GWAS Integration

A modular pipeline to predict and interpret regulatory hotspots in long non–coding RNAs by integrating sequence, structure, GWAS, conservation, motif and regulatory-feature data, and optimizing model attention via Deep Q-Network (DQN) reinforcement learning.

## Repository Structure

```
lncWrapper/
├── data/
│   ├── gwas/
│   │   └── gwas_catalog_v1.0-associations_…tsv
│   ├── preprocessed/              ← training dataset
│   │   ├── summary.csv
│   │   └── base_arrays/
│   │       └── <transcript_id>.npz
│   └── preprocessed_test/         ← held-out evaluation dataset
│       ├── summary.csv
│       └── base_arrays/
├── preprocessing/
│   └── generate_preprocessed.py   ← builds summary.csv and .npz from raw data
├── environment/
│   └── lncrna_explain_env.py      ← custom Gym environment
├── agent/
│   └── drl_agent_explain.py       ← DQN implementation
├── training/
│   └── train_agent_lncrna_explain.py
├── evaluation/
│   └── evaluate_agent_lncrna_explain.py
├── deployment/
│   └── lncrna_api_server.py       ← FastAPI inference service
├── ui/
│   └── streamlit_app.py           ← Streamlit dashboard (Live API)
├── frontend-demo/
│   └── demo_script.py             ← Static placeholder demo (no live model)
└── checkpoints/
    └── final_model.pth
```

## System Requirements

- **Operating System**: Linux (Ubuntu ≥18.04), macOS, or WSL2 on Windows
    
- **Python**: ≥3.10
    
- **Memory**: ≥32 GB RAM recommended
    
- **GPU**: NVIDIA CUDA-capable GPU recommended for DQN training
    

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
uvicorn[standard]>=0.17
langchain_community
duckduckgo-search
streamlit>=1.15
typing_extensions>=4.0
python-multipart>=0.0.5
```

## Installation and Setup

1. **Clone the repository**
    
    ```bash
    git clone https://github.com/your_org/lncWrapper.git
    cd lncWrapper
    ```
    
2. **Create and activate a Conda environment**
    
    ```bash
    conda create -n lncwrapper python=3.10
    conda activate lncwrapper
    ```
    
3. **Install dependencies**
    
    ```bash
    pip install -r requirements.txt
    ```
    
4. **Install ViennaRNA**
    
    ```bash
    conda install -c bioconda viennarna
    ```
    

## Pipeline Workflow

### 1. Data Preprocessing

Parse raw FASTA and GWAS catalog, fetch conservation/motif/regulatory data via REST, and build per-base arrays:. Or please use synthetic datasets provided in current preprocessed files.

```bash
python preprocessing/generate_preprocessed.py \
  --fasta data/sequences/lncRNA_LncBookv2.1_GRCh38.fa.gz \
  --gwas data/gwas/gwas_catalog_v1.0-associations_…tsv \
  --out_dir data/preprocessed
```

**Outputs**

- `data/preprocessed/summary.csv`
    
- `data/preprocessed/base_arrays/*.npz`
    

---

### 2. Agent Training

Train the DQN agent to tune attention scaling and bias:

```bash
python training/train_agent_lncrna_explain.py \
  --summary_csv data/preprocessed/summary.csv \
  --base_array_dir data/preprocessed/base_arrays \
  --num_episodes 200
```

**Outputs**

- `training_metrics.png`
    
- `episode_lengths_hist.png`
    
- `policy_net_checkpoint.pth`
    

### 3. Model Evaluation

Evaluate fixed-policy performance on held-out transcripts:

```bash
python evaluation/evaluate_agent_lncrna_explain.py \
  --checkpoint checkpoints/final_model.pth \
  --summary_csv data/preprocessed_test/summary.csv \
  --base_array_dir data/preprocessed_test/base_arrays
```

**Outputs**

- `evaluation_metrics_test.json`
    
- `evaluation_improvements_test.png`
    

### 4. Deployment (REST API)

Launch the FastAPI inference service:

```bash
uvicorn deployment/lncrna_api_server:app \
  --reload --host 0.0.0.0 --port 8000
```

**Endpoint**

- `POST /infer` accepts `transcript_id` or FASTA upload; returns comprehensive JSON (per-base arrays, metrics, hotspots, GO/KEGG, LLM narrative, etc.)
    

### 5. Interactive Dashboard

Run the Streamlit dashboard (connects to the live API):

```bash
streamlit run frontend/frontend.py
```

- Sidebar: select a transcript ID or upload a FASTA
    
- Main panel: multi-track visualization, DRL metrics, GO/KEGG enrichment, and LLM-generated insights
    

## Demo Placeholder

The **`frontend-demo/`** directory contains a static demonstration script (`demo_script.py`) with synthetic data for illustrative purposes. This placeholder **does not** invoke the live model or API. For a fully interactive experience, use what is in frontend folder.

## Additional Notes

- Populate both `data/preprocessed/` and `data/preprocessed_test/` before training or evaluation.
    
- Training hyperparameters (episodes, ε-decay, reward weights) are configurable in the training script.
    
- To optimize performance in production, consider pre-caching REST queries and enabling GPU acceleration.