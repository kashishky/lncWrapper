# lncWrapper
A concise, explainable deep reinforcement learning framework for predicting regulatory functions of long non-coding RNAs by integrating sequence, secondary structure, and GWAS data. 


Overview
lncWrapper is a modular deep reinforcement learning (DRL) framework designed to predict the regulatory functions of long non-coding RNAs. It integrates sequence data, secondary structure, and genome-wide association studies (GWAS) data to generate accurate and interpretable predictions.


Project Structure:

agent/         # DRL agent implementations

data/          # Datasets and data loaders

deployment/    # API server for inference (FastAPI)

environment/   # Environment logic for the agent

evaluation/    # Tools for performance evaluation

frontend/      # Streamlit dashboard for user interaction

preprocessing/ # Data preprocessing scripts

training/      # Model training logic

demo.py        # Original local-only demo script

requirements.txt


Prerequisites:
Python 3.7 or higher, PyTorch (compatible version), Streamlit, Other libraries in requirements.txt


Installation:
1. Clone the repository:
   git clone https://github.com/kashishky/lncWrapper.git
   cd lncWrapper
2. Create a virtual environment (optional but recommended):
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install the required packages:
   pip install -r requirements.txt


Running the API Server Locally: 
1. All scripts are designed to be run locally. The API server is not externally hosted at this stage.
2. From the root folder, launch the FastAPI backend: uvicorn deployment.lncrna_api_server:app --reload --host 0.0.0.0 --port 8000

Running the Streamlit Frontend:
1. The Streamlit dashboard is standalone and not fully integrated with the API outputs. This design choice was made intentionally to allow independent development and testing.
2. From the root folder: streamlit run frontend/frontend.py




