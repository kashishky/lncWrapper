# lncWrapper
A concise, explainable deep reinforcement learning framework for predicting regulatory functions of long non-coding RNAs by integrating sequence, secondary structure, and GWAS data. 

lncWrapper is a modular deep reinforcement learning (DRL) framework designed to predict the regulatory functions of long non-coding RNAs (lncRNAs). By integrating sequence data, secondary structure information, and genome-wide association studies (GWAS) data, the framework aims to provide accurate and explainable predictions.

Project Structure:

agent/: Contains the DRL agent implementations.

data/: Includes datasets and data loaders.

deployment/: Scripts and configurations for deploying the model.

environment/: Defines the environment in which the agent operates.

evaluation/: Tools for evaluating model performance.

frontend/: Streamlit application for visualization and interaction.

preprocessing/: Data preprocessing scripts.

training/: Training routines and configurations.

demo.py: Demonstration script to showcase model capabilities.

requirements.txt: Lists all Python dependencies.

Prerequisites:
Python 3.7 or higher
PyTorch (compatible version)
Streamlit
Other dependencies listed in requirements.txt


Installation:
1. Clone the repository:
   git clone https://github.com/kashishky/lncWrapper.git
   cd lncWrapper
2. Create a virtual environment (optional but recommended):
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install the required packages:
   pip install -r requirements.txt


Running the API Server Locally: All scripts are designed to be run locally. The API server is not externally hosted at this stage.

Streamlit Dashboard: The Streamlit dashboard is standalone and not fully integrated with the API outputs. This design choice was made intentionally to allow independent development and testing.


