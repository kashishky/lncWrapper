import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Configure page
st.set_page_config("lncRNA Explorer (API‐backed)", layout="wide")

# Sidebar: input form
st.sidebar.header("Load lncRNA via API")
with st.sidebar.form("infer_form"):
    transcript_id = st.text_input("Transcript ID (e.g. ENSG00000260032)")
    fasta_file = st.file_uploader("Or upload FASTA (>chr:start-end)", type=["fa","fasta"])
    submit = st.form_submit_button("Run Inference")

if not submit:
    st.info("Enter a transcript ID or upload a FASTA, then click ‘Run Inference’")
    st.stop()

# Send request to API
api_url = "http://localhost:8000/infer"
files = {}
data = {}
if transcript_id:
    data["transcript_id"] = transcript_id
elif fasta_file:
    files["fasta"] = (fasta_file.name, fasta_file.getvalue())
else:
    st.error("Provide either Transcript ID or FASTA")
    st.stop()

with st.spinner("Contacting inference API…"):
    resp = requests.post(api_url, data=data, files=files)
if resp.status_code != 200:
    st.error(f"API error {resp.status_code}: {resp.text}")
    st.stop()

result = resp.json()

# Header info
st.title("lncRNA Explorer (API Results)")
tid = transcript_id or "CUSTOM"
length = len(result["attention_weights"])
st.markdown(
    f"**Transcript:** {tid}    |    **Length:** {length} nt    |    "
    f"**Loss:** {result['final_loss']:.3f}    |    "
    f"**Interp:** {result['interpretability']:.3f}    |    "
    f"**Reward:** {result['composite_reward']:.3f}"
)

# Plot multi‐track viewer
st.subheader("Sequence & Annotation Tracks")
pos = np.arange(length)
att = np.array(result["attention_weights"])
# Build SNP mask and values
snp_mask = np.zeros(length, bool)
snp_vals = np.zeros(length)
for hit in result["snp_overlap"]:
    p = hit["position"]
    snp_mask[p] = True
    snp_vals[p] = hit.get("log10p", 0.0)

# Create subplots: Sequence, Attention, SNPs, (stub) Conservation & Others
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    row_heights=[0.1,0.3,0.3,0.3],
    subplot_titles=["Sequence", "Attention", "GWAS SNPs & Scores", "Other Features"]
)

# 1) Nucleotide sequence
seq = result.get("hotspots", [])
# We don't have raw seq from API; show positions instead
fig.add_trace(
    go.Scatter(x=pos, y=[0]*length, mode="markers+text",
               text=["•"]*length, textfont=dict(size=8),
               hovertext=[f"Pos {i}" for i in pos],
               showlegend=False),
    row=1, col=1
)

# 2) Attention line
fig.add_trace(
    go.Scatter(x=pos, y=att, mode="lines", line_color="crimson", name="Attention"),
    row=2, col=1
)

# 3) SNP bar + ticks
fig.add_trace(
    go.Bar(x=pos, y=snp_vals, marker_color="black", name="-log10(p)"),
    row=3, col=1
)
# SNP ticks
snp_pos = pos[snp_mask]
fig.add_trace(
    go.Scatter(x=snp_pos, y=snp_vals[snp_mask]*1.1,
               mode="markers", marker_symbol="line-ns-open",
               marker_color="black", marker_size=10,
               name="SNPs",
               hovertext=[f"{hit['rsID']}: {hit['log10p']:.2f}" 
                          for hit in result["snp_overlap"]],
               hoverinfo="text"),
    row=3, col=1
)

# 4) Other features: conservation, tfbs, reg, atac if present
# Many will be stubbed zeros; but we check keys
# For simplicity, just show counts per-base if non-zero
# In real API extend to include arrays
for key, color in [("cons", "blue"), ("tfbs", "green"), ("reg", "purple"), ("atac", "orange")]:
    arr = result.get(key)
    if arr:
        arr = np.array(arr)
        fig.add_trace(
            go.Scatter(x=pos, y=arr, mode="lines", line_color=color, name=key),
            row=4, col=1
        )

fig.update_yaxes(visible=False)
fig.update_xaxes(title_text="Position", row=4)
fig.update_layout(height=700, showlegend=True, margin=dict(l=50,r=50,t=50,b=50))
st.plotly_chart(fig, use_container_width=True)

# Parameter adjustments
adj = result["parameter_adjustments"]
st.subheader("DRL Parameter Adjustments")
st.write(f"Scaling adjustments per step: {[a[0] for a in adj]}")
st.write(f"Bias adjustments per step:    {[a[1] for a in adj]}")

# GO terms & eQTLs
st.subheader("GO Annotations")
st.write(result["go_terms"])
st.subheader("eQTL Associations")
st.write(result["eqtls"])

# Narrative & literature
st.subheader("LLM Narrative")
st.write(result["narrative"])
st.subheader("Literature Snippets")
for snip in result["literature_snippets"]:
    st.markdown(f"- {snip}")
