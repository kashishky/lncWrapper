"""
This is a pre-loaded visual set up of the demo calling form the API server using examples, not a live represnetation of calling the model

"""


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Bio import SeqIO
import io, re



# Page config
st.set_page_config(page_title="lncRNA Explorer Demo", layout="wide")

# Sidebar controls
st.sidebar.header("Controls: Load & Display")
example = st.sidebar.selectbox(
    "Example transcript:", ["LINC00115", "HSALNT0815277", "CUSTOM_DEMO"]
)
fasta_file = st.sidebar.file_uploader("Or upload FASTA (header >chr:start-end)", type=["fa","fasta"])

# Track toggles
show_att = st.sidebar.checkbox("Attention Hotspots", True)
show_gwas = st.sidebar.checkbox("GWAS SNPs & Scores", True)
show_cons = st.sidebar.checkbox("Conservation", True)
show_tfbs = st.sidebar.checkbox("TFBS Motifs", True)
show_reg  = st.sidebar.checkbox("Regulatory Features", True)
show_atac = st.sidebar.checkbox("ATAC Peaks", True)

# Load or generate data
@st.cache_data
def load_data(example, fasta_file):
    if fasta_file:
        content = fasta_file.read().decode()
        rec = next(SeqIO.parse(io.StringIO(content), 'fasta'), None)
        header = rec.description if rec else ''
        m = re.match(r"(\w+):(\d+)-(\d+)", header)
        chr_  = m.group(1) if m else 'chr1'
        start = int(m.group(2)) if m else 1000000
        seq   = str(rec.seq) if rec else ''
    else:
        coords = {'LINC00115':('chr1',150000000,800), 'HSALNT0815277':('chr4',136269570,650)}
        chr_, start, length = coords.get(example,('chrX',1000000,700))
        seq = ''.join(np.random.choice(list('ACGU'), length))
    length = len(seq)
    end = start + length - 1
    pos = np.arange(length)
    att = np.exp(-((pos-length*0.3)/(length*0.1))**2)
    att += np.exp(-((pos-length*0.7)/(length*0.05))**2)
    att /= att.max()
    gmask = np.random.binomial(1,0.02,length); gmask[0]=1
    gvals = gmask * np.random.uniform(3,7,length)
    cons  = np.clip(np.random.normal(0.6,0.2,length),0,1)
    tfbs  = np.random.binomial(1,0.007,length)
    reg   = np.random.binomial(1,0.01,length)
    atac  = np.random.binomial(1,0.015,length)
    df = pd.DataFrame({'pos':pos,'nuc':list(seq),'attention':att,
                       'gmask':gmask,'gvals':gvals,'cons':cons,
                       'tfbs':tfbs,'reg':reg,'atac':atac})
    go_df = pd.DataFrame([
        {'GO ID':'GO:0006355','Term':'Regulation of transcription','FDR':0.002},
        {'GO ID':'GO:0008380','Term':'RNA splicing','FDR':0.005},
        {'GO ID':'GO:0003723','Term':'RNA binding','FDR':0.010}
    ])
    kegg_df = pd.DataFrame([
        {'Pathway':'Cell cycle','FDR':0.004},
        {'Pathway':'Spliceosome','FDR':0.019}
    ])
    metrics = pd.DataFrame({
        'Metric':['Prediction Loss','Interpretability','Composite Reward','Bias Adj','Scaling Adj'],
        'Value':[0.18,0.88,1.76,1,0]
    })
    return chr_,start,end,seq,df,go_df,kegg_df,metrics

chr_,start,end,seq,df,go_df,kegg_df,metrics = load_data(example,fasta_file)
length = len(seq)

# Header info
st.title('lncRNA Explorer Demo')
st.markdown(f"**Transcript:** {example} | **Coordinates:** {chr_}:{start}-{end} | **Length:** {length} nt")

# Region selection
window = st.slider('View region (nt):', 0, length-1, (0, min(200,length-1)))
region = df.iloc[window[0]:window[1]+1]

# Determine tracks order and titles
titles = ['Sequence']
if show_att: titles.append('Attention')
if show_gwas: titles.append('GWAS')
if show_cons: titles.append('Conservation')
if show_tfbs: titles.append('TFBS Motifs')
if show_reg: titles.append('Regulatory Features')
if show_atac: titles.append('ATAC Peaks')

rows = len(titles)
heights = [0.1] + [0.9/(rows-1)]*(rows-1)

# Create subplots
grid = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                     row_heights=heights, vertical_spacing=0.02,
                     subplot_titles=titles)
row = 1
# Sequence row
grid.add_trace(go.Scatter(x=region.pos, y=[0]*len(region), mode='text',
                          text=region.nuc, textfont=dict(family='Courier New',size=14),
                          showlegend=False), row=row, col=1)
row +=1
# Attention row
if show_att:
    grid.add_trace(go.Scatter(x=region.pos, y=region.attention, mode='lines',
                              line=dict(color='crimson',width=2), name='Attention'),
                   row=row, col=1)
    row +=1
# GWAS row
if show_gwas:
    grid.add_trace(go.Bar(x=region.pos, y=region.gvals, marker_color='black',
                          name='-log10(p)'), row=row, col=1)
    snp = region[region.gmask==1]
    grid.add_trace(go.Scatter(x=snp.pos, y=snp.gvals*1.1, mode='markers',
        marker_symbol='line-ns-open', marker_color='black', marker_size=10,
        name='SNPs', hovertext=[f"rsID: rs{10000+i}" for i in snp.index],
        hoverinfo='text'), row=row, col=1)
    row +=1
# Conservation row
if show_cons:
    grid.add_trace(go.Scatter(x=region.pos, y=region.cons, mode='lines',
                              line=dict(color='blue',width=2), name='Conservation'),
                   row=row, col=1)
    row +=1
# TFBS row
if show_tfbs:
    pts = region[region.tfbs==1]
    grid.add_trace(go.Scatter(x=pts.pos, y=[0.5]*len(pts), mode='markers',
                              marker_symbol='triangle-up', marker_color='green',
                              marker_size=10, name='TFBS'), row=row, col=1)
    row +=1
# Regulatory row
if show_reg:
    pts = region[region.reg==1]
    grid.add_trace(go.Scatter(x=pts.pos, y=[0.5]*len(pts), mode='markers',
                              marker_symbol='diamond', marker_color='purple',
                              marker_size=10, name='Regulatory Features'),
                   row=row, col=1)
    row +=1
# ATAC row
if show_atac:
    pts = region[region.atac==1]
    grid.add_trace(go.Scatter(x=pts.pos, y=[0.5]*len(pts), mode='markers',
                              marker_symbol='triangle-down', marker_color='orange',
                              marker_size=10, name='ATAC'), row=row, col=1)
    row +=1
# Final layout
grid.update_yaxes(visible=False)
grid.update_xaxes(title_text='Genomic Position', row=rows)
grid.update_layout(height=100*rows, showlegend=True, margin=dict(l=50,r=50,t=50,b=50))
st.plotly_chart(grid, use_container_width=True)

# DRL Metrics and explanation
st.subheader('DRL Agent Parameters & Metrics')
st.markdown(
    'The DRL agent optimizes a composite reward balancing prediction accuracy (low loss), ' +
    'interpretability (focused attention), and biological overlap (GWAS). ' +
    'Bias and scaling adjustments are the discrete action steps modifying attention distribution.'
)
fig2 = go.Figure(go.Bar(
    x=metrics.Metric, y=metrics.Value, marker_color='teal', text=metrics.Value, textposition='outside'
))
fig2.update_layout(title='DRL Metrics Overview', xaxis_tickangle=-45, margin=dict(b=150))
st.plotly_chart(fig2, use_container_width=True)

# Enrichment panels
st.subheader('GO-term Enrichment')
st.table(go_df)
st.subheader('KEGG Pathway Enrichment')
st.table(kegg_df)

# LLM Interpretation
st.subheader('LLM Interpretation & Insights')
if st.button('Generate Detailed Insights'):
    st.info(
        'The DRL-tuned model exhibits strong attention at bases 180–260, overlapping SNP rs12345 ' +
        '(-log10 p=6.5) and dense TFBS motif clusters, indicating enhancer-like function. ' +
        'Conservation is elevated (mean=0.78), suggesting evolutionary constraint. ' +
        'A secondary hotspot at 480–520 corresponds to regulatory mark accumulation and ATAC peaks, ' +
        'implying promoter accessibility. GO enrichment in RNA binding and transcription regulation, ' +
        'alongside KEGG pathways (Cell cycle, Spliceosome), supports roles in chromatin remodeling and splicing lineage.'
    )

# Footer
st.markdown('---')

st.caption("Synthetic data for interactive,live demo of lncRNA features.")
