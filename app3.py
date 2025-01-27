import streamlit as st
import plotly.graph_objects as go
import math
import json
from pathlib import Path
import openai
from typing import List, Dict
from langchain_openai import ChatOpenAI
import asyncio
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import functools
import time
import re

def main():
    # Initialize session state variables
    if 'embeddings_cache' not in st.session_state:
        st.session_state['embeddings_cache'] = {}
    if 'similar_chunks_cache' not in st.session_state:
        st.session_state['similar_chunks_cache'] = {}
    if 'selected_cluster' not in st.session_state:
        st.session_state['selected_cluster'] = None
    if 'selected_chunk' not in st.session_state:
        st.session_state['selected_chunk'] = None
    if 'similar_chunks' not in st.session_state:
        st.session_state['similar_chunks'] = None

    # Initialize OpenAI client
    client = openai.Client(api_key=st.secrets["OPENAI_API_KEY"])
    llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")

    # Custom theme CSS
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        :root {
            --pink-accent: #FF69B4;
        }
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #1a1a1a;
            color: #e0e0e0;
        }
        .stApp {
            background-color: #1a1a1a;
        }
        .stApp h1, .stApp h2, .stApp h3 {
            color: #e0e0e0;
            font-weight: 300;
            text-align: center;
            letter-spacing: 1px;
        }
        .stApp h1 {
            border-bottom: 2px solid var(--pink-accent);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .stDataFrame {
            background-color: #2a2a2a;
            border-radius: 10px;
            border: 1px solid #3a3a3a;
        }
        .stButton > button {
            background-color: #4a4a4a;
            color: #e0e0e0;
            border-radius: 25px;
            border: none;
            padding: 10px 25px;
            font-weight: bold;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stButton > button:hover {
            background-color: #5a5a5a;
            box-shadow: 0 0 15px var(--pink-accent);
            transform: translateY(-2px);
        }
        .stTextArea > div > div > textarea {
            background-color: #2a2a2a;
            color: #e0e0e0;
            border: 1px solid #3a3a3a;
            border-radius: 10px;
            padding: 10px;
        }
        .stExpander {
            background-color: #2a2a2a;
            border: 1px solid #3a3a3a;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stExpander:hover {
            border-color: var(--pink-accent);
        }
        .stExpander > div:first-child {
            border-bottom: 1px solid #3a3a3a;
            padding: 10px;
            font-weight: bold;
        }
        .stExpander > div:last-child {
            padding: 15px;
        }
        .stProgress > div > div > div > div {
            background-color: var(--pink-accent);
        }
    </style>
    """, unsafe_allow_html=True)

    @st.cache_data(ttl=3600)
    def load_data():
        try:
            with open("DATA/output/mindmap_structure.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            st.error("mindmap_structure.json not found")
            return None

    def get_embedding(text: str) -> List[float]:
        """Get embedding for a text using OpenAI Ada"""
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def find_similar_chunks(query_text: str, all_chunks: List[Dict], top_k: int = 6) -> List[Dict]:
        """Find most similar chunks to the query text"""
        query_embedding = get_embedding(query_text)
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        chunk_embeddings = [get_embedding(text) for text in chunk_texts]
        
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        chunk_similarities = list(zip(all_chunks, similarities))
        chunk_similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, sim in chunk_similarities[:top_k]]

    def remove_quotes(text: str) -> str:
        """Remove leading and trailing double quotes from a string"""
        return re.sub(r'^"|"$', '', text)

    def create_mindmap_visualization(center_text, nodes_data, is_cluster_view=True):
        """Creates an improved mindmap visualization with Plotly"""
        num_nodes = len(nodes_data)
        if num_nodes == 0:
            return None
        
        radius = 450  # Increased radius for larger mindmap
        angles = [2 * math.pi * i / num_nodes for i in range(num_nodes)]
        
        x_center, y_center = [0], [0]
        x_nodes = [math.cos(angle) * radius for angle in angles]
        y_nodes = [math.sin(angle) * radius for angle in angles]
        
        fig = go.Figure()
        
        # Add connections with gradient
        for i, (x, y) in enumerate(zip(x_nodes, y_nodes)):
            fig.add_trace(go.Scatter(
                x=[0, x],
                y=[0, y],
                mode='lines',
                line=dict(
                    color='#4a4a4a',
                    width=2,
                    shape='spline',
                ),
                opacity=0.6,
                hoverinfo='none'
            ))
        
        # Center node
        fig.add_trace(go.Scatter(
            x=x_center,
            y=y_center,
            mode='markers+text',
            marker=dict(
                size=140,
                color='#2a2a2a',
                symbol='circle',
                line=dict(color='#FF69B4', width=4)
            ),
            text=[remove_quotes(center_text).upper()],
            textposition='middle center',
            textfont=dict(size=22, color='#ffffff', family='Roboto'),
            hoverinfo='text'
        ))
        
        # Surrounding nodes with hover effect
        hover_template = (
            '<b>%{text}</b><br>' +
            ('Cluster Size: %{customdata[1]} chunks' if is_cluster_view else '')
        )
        
        node_colors = ['#3a3a3a', '#4a4a4a', '#5a5a5a', '#6a6a6a', '#7a7a7a', '#FF69B4']
        
        fig.add_trace(go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode='markers+text',
            marker=dict(
                size=110,
                color=[node_colors[i % len(node_colors)] for i in range(num_nodes)],
                symbol='circle',
                line=dict(color='#e0e0e0', width=3)
            ),
            text=[remove_quotes(d["title"]).upper() for d in nodes_data],
            textposition='middle center',
            textfont=dict(size=18, color='#ffffff', family='Roboto'),
            hovertemplate=hover_template,
            customdata=[
                [
                    d["id"],
                    d.get("total_chunks", 0)
                ] for d in nodes_data
            ]
        ))
        
        # Update layout with improved aesthetics
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            width=1400,
            height=1200,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-radius*1.2, radius*1.2]
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-radius*1.2, radius*1.2]
            ),
            margin=dict(t=50, b=50),
            font=dict(color='#e0e0e0'),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(
                    label="Reset View",
                    method="relayout",
                    args=[{"xaxis.range": [-radius*1.2, radius*1.2],
                           "yaxis.range": [-radius*1.2, radius*1.2]}]
                )]
            )]
        )
        
        return fig

    # Main application logic
    st.title("Mindmap")
    
    data = load_data()
    if not data:
        return
    
    # Extract all chunks from all clusters
    all_chunks = [chunk for cluster in data['clusters'] for chunk in cluster['chunks']]
    
    # Main clusters view
    if st.session_state['selected_cluster'] is None:
        st.header("Clusters Overview")
        
        clusters_data = [
            {
                "id": cluster["cluster_id"],
                "title": remove_quotes(cluster["cluster_title"]),
                "total_chunks": cluster["total_chunks"]
            }
            for cluster in data["clusters"]
        ]
        
        fig = create_mindmap_visualization("DIGITAL STRATEGY", clusters_data, is_cluster_view=True)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Select a cluster to explore")
            cols = st.columns(3)
            for i, cluster in enumerate(clusters_data):
                if i % 3 == 0 and i > 0:
                    cols = st.columns(3)
                if cols[i % 3].button(f"Cluster {cluster['id']}: {cluster['title']}"):
                    st.session_state['selected_cluster'] = cluster['id']
                    st.session_state['selected_chunk'] = None
                    st.session_state['similar_chunks'] = None
                    st.rerun()
    
    # Cluster view or Similar chunks view
    else:
        selected_cluster = next(
            cluster for cluster in data["clusters"] 
            if cluster["cluster_id"] == st.session_state['selected_cluster']
        )
        
        # Handle back navigation
        if st.session_state['selected_chunk'] is None:
            st.header(f"Cluster: {remove_quotes(selected_cluster['cluster_title'])}")
            if st.button("← Back to Overview"):
                st.session_state['selected_cluster'] = None
                st.rerun()
        else:
            st.header(f"Similar Content: {remove_quotes(st.session_state['selected_chunk']['title'])}")
            if st.button("← Back to Cluster"):
                st.session_state['selected_chunk'] = None
                st.session_state['similar_chunks'] = None
                st.rerun()
        
        if st.session_state['selected_chunk'] is None:
            # Show cluster chunks
            chunks_data = [
                {
                    "id": chunk["chunk_id"],
                    "title": remove_quotes(chunk["title"])
                }
                for chunk in selected_cluster["chunks"]
            ]
            
            fig = create_mindmap_visualization(
                selected_cluster["cluster_title"],
                chunks_data,
                is_cluster_view=False
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Show chunk details with explore button
            st.subheader("Source Texts")
            for chunk in selected_cluster["chunks"]:
                with st.expander(f"{remove_quotes(chunk['title'])}"):
                    st.text_area("Text:", chunk["text"], height=150)
                    if st.button(f"Explore Similar → {chunk['chunk_id']}", key=f"explore_{chunk['chunk_id']}"):
                        st.session_state['selected_chunk'] = chunk
                        with st.spinner("Finding similar chunks..."):
                            progress_bar = st.progress(0)
                            start_time = time.time()
                            st.session_state['similar_chunks'] = find_similar_chunks(chunk['text'], all_chunks)
                            end_time = time.time()
                            progress_bar.progress(100)
                        st.success(f"Found similar chunks in {end_time - start_time:.2f} seconds")
                        st.rerun()
        
        else:
            # Show similar chunks
            if st.session_state['similar_chunks']:
                similar_chunks_data = [
                    {
                        "id": idx,
                        "title": remove_quotes(chunk["title"])
                    }
                    for idx, chunk in enumerate(st.session_state['similar_chunks'])
                ]
                
                fig = create_mindmap_visualization(
                    st.session_state['selected_chunk']["title"],
                    similar_chunks_data,
                    is_cluster_view=False
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Similar Content")
                for chunk in st.session_state['similar_chunks']:
                    with st.expander(f"{remove_quotes(chunk['title'])}"):
                        st.text_area("Text:", chunk["text"], height=150)

if __name__ == "__main__":
    main()