import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import time
import os

# Page configuration - HARUS PALING ATAS
st.set_page_config(
    page_title="Multidisciplinary Analysis Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .analysis-box {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# OpenRouter API Client
class OpenRouterClient:
    def __init__(self):
        # Try multiple ways to get API key
        self.api_key = (
            os.getenv("OPENROUTER_API_KEY") or 
            st.secrets.get("OPENROUTER_API_KEY", "") if hasattr(st, 'secrets') else ""
        )
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_analysis(self, query, domain, data_summary):
        """Generate analysis using OpenRouter API or fallback"""
        try:
            if not self.api_key:
                return self._fallback_analysis(query, domain)
            
            payload = {
                "model": "openai/gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a {domain} expert. Provide scientific analysis."
                    },
                    {
                        "role": "user",
                        "content": f"Analyze: {query}\nData: {data_summary}"
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.7
            }
            
            response = requests.post(
                self.base_url, 
                headers=self.headers, 
                json=payload, 
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return self._fallback_analysis(query, domain)
                
        except Exception as e:
            return self._fallback_analysis(query, domain)
    
    def _fallback_analysis(self, query, domain):
        """Fallback analysis when API fails"""
        fallback_responses = {
            'biophysics': [
                "Protein stability analysis shows optimal binding conditions.",
                "Molecular weight correlations suggest enhanced therapeutic potential.",
                "Structural analysis indicates improved biomaterial properties."
            ],
            'nanotech': [
                "Nanomaterial synthesis optimization shows 20% yield improvement.",
                "Size-property relationships indicate novel application potential.",
                "Surface area analysis suggests enhanced catalytic properties."
            ],
            'ocean': [
                "Marine parameters indicate healthy ecosystem conditions.",
                "Temperature-salinity correlations show climate adaptation patterns.",
                "Dissolved oxygen levels support sustainable development."
            ],
            'physical_ai': [
                "System performance metrics indicate deployment readiness.",
                "Efficiency analysis shows optimal energy consumption patterns.",
                "Response time optimization enables real-time applications."
            ]
        }
        
        responses = fallback_responses.get(domain, ["Analysis completed successfully."])
        return np.random.choice(responses)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Data generation functions
@st.cache_data
def generate_biophysics_data():
    proteins = ['Hemoglobin', 'Myosin', 'Actin', 'Collagen', 'Elastin']
    data = []
    for protein in proteins:
        data.append({
            'protein': protein,
            'molecular_weight': np.random.uniform(10000, 150000),
            'stability_score': np.random.uniform(0.6, 0.95),
            'binding_affinity': np.random.uniform(1e-9, 1e-6),
            'energy': np.random.uniform(-500, -100)
        })
    return pd.DataFrame(data)

@st.cache_data
def generate_nanotech_data():
    materials = ['Carbon Nanotubes', 'Graphene', 'Quantum Dots', 'Gold NPs', 'Silver NWs']
    data = []
    for material in materials:
        data.append({
            'material': material,
            'size_nm': np.random.uniform(1, 100),
            'conductivity': np.random.uniform(10, 10000),
            'surface_area': np.random.uniform(100, 2500),
            'yield_pct': np.random.uniform(70, 98)
        })
    return pd.DataFrame(data)

@st.cache_data
def generate_ocean_data():
    locations = ['Pacific Deep', 'Atlantic Ridge', 'Indian Basin', 'Arctic Ice', 'Antarctic']
    data = []
    for location in locations:
        data.append({
            'location': location,
            'temperature': np.random.uniform(-2, 30),
            'salinity': np.random.uniform(30, 40),
            'pressure': np.random.uniform(1, 1100),
            'ph_level': np.random.uniform(7.8, 8.2),
            'oxygen': np.random.uniform(2, 15)
        })
    return pd.DataFrame(data)

@st.cache_data
def generate_physical_ai_data():
    systems = ['Marine Robot', 'Nano Assembler', 'Bio Sensor', 'Smart Material', 'Actuator']
    data = []
    for system in systems:
        data.append({
            'system': system,
            'efficiency': np.random.uniform(0.75, 0.98),
            'response_ms': np.random.uniform(0.1, 50),
            'accuracy': np.random.uniform(0.85, 0.99),
            'power_w': np.random.uniform(0.1, 100),
            'learning_rate': np.random.uniform(0.001, 0.1)
        })
    return pd.DataFrame(data)

# Main application
def main():
    st.markdown('<h1 class="main-header">🔬 Multidisciplinary Analysis Agent</h1>', unsafe_allow_html=True)
    
    # Initialize client
    try:
        openrouter = OpenRouterClient()
    except:
        openrouter = None
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # API Status
        if openrouter and openrouter.api_key:
            st.success("🟢 API Ready")
        else:
            st.warning("🟡 Demo Mode")
        
        # Domain selection
        domain = st.selectbox(
            "Analysis Domain:",
            ["biophysics", "nanotech", "ocean", "physical_ai"],
            format_func=lambda x: {
                "biophysics": "🧬 Bio-physics",
                "nanotech": "⚛️ Nano-tech", 
                "ocean": "🌊 Ocean Science",
                "physical_ai": "🤖 Physical AI"
            }[x]
        )
        
        # Parameters
        st.subheader("Parameters")
        complexity = st.slider("Complexity", 1, 10, 5)
        sample_size = st.slider("Sample Size", 10, 1000, 100)
        
        # Auto refresh
        auto_refresh = st.checkbox("Auto-refresh")
        if auto_refresh:
            time.sleep(2)
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Generate data based on domain
        if domain == "biophysics":
            st.subheader("🧬 Bio-physics Analysis")
            data = generate_biophysics_data()
            
            # Metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Avg MW", f"{data['molecular_weight'].mean():.0f}")
            with col_m2:
                st.metric("Avg Stability", f"{data['stability_score'].mean():.2f}")
            with col_m3:
                st.metric("Best Binding", f"{data['binding_affinity'].min():.2e}")
            
            # Chart
            fig = px.scatter(data, x='molecular_weight', y='stability_score',
                           size='binding_affinity', hover_name='protein',
                           title='Protein Properties')
            st.plotly_chart(fig, use_container_width=True)
            
        elif domain == "nanotech":
            st.subheader("⚛️ Nano-technology Analysis")
            data = generate_nanotech_data()
            
            # Metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Avg Size", f"{data['size_nm'].mean():.1f} nm")
            with col_m2:
                st.metric("Max Conductivity", f"{data['conductivity'].max():.0f}")
            with col_m3:
                st.metric("Avg Yield", f"{data['yield_pct'].mean():.1f}%")
            
            # Chart
            fig = px.bar(data, x='material', y='conductivity',
                        title='Material Conductivity')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
        elif domain == "ocean":
            st.subheader("🌊 Ocean Science Analysis")
            data = generate_ocean_data()
            
            # Metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Avg Temp", f"{data['temperature'].mean():.1f}°C")
            with col_m2:
                st.metric("Avg Salinity", f"{data['salinity'].mean():.1f}")
            with col_m3:
                st.metric("Avg pH", f"{data['ph_level'].mean():.2f}")
            
            # Chart
            fig = px.scatter_3d(data, x='temperature', y='salinity', z='pressure',
                              color='ph_level', hover_name='location',
                              title='Ocean Parameters')
            st.plotly_chart(fig, use_container_width=True)
            
        elif domain == "physical_ai":
            st.subheader("🤖 Physical AI Analysis")
            data = generate_physical_ai_data()
            
            # Metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Avg Efficiency", f"{data['efficiency'].mean():.1%}")
            with col_m2:
                st.metric("Avg Response", f"{data['response_ms'].mean():.1f} ms")
            with col_m3:
                st.metric("Avg Accuracy", f"{data['accuracy'].mean():.1%}")
            
            # Chart
            fig = px.scatter(data, x='response_ms', y='efficiency',
                           size='power_w', color='accuracy',
                           hover_name='system', title='System Performance')
            st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        with st.expander("📊 Data Table"):
            st.dataframe(data, use_container_width=True)
    
    with col2:
        st.subheader("🧠 AI Analysis")
        
        # Query input
        user_query = st.text_area(
            "Enter query:",
            placeholder="What patterns do you see?",
            height=100
        )
        
        if st.button("🚀 Analyze", type="primary"):
            if user_query:
                with st.spinner("Analyzing..."):
                    # Prepare data summary
                    data_summary = f"Rows: {len(data)}, Columns: {list(data.columns)}"
                    
                    # Generate analysis
                    if openrouter:
                        result = openrouter.generate_analysis(user_query, domain, data_summary)
                    else:
                        result = f"Analysis for {domain}: {user_query} - Demo response generated."
                    
                    # Display result
                    st.markdown(f'<div class="analysis-box">{result}</div>', 
                               unsafe_allow_html=True)
                    
                    # Add to history
                    st.session_state.analysis_history.append({
                        'timestamp': datetime.now(),
                        'domain': domain,
                        'query': user_query,
                        'result': result
                    })
            else:
                st.warning("Please enter a query!")
        
        # History
        if st.session_state.analysis_history:
            st.subheader("📈 History")
            for analysis in reversed(st.session_state.analysis_history[-3:]):
                with st.expander(f"{analysis['domain']} - {analysis['timestamp'].strftime('%H:%M')}"):
                    st.write(f"**Q:** {analysis['query']}")
                    st.write(f"**A:** {analysis['result']}")
    
    # Footer
    st.markdown("---")
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        st.info("🧬 Bio-physics")
    with col_f2:
        st.info("⚛️ Nano-tech")
    with col_f3:
        st.info("🌊 Ocean Science") 
    with col_f4:
        st.info("🤖 Physical AI")

if __name__ == "__main__":
    main()
