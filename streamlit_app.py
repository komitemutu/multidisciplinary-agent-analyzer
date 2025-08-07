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
from typing import Dict, List, Any

# Page configuration
st.set_page_config(
    page_title="Multidisciplinary Analysis Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with modern design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .analysis-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.3);
    }
    .domain-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# OpenRouter API Configuration
class OpenRouterClient:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", "")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://multidisciplinary-agent.streamlit.app",
            "X-Title": "Multidisciplinary Analysis Agent"
        }
    
    def generate_analysis(self, query: str, domain: str, data: pd.DataFrame, context: Dict[str, Any]) -> str:
        """Generate analysis using OpenRouter API"""
        try:
            # Prepare data summary for context
            data_summary = {
                "rows": len(data),
                "columns": list(data.columns),
                "statistics": data.describe().to_dict() if not data.empty else {}
            }
            
            # Create sophisticated prompt based on domain
            domain_prompts = {
                "biophysics": f"""
                As a bio-physics expert analyzing protein data, provide insights on:
                - Molecular dynamics and stability patterns
                - Structure-function relationships
                - Potential biomaterial applications
                - Optimization strategies for {query}
                
                Data context: {json.dumps(data_summary, indent=2)}
                Analysis parameters: {json.dumps(context, indent=2)}
                """,
                
                "nanotech": f"""
                As a nanotechnology specialist examining nanomaterial properties, analyze:
                - Size-property relationships
                - Synthesis optimization opportunities
                - Application potential assessment
                - Performance enhancement strategies for {query}
                
                Data context: {json.dumps(data_summary, indent=2)}
                Analysis parameters: {json.dumps(context, indent=2)}
                """,
                
                "ocean": f"""
                As an ocean science researcher studying marine environments, evaluate:
                - Environmental parameter correlations
                - Climate change implications
                - Ecosystem health indicators
                - Sustainable development opportunities for {query}
                
                Data context: {json.dumps(data_summary, indent=2)}
                Analysis parameters: {json.dumps(context, indent=2)}
                """,
                
                "physical_ai": f"""
                As a physical AI systems engineer, assess:
                - System performance optimization
                - Real-world deployment readiness
                - Adaptive learning capabilities
                - Integration strategies for {query}
                
                Data context: {json.dumps(data_summary, indent=2)}
                Analysis parameters: {json.dumps(context, indent=2)}
                """
            }
            
            payload = {
                "model": "anthropic/claude-3.5-sonnet",  # Free tier model
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a multidisciplinary scientific AI agent specializing in bio-physics, nanotechnology, ocean science, and physical AI systems. Provide detailed, actionable insights based on data analysis."
                    },
                    {
                        "role": "user",
                        "content": domain_prompts.get(domain, f"Analyze this query: {query}\nData: {data_summary}")
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"API Error {response.status_code}: {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"Connection error: {str(e)}"
        except Exception as e:
            return f"Analysis error: {str(e)}"

# Initialize OpenRouter client
@st.cache_resource
def get_openrouter_client():
    return OpenRouterClient()

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_data' not in st.session_state:
    st.session_state.current_data = None

# Enhanced dummy data generators with more realistic patterns
@st.cache_data(ttl=300)  # Cache for 5 minutes
def generate_biophysics_data():
    """Generate realistic bio-physics data"""
    proteins = ['Hemoglobin', 'Myosin', 'Actin', 'Collagen', 'Elastin', 'Keratin', 'Albumin', 'Fibrinogen']
    data = []
    
    for i, protein in enumerate(proteins):
        # Add realistic correlations
        base_mw = 50000 + i * 15000
        stability = 0.7 + (i % 3) * 0.1 + np.random.normal(0, 0.05)
        binding = 1e-9 * (10 ** (i * 0.5)) + np.random.normal(0, 1e-10)
        
        data.append({
            'protein': protein,
            'molecular_weight': base_mw + np.random.normal(0, 5000),
            'stability_score': max(0.5, min(1.0, stability)),
            'binding_affinity': max(1e-12, binding),
            'conformational_energy': -200 - i * 50 + np.random.normal(0, 25),
            'hydrophobicity': np.random.uniform(-2, 2),
            'isoelectric_point': np.random.uniform(4, 10)
        })
    return pd.DataFrame(data)

@st.cache_data(ttl=300)
def generate_nanotech_data():
    """Generate realistic nanotechnology data"""
    materials = ['Carbon Nanotubes', 'Graphene', 'Quantum Dots', 'Gold Nanoparticles', 'Silver Nanowires', 
                'Titanium Dioxide', 'Silicon Nanowires', 'Fullerenes']
    data = []
    
    for i, material in enumerate(materials):
        # Realistic property correlations
        size = 1 + i * 12 + np.random.exponential(10)
        conductivity = 1000 / (1 + size/10) if 'Nano' in material else np.random.uniform(0.1, 100)
        
        data.append({
            'material': material,
            'size_nm': size,
            'conductivity': conductivity + np.random.normal(0, conductivity * 0.1),
            'surface_area_m2g': 2500 / (1 + size/5) + np.random.normal(0, 100),
            'synthesis_yield': 0.6 + 0.3 * np.random.beta(2, 1),
            'band_gap_ev': np.random.uniform(0, 6),
            'thermal_stability_c': 200 + i * 100 + np.random.normal(0, 50)
        })
    return pd.DataFrame(data)

@st.cache_data(ttl=300)
def generate_ocean_data():
    """Generate realistic ocean science data"""
    locations = ['Pacific Abyssal', 'Atlantic Ridge', 'Indian Gyre', 'Arctic Basin', 'Antarctic Circumpolar',
                'Mediterranean Deep', 'Coral Triangle', 'Benguela Upwelling']
    data = []
    
    for i, location in enumerate(locations):
        # Realistic oceanographic correlations
        latitude = -80 + i * 20 + np.random.normal(0, 5)
        temp = 15 + latitude * 0.3 + np.random.normal(0, 2)
        depth = 500 + i * 800 + np.random.exponential(500)
        
        data.append({
            'location': location,
            'latitude': latitude,
            'depth_m': depth,
            'temperature_c': max(-2, temp - depth/1000),
            'salinity_ppt': 34 + np.random.normal(0, 2),
            'pressure_bar': 1 + depth/10,
            'ph_level': 8.1 + np.random.normal(0, 0.1),
            'dissolved_oxygen': 15 * np.exp(-depth/1000) + np.random.normal(0, 1),
            'chlorophyll_mg_m3': max(0.1, 5 * np.exp(-depth/100) + np.random.exponential(1))
        })
    return pd.DataFrame(data)

@st.cache_data(ttl=300)
def generate_physical_ai_data():
    """Generate realistic physical AI data"""
    systems = ['Marine AUV', 'Nano Assembler', 'Bio Sensor Array', 'Smart Metamaterial', 'Adaptive Gripper',
              'Swarm Robots', 'Neural Prosthetic', 'Soft Actuator']
    data = []
    
    for i, system in enumerate(systems):
        # Realistic performance trade-offs
        complexity = 1 + i * 0.3
        efficiency = 0.95 / (1 + complexity * 0.1) + np.random.normal(0, 0.02)
        response_time = 0.1 * complexity ** 1.5 + np.random.exponential(5)
        
        data.append({
            'system': system,
            'complexity_index': complexity,
            'efficiency': max(0.5, min(0.99, efficiency)),
            'response_time_ms': response_time,
            'accuracy': 0.85 + 0.1 * (1 - complexity/10) + np.random.normal(0, 0.02),
            'energy_consumption_w': 0.1 * complexity ** 2 + np.random.exponential(10),
            'learning_rate': 0.001 + 0.01 * np.random.beta(1, 3),
            'adaptability_score': np.random.uniform(0.6, 0.9)
        })
    return pd.DataFrame(data)

# Enhanced visualization functions
def create_correlation_heatmap(data, title):
    """Create correlation heatmap for numerical data"""
    numeric_data = data.select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 1:
        corr_matrix = numeric_data.corr()
        fig = px.imshow(corr_matrix, 
                       color_continuous_scale='RdBu',
                       title=title)
        return fig
    return None

def create_3d_scatter(data, x, y, z, color, title):
    """Create 3D scatter plot"""
    fig = px.scatter_3d(data, x=x, y=y, z=z, color=color,
                       title=title, height=500)
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    return fig

# Main application
def main():
    st.markdown('<h1 class="main-header">🔬 Multidisciplinary Analysis Agent</h1>', unsafe_allow_html=True)
    
    # Initialize OpenRouter client
    openrouter = get_openrouter_client()
    
    # Sidebar with enhanced controls
    with st.sidebar:
        st.header("🎛️ Advanced Control Panel")
        
        # API Status indicator
        if openrouter.api_key:
            st.success("🟢 OpenRouter API Connected")
        else:
            st.error("🔴 OpenRouter API Key Missing")
            st.info("Add OPENROUTER_API_KEY to environment variables")
        
        # Domain selection with icons
        domain = st.selectbox(
            "Select Analysis Domain:",
            ["biophysics", "nanotech", "ocean", "physical_ai"],
            format_func=lambda x: {
                "biophysics": "🧬 Bio-physics Research",
                "nanotech": "⚛️ Nano-technology Labs",
                "ocean": "🌊 Ocean Science Station",
                "physical_ai": "🤖 Physical AI Systems"
            }[x]
        )
        
        # Advanced analysis parameters
        st.subheader("🔬 Analysis Configuration")
        complexity = st.slider("Analysis Complexity", 1, 10, 7)
        sample_size = st.slider("Sample Size", 50, 2000, 500)
        confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)
        
        # Real-time features
        st.subheader("⚡ Real-time Features")
        auto_refresh = st.checkbox("Auto-refresh data", value=False)
        live_analysis = st.checkbox("Live AI analysis", value=False)
        
        if auto_refresh:
            refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 30)
            if st.session_state.get('last_refresh', 0) + refresh_interval < time.time():
                st.session_state.last_refresh = time.time()
                st.rerun()
        
        # Export options
        st.subheader("📊 Export Options")
        if st.button("📥 Export Data", type="secondary"):
            st.session_state.export_requested = True
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📊 Data Analysis", "🤖 AI Insights", "📈 Historical Trends"])
    
    with tab1:
        # Generate and display data based on selected domain
        if domain == "biophysics":
            st.subheader("🧬 Bio-physics Research Dashboard")
            data = generate_biophysics_data()
            
            # Enhanced metrics with gradients
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                avg_mw = data['molecular_weight'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Avg Molecular Weight</h3>
                    <h2>{avg_mw:.0f} Da</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m2:
                avg_stability = data['stability_score'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Avg Stability Score</h3>
                    <h2>{avg_stability:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m3:
                best_binding = data['binding_affinity'].min()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Best Binding Affinity</h3>
                    <h2>{best_binding:.2e} M</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m4:
                energy_range = data['conformational_energy'].max() - data['conformational_energy'].min()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Energy Range</h3>
                    <h2>{energy_range:.0f} kJ/mol</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Advanced visualizations
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                fig1 = px.scatter(data, x='molecular_weight', y='stability_score', 
                                size='binding_affinity', color='hydrophobicity',
                                hover_name='protein', title='Protein Properties Matrix')
                fig1.update_traces(marker=dict(opacity=0.8, sizeref=2.*max(data['binding_affinity'])/(40.**2)))
                st.plotly_chart(fig1, use_container_width=True)
            
            with col_v2:
                fig2 = px.bar(data, x='protein', y='isoelectric_point',
                            color='conformational_energy', title='Isoelectric Points')
                fig2.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig2, use_container_width=True)
            
            # 3D visualization
            fig3d = create_3d_scatter(data, 'molecular_weight', 'stability_score', 
                                    'conformational_energy', 'hydrophobicity',
                                    '3D Protein Analysis Space')
            st.plotly_chart(fig3d, use_container_width=True)
            
        elif domain == "nanotech":
            st.subheader("⚛️ Nano-technology Laboratory")
            data = generate_nanotech_data()
            
            # Metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            metrics = [
                ("Avg Size", f"{data['size_nm'].mean():.1f} nm"),
                ("Max Conductivity", f"{data['conductivity'].max():.0f} S/m"),
                ("Avg Surface Area", f"{data['surface_area_m2g'].mean():.0f} m²/g"),
                ("Avg Yield", f"{data['synthesis_yield'].mean():.1%}")
            ]
            
            for col, (title, value) in zip([col_m1, col_m2, col_m3, col_m4], metrics):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{title}</h3>
                        <h2>{value}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Visualizations
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                fig1 = px.scatter(data, x='size_nm', y='conductivity', 
                                size='surface_area_m2g', color='band_gap_ev',
                                hover_name='material', title='Size vs Conductivity',
                                log_y=True)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col_v2:
                fig2 = px.parallel_coordinates(data, 
                                             dimensions=['size_nm', 'conductivity', 'surface_area_m2g', 
                                                       'synthesis_yield', 'thermal_stability_c'],
                                             color='band_gap_ev', title='Material Properties Parallel Plot')
                st.plotly_chart(fig2, use_container_width=True)
            
        elif domain == "ocean":
            st.subheader("🌊 Ocean Science Monitoring Station")
            data = generate_ocean_data()
            
            # Metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            metrics = [
                ("Avg Temperature", f"{data['temperature_c'].mean():.1f}°C"),
                ("Avg Salinity", f"{data['salinity_ppt'].mean():.1f} ppt"),
                ("Max Depth", f"{data['depth_m'].max():.0f} m"),
                ("Avg pH", f"{data['ph_level'].mean():.2f}")
            ]
            
            for col, (title, value) in zip([col_m1, col_m2, col_m3, col_m4], metrics):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{title}</h3>
                        <h2>{value}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Ocean-specific visualizations
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                fig1 = px.scatter(data, x='temperature_c', y='dissolved_oxygen',
                                size='depth_m', color='latitude',
                                hover_name='location', title='Temperature vs Oxygen Profile')
                st.plotly_chart(fig1, use_container_width=True)
            
            with col_v2:
                fig2 = px.scatter(data, x='depth_m', y='chlorophyll_mg_m3',
                                color='temperature_c', size='salinity_ppt',
                                hover_name='location', title='Depth vs Chlorophyll',
                                log_y=True)
                st.plotly_chart(fig2, use_container_width=True)
            
            # 3D Ocean visualization
            fig3d = create_3d_scatter(data, 'latitude', 'depth_m', 'temperature_c', 
                                    'ph_level', '3D Ocean Parameter Space')
            st.plotly_chart(fig3d, use_container_width=True)
            
        elif domain == "physical_ai":
            st.subheader("🤖 Physical AI Systems Laboratory")
            data = generate_physical_ai_data()
            
            # Metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            metrics = [
                ("Avg Efficiency", f"{data['efficiency'].mean():.1%}"),
                ("Avg Response", f"{data['response_time_ms'].mean():.1f} ms"),
                ("Avg Accuracy", f"{data['accuracy'].mean():.1%}"),
                ("Total Power", f"{data['energy_consumption_w'].sum():.1f} W")
            ]
            
            for col, (title, value) in zip([col_m1, col_m2, col_m3, col_m4], metrics):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{title}</h3>
                        <h2>{value}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            # AI-specific visualizations
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                fig1 = px.scatter(data, x='response_time_ms', y='efficiency',
                                size='energy_consumption_w', color='complexity_index',
                                hover_name='system', title='Performance vs Efficiency')
                st.plotly_chart(fig1, use_container_width=True)
            
            with col_v2:
                # Radar chart for system comparison
                categories = ['efficiency', 'accuracy', 'adaptability_score']
                fig2 = go.Figure()
                
                for _, row in data.iterrows():
                    values = [row[cat] for cat in categories]
                    values += values[:1]  # Complete the circle
                    
                    fig2.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories + categories[:1],
                        fill='toself',
                        name=row['system'],
                        opacity=0.6
                    ))
                
                fig2.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="System Performance Radar",
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        # Correlation heatmap for current domain
        heatmap = create_correlation_heatmap(data, f"{domain.title()} Parameter Correlations")
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True)
        
        # Data table with search and filter
        with st.expander("📊 Interactive Data Table", expanded=False):
            # Search functionality
            search_term = st.text_input("🔍 Search data:", placeholder="Enter search term...")
            
            if search_term:
                # Filter data based on search term
                text_columns = data.select_dtypes(include=['object']).columns
                mask = data[text_columns].apply(lambda x: x.astype(str).str.contains(search_term, case=False, na=False)).any(axis=1)
                filtered_data = data[mask]
            else:
                filtered_data = data
            
            st.dataframe(filtered_data, use_container_width=True, height=300)
            
            # Download button
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{domain}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with tab2:
        st.subheader("🤖 AI-Powered Analysis")
        
        col_ai1, col_ai2 = st.columns([2, 1])
        
        with col_ai1:
            # Enhanced query interface
            st.markdown("### 💬 Ask the AI Agent")
            
            # Predefined query templates
            query_templates = {
                "biophysics": [
                    "What proteins show the best stability-binding balance?",
                    "How does molecular weight correlate with functionality?",
                    "Which proteins are optimal for drug delivery systems?"
                ],
                "nanotech": [
                    "What size range offers the best conductivity?",
                    "How can we optimize synthesis yield?",
                    "Which materials show the most promise for applications?"
                ],
                "ocean": [
                    "What environmental factors indicate ecosystem health?",
                    "How do temperature and salinity affect marine life?",
                    "Which locations show signs of climate change impact?"
                ],
                "physical_ai": [
                    "Which systems are ready for deployment?",
                    "How can we improve energy efficiency?",
                    "What factors limit response time performance?"
                ]
            }
            
            # Template selection
            selected_template = st.selectbox(
                "💡 Choose a query template or write your own:",
                ["Custom Query"] + query_templates.get(domain, []),
                index=0
            )
            
            if selected_template == "Custom Query":
                user_query = st.text_area(
                    "Enter your analysis query:",
                    placeholder=f"Ask about {domain} data patterns, correlations, or optimization strategies...",
                    height=100
                )
            else:
                user_query = st.text_area(
                    "Modify or use this template:",
                    value=selected_template,
                    height=100
                )
            
            # Analysis context
            analysis_context = {
                "complexity": complexity,
                "sample_size": sample_size,
                "confidence_level": confidence_level,
                "domain": domain,
                "timestamp": datetime.now().isoformat()
            }
            
            # Analysis button with enhanced styling
            if st.button("🚀 Analyze with AI", type="primary", use_container_width=True):
                if user_query.strip():
                    with st.spinner("🧠 AI is analyzing your query..."):
                        # Get current data
                        current_data = {
                            "biophysics": generate_biophysics_data(),
                            "nanotech": generate_nanotech_data(),
                            "ocean": generate_ocean_data(),
                            "physical_ai": generate_physical_ai_data()
                        }[domain]
                        
                        # Generate AI analysis
                        analysis_result = openrouter.generate_analysis(
                            user_query, domain, current_data, analysis_context
                        )
                        
                        # Display result with enhanced styling
                        st.markdown(f"""
                        <div class="analysis-box">
                            <h3>🎯 AI Analysis Result</h3>
                            <p>{analysis_result}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add to history
                        st.session_state.analysis_history.append({
                            'id': len(st.session_state.analysis_history) + 1,
                            'timestamp': datetime.now(),
                            'domain': domain,
                            'query': user_query,
                            'result': analysis_result,
                            'context': analysis_context,
                            'data_shape': current_data.shape
                        })
                        
                        # Success message
                        st.success("✅ Analysis completed and saved to history!")
                        
                else:
                    st.warning("⚠️ Please enter a query first!")
        
        with col_ai2:
            st.markdown("### 📊 Analysis Stats")
            
            # Statistics about current analysis
            if st.session_state.analysis_history:
                total_analyses = len(st.session_state.analysis_history)
                domain_counts = {}
                for analysis in st.session_state.analysis_history:
                    domain_counts[analysis['domain']] = domain_counts.get(analysis['domain'], 0) + 1
                
                st.metric("Total Analyses", total_analyses)
                st.metric("Current Domain", domain_counts.get(domain, 0))
                
                # Domain distribution
                if domain_counts:
                    fig_pie = px.pie(
                        values=list(domain_counts.values()),
                        names=list(domain_counts.keys()),
                        title="Analysis Distribution by Domain"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            # Live analysis feature
            if live_analysis and user_query:
                with st.container():
                    st.markdown("### ⚡ Live Analysis Preview")
                    preview_data = {
                        "biophysics": generate_biophysics_data(),
                        "nanotech": generate_nanotech_data(), 
                        "ocean": generate_ocean_data(),
                        "physical_ai": generate_physical_ai_data()
                    }[domain]
                    
                    st.info(f"📈 Analyzing {len(preview_data)} samples in {domain} domain...")
                    
                    # Quick stats
                    numeric_cols = preview_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.write("Key metrics:", preview_data[numeric_cols].mean().to_dict())
    
    with tab3:
        st.subheader("📈 Historical Analysis Trends")
        
        if st.session_state.analysis_history:
            # Analysis history visualization
            history_df = pd.DataFrame([
                {
                    'timestamp': analysis['timestamp'],
                    'domain': analysis['domain'],
                    'query_length': len(analysis['query']),
                    'result_length': len(analysis['result']),
                    'hour': analysis['timestamp'].hour
                }
                for analysis in st.session_state.analysis_history
            ])
            
            col_h1, col_h2 = st.columns(2)
            
            with col_h1:
                # Timeline of analyses
                fig_timeline = px.scatter(history_df, x='timestamp', y='domain',
                                        size='result_length', color='query_length',
                                        title="Analysis Timeline")
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            with col_h2:
                # Analysis by hour
                hourly_counts = history_df.groupby('hour').size().reset_index(name='count')
                fig_hourly = px.bar(hourly_counts, x='hour', y='count',
                                  title="Analysis Activity by Hour")
                st.plotly_chart(fig_hourly, use_container_width=True)
            
            # Recent analysis history
            st.markdown("### 📋 Recent Analysis History")
            
            for i, analysis in enumerate(reversed(st.session_state.analysis_history[-10:])):
                with st.expander(
                    f"🔬 {analysis['domain'].title()} Analysis #{analysis['id']} - "
                    f"{analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}", 
                    expanded=False
                ):
                    col_exp1, col_exp2 = st.columns([2, 1])
                    
                    with col_exp1:
                        st.write("**Query:**", analysis['query'])
                        st.write("**Result:**", analysis['result'])
                    
                    with col_exp2:
                        st.json(analysis['context'])
                        
                        if st.button(f"🔄 Rerun Analysis #{analysis['id']}", key=f"rerun_{analysis['id']}"):
                            # Rerun the analysis
                            st.info("Feature coming soon: Rerun historical analyses!")
            
            # Clear history button
            if st.button("🗑️ Clear Analysis History", type="secondary"):
                st.session_state.analysis_history = []
                st.success("Analysis history cleared!")
                st.rerun()
        
        else:
            st.info("📭 No analysis history yet. Start by running some analyses in the AI Insights tab!")
    
    # Footer with enhanced information
    st.markdown("---")
    st.markdown("### 🌟 System Information")
    
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        st.markdown("""
        <div class="domain-card">
            <h4>🧬 Bio-physics</h4>
            <p>Molecular dynamics, protein analysis, biomaterial optimization</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_f2:
        st.markdown("""
        <div class="domain-card">
            <h4>⚛️ Nano-technology</h4>
            <p>Material synthesis, property analysis, application development</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_f3:
        st.markdown("""
        <div class="domain-card">
            <h4>🌊 Ocean Science</h4>
            <p>Environmental monitoring, ecosystem analysis, climate research</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_f4:
        st.markdown("""
        <div class="domain-card">
            <h4>🤖 Physical AI</h4>
            <p>Robotics, adaptive systems, intelligent materials</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Version and status
    st.markdown("---")
    col_status1, col_status2, col_status3 = st.columns(3)
    
    with col_status1:
        st.info(f"🔄 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with col_status2:
        st.info(f"📊 Current domain: {domain.upper()}")
    with col_status3:
        st.info(f"🤖 AI analyses: {len(st.session_state.analysis_history)}")

if __name__ == "__main__":
    main()
