import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import time
import hashlib
import base64

# Page configuration
st.set_page_config(
    page_title="Multidisciplinary Analysis Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.3rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
    }
    
    .analysis-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.2);
    }
    
    .ai-response {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.8rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .tab-container {
        margin-top: 1rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-green { background-color: #10b981; }
    .status-yellow { background-color: #f59e0b; }
    
    .chat-container {
        height: 400px;
        overflow-y: auto;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .chat-message {
        margin-bottom: 1rem;
        padding: 0.8rem;
        border-radius: 12px;
        max-width: 80%;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
    }
    
    .ai-message {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_domain' not in st.session_state:
    st.session_state.current_domain = "biophysics"

# Simple AI Client with fallback responses
class SimpleAIClient:
    def __init__(self):
        self.demo_mode = True
    
    def generate_analysis(self, query, domain, data_summary=None):
        # Simple domain-specific responses
        responses = {
            'biophysics': f"🧬 **Bio-physics Analysis**\n\nBased on your query about '{query}', I've analyzed the protein data patterns. Key findings show interesting correlations between molecular weight and stability scores. The data suggests potential optimization opportunities for therapeutic applications. Consider investigating the high-binding-affinity proteins for drug development.",
            
            'nanotech': f"⚛️ **Nano-technology Analysis**\n\nYour query about '{query}' reveals important insights about material properties. The size-conductivity relationship shows quantum effects at nanoscale. Optimization opportunities exist in the 10-50nm range where surface area to volume ratios maximize catalytic potential.",
            
            'ocean': f"🌊 **Ocean Science Analysis**\n\nRegarding your query on '{query}', the oceanographic data indicates generally healthy ecosystem conditions. Temperature-salinity relationships follow expected patterns, though some regions show slight warming trends that warrant long-term monitoring for climate change impacts.",
            
            'physical_ai': f"🤖 **Physical AI Analysis**\n\nFor your query about '{query}', the system performance data shows excellent efficiency-accuracy tradeoffs. Response times are within acceptable ranges for real-time applications, with opportunities for power consumption optimization in high-complexity systems."
        }
        
        return responses.get(domain, f"🔬 **Analysis Complete**\n\nI've analyzed your query about '{query}' in the {domain} domain. The data shows interesting patterns that warrant further investigation for optimization opportunities.")

# Data generation functions
@st.cache_data(ttl=600)
def generate_biophysics_data():
    proteins = ['Hemoglobin', 'Myosin', 'Actin', 'Collagen', 'Elastin', 'Keratin', 'Albumin', 'Fibrinogen']
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

@st.cache_data(ttl=600)
def generate_nanotech_data():
    materials = ['Carbon Nanotubes', 'Graphene', 'Quantum Dots', 'Gold NPs', 'Silver NWs', 'TiO2 NPs']
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

@st.cache_data(ttl=600)
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

@st.cache_data(ttl=600)  
def generate_physical_ai_data():
    systems = ['Marine Robot', 'Nano Assembler', 'Bio Sensor', 'Smart Material', 'Actuator']
    data = []
    for system in systems:
        data.append({
            'system': system,
            'efficiency': np.random.uniform(0.75, 0.98),
            'response_ms': np.random.uniform(0.1, 50),
            'accuracy': np.random.uniform(0.85, 0.99),
            'power_w': np.random.uniform(0.1, 100)
        })
    return pd.DataFrame(data)

# Initialize AI client
ai_client = SimpleAIClient()

# Main app header
st.markdown('<h1 class="main-header">🔬 Multidisciplinary Analysis Agent</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Domain selection
    domain = st.selectbox(
        "🔬 Analysis Domain:",
        ["biophysics", "nanotech", "ocean", "physical_ai"],
        format_func=lambda x: {
            "biophysics": "🧬 Bio-physics",
            "nanotech": "⚛️ Nano-technology", 
            "ocean": "🌊 Ocean Science",
            "physical_ai": "🤖 Physical AI"
        }[x]
    )
    
    # Update current domain in session state
    st.session_state.current_domain = domain
    
    # Parameters
    st.subheader("Parameters")
    complexity = st.slider("Analysis Complexity", 1, 10, 7)
    enhanced_mode = st.checkbox("Enhanced Data Mode", value=True)
    
    # Status
    st.markdown("---")
    st.markdown(f"""
    <div>
        <span class="status-indicator status-green"></span>
        AI Status: Active
    </div>
    """, unsafe_allow_html=True)
    
    st.metric("Analyses Run", st.session_state.analysis_count)

# Generate data based on domain
if domain == "biophysics":
    data = generate_biophysics_data()
    domain_title = "🧬 Bio-physics Research"
    metrics = [
        ("Avg Molecular Weight", f"{data['molecular_weight'].mean():.0f} Da"),
        ("Avg Stability", f"{data['stability_score'].mean():.2f}"),
        ("Best Binding", f"{data['binding_affinity'].min():.2e} M")
    ]
elif domain == "nanotech":
    data = generate_nanotech_data()
    domain_title = "⚛️ Nano-technology"
    metrics = [
        ("Avg Size", f"{data['size_nm'].mean():.1f} nm"),
        ("Max Conductivity", f"{data['conductivity'].max():.0f} S/m"),
        ("Avg Surface Area", f"{data['surface_area'].mean():.0f} m²/g")
    ]
elif domain == "ocean":
    data = generate_ocean_data()
    domain_title = "🌊 Ocean Science"
    metrics = [
        ("Avg Temperature", f"{data['temperature'].mean():.1f}°C"),
        ("Avg Salinity", f"{data['salinity'].mean():.1f} ppt"),
        ("Avg pH", f"{data['ph_level'].mean():.2f}")
    ]
else:  # physical_ai
    data = generate_physical_ai_data()
    domain_title = "🤖 Physical AI"
    metrics = [
        ("Avg Efficiency", f"{data['efficiency'].mean():.1%}"),
        ("Avg Response", f"{data['response_ms'].mean():.1f} ms"),
        ("Avg Accuracy", f"{data['accuracy'].mean():.1%}")
    ]

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Analysis", "🤖 AI Insights", "📈 Advanced Analytics", "🔬 Research Tools"])

# Tab 1: Data Analysis
with tab1:
    st.subheader(domain_title)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    for i, (title, value) in enumerate(metrics):
        with [col1, col2, col3][i]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{title}</h4>
                <h2>{value}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Create visualizations
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        if domain == "biophysics":
            fig = px.scatter(
                data, x='molecular_weight', y='stability_score',
                size='binding_affinity', hover_name='protein',
                title='Protein Properties'
            )
        elif domain == "nanotech":
            fig = px.scatter(
                data, x='size_nm', y='conductivity',
                size='surface_area', hover_name='material',
                title='Size vs Conductivity', log_y=True
            )
        elif domain == "ocean":
            fig = px.scatter(
                data, x='temperature', y='salinity',
                size='pressure', color='ph_level',
                hover_name='location', title='Ocean Parameters'
            )
        else:  # physical_ai
            fig = px.scatter(
                data, x='response_ms', y='efficiency',
                size='power_w', color='accuracy',
                hover_name='system', title='System Performance'
            )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_v2:
        if domain == "biophysics":
            fig = px.bar(data, x='protein', y='binding_affinity', 
                        title='Binding Affinity', log_y=True)
        elif domain == "nanotech":
            fig = px.bar(data, x='material', y='yield_pct',
                        title='Synthesis Yield')
        elif domain == "ocean":
            fig = px.bar(data, x='location', y='oxygen',
                        title='Oxygen Levels')
        else:  # physical_ai
            fig = px.bar(data, x='system', y='accuracy',
                        title='System Accuracy')
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Data explorer
    with st.expander("📊 Data Explorer", expanded=False):
        st.dataframe(data, use_container_width=True)

# Tab 2: AI Insights
with tab2:
    st.subheader("🤖 AI Analysis Assistant")
    
    # Query input
    query = st.text_area(
        "Enter your research question:",
        placeholder=f"Ask about {domain} data patterns, correlations, or optimization strategies...",
        height=100
    )
    
    # Analysis button
    if st.button("🚀 Analyze with AI", type="primary"):
        if query.strip():
            with st.spinner("AI is analyzing your query..."):
                # Update analysis count
                st.session_state.analysis_count += 1
                
                # Generate analysis
                result = ai_client.generate_analysis(query, domain)
                
                # Display result
                st.markdown(f"""
                <div class="analysis-box">
                    <div class="ai-response">
                        {result}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add to history
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now(),
                    'domain': domain,
                    'query': query,
                    'result': result
                })
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': query,
                    'timestamp': datetime.now()
                })
                
                st.session_state.chat_history.append({
                    'role': 'ai',
                    'content': result,
                    'timestamp': datetime.now()
                })
        else:
            st.warning("Please enter a research question first.")
    
    # Analysis history
    if st.session_state.analysis_history:
        st.subheader("📝 Recent Analysis")
        for analysis in st.session_state.analysis_history[-3:]:
            with st.expander(f"{analysis['domain'].title()} - {analysis['timestamp'].strftime('%H:%M')}"):
                st.write(f"**Query:** {analysis['query']}")
                st.markdown("---")
                st.write(analysis['result'])

# Tab 3: Advanced Analytics
with tab3:
    st.subheader("📈 Advanced Analytics")
    
    # Domain-specific insights
    st.markdown("### 💡 Key Insights")
    
    if domain == "biophysics":
        insights = [
            "Proteins with higher molecular weights show increased stability",
            "Strong correlation between stability and binding affinity",
            "Optimal molecular weight range for therapeutic applications: 50-80 kDa"
        ]
    elif domain == "nanotech":
        insights = [
            "Quantum effects observed in materials below 20nm",
            "Conductivity peaks in the 10-50nm size range",
            "Surface area optimization critical for catalytic applications"
        ]
    elif domain == "ocean":
        insights = [
            "Temperature-salinity relationships follow expected patterns",
            "pH levels within healthy range for marine ecosystems",
            "Oxygen levels adequate for diverse marine life"
        ]
    else:  # physical_ai
        insights = [
            "Efficiency-accuracy trade-off well balanced across systems",
            "Response times suitable for real-time applications",
            "Power consumption optimization opportunities in high-complexity systems"
        ]
    
    for insight in insights:
        st.info(insight)
    
    # Advanced visualization
    st.markdown("### 🔍 Advanced Visualization")
    
    if domain == "biophysics":
        fig = px.scatter_3d(
            data, x='molecular_weight', y='stability_score', z='binding_affinity',
            color='protein', title='3D Protein Analysis'
        )
    elif domain == "nanotech":
        fig = px.parallel_coordinates(
            data, dimensions=['size_nm', 'conductivity', 'surface_area', 'yield_pct'],
            color='material', title='Material Properties'
        )
    elif domain == "ocean":
        fig = px.scatter_matrix(
            data, dimensions=['temperature', 'salinity', 'ph_level', 'oxygen'],
            color='location', title='Ocean Parameter Relationships'
        )
    else:  # physical_ai
        fig = px.scatter(
            data, x='efficiency', y='accuracy',
            size='power_w', color='response_ms',
            hover_name='system', title='System Performance Matrix'
        )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# Tab 4: Research Tools
with tab4:
    st.subheader("🔬 Research Tools")
    
    # Chat interface
    st.markdown("### 💬 AI Research Assistant")
    
    # Chat container
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history[-10:]:  # Show last 10 messages
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message ai-message">
                    <strong>AI:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        chat_input = st.text_input(
            "Ask a question:",
            key="chat_input",
            placeholder="Ask about data analysis or research methods..."
        )
    
    with col2:
        if st.button("Send"):
            if chat_input.strip():
                # Add user message
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': chat_input,
                    'timestamp': datetime.now()
                })
                
                # Get AI response
                with st.spinner("AI is thinking..."):
                    response = ai_client.generate_analysis(chat_input, domain)
                
                # Add AI response
                st.session_state.chat_history.append({
                    'role': 'ai',
                    'content': response,
                    'timestamp': datetime.now()
                })
                
                # Clear input
                st.session_state.chat_input = ""
                st.rerun()
    
    # Research tools
    st.markdown("### 🛠️ Analysis Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("Statistical Analysis"):
            if st.button("Generate Statistics"):
                st.write("**Descriptive Statistics:**")
                st.dataframe(data.describe(), use_container_width=True)
    
    with col2:
        with st.expander("Data Export"):
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                csv = data.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name=f"{domain}_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col_exp2:
                json_data = data.to_json(orient='records')
                st.download_button(
                    "Download JSON",
                    data=json_data,
                    file_name=f"{domain}_data_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

# Custom visualization tool
    with st.expander("Custom Visualization"):
        if len(data.columns) > 1:
            x_col = st.selectbox("X-axis:", data.columns)
            y_col = st.selectbox("Y-axis:", data.columns)
            
            chart_type = st.selectbox("Chart Type:", ["Scatter", "Line", "Bar"])
            
            if st.button("Generate Chart"):
                if chart_type == "Scatter":
                    fig = px.scatter(data, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                elif chart_type == "Line":
                    fig = px.line(data, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                else:  # Bar
                    fig = px.bar(data, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🔬 Multidisciplinary Analysis Agent | AI-Powered Research Platform"
    "</div>",
    unsafe_allow_html=True
)
