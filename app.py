import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Crop Recommendation Engine",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Load Cached Models & Encoders
# ------------------------------
@st.cache_resource
def load_assets():
    """Loads all necessary pickled models and encoders with error handling."""
    try:
        model = pickle.load(open("xgb_crop_model.pkl", "rb"))
        label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
        feature_label_encoders = pickle.load(open("feature_label_encoders.pkl", "rb"))
        feature_names = pickle.load(open("feature_names.pkl", "rb"))
        return model, label_encoder, feature_label_encoders, feature_names
    except FileNotFoundError:
        st.error("Essential model files are missing. Please ensure all .pkl files are in the application's root directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading model assets: {e}")
        st.stop()

model, label_encoder, feature_label_encoders, feature_names = load_assets()

# ------------------------------
# Custom CSS for Professional UI
# ------------------------------
def apply_custom_styling():
    st.markdown("""
        <style>
            /* Main app background */
            .stApp {
                background-color: #f8f9fa;
            }
            
            /* Header styling */
            .main-header {
                text-align: center;
                color: #2c3e50;
                font-weight: 700;
                margin-bottom: 0.5rem;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid #4ECDC4;
            }
            
            /* Subheader styling */
            .subheader {
                text-align: center;
                color: #7f8c8d;
                font-weight: 400;
                margin-bottom: 2rem;
            }
            
            /* Card styling */
            .card {
                background-color: white;
                border-radius: 10px;
                padding: 1.5rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                margin-bottom: 1.5rem;
                border-left: 4px solid #4ECDC4;
            }
            
            /* Metric styling */
            .metric-card {
                background-color: white;
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                text-align: center;
                border-top: 4px solid #4ECDC4;
            }
            
            /* Button styling */
            .stButton>button {
                background-color: #4ECDC4;
                color: white;
                font-weight: 600;
                padding: 0.7rem 1.5rem;
                border-radius: 6px;
                border: none;
                width: 100%;
                transition: all 0.2s ease;
            }
            
            .stButton>button:hover {
                background-color: #2a9088;
                transform: translateY(-1px);
            }
            
            /* Slider styling */
            .stSlider {
                margin-bottom: 1rem;
            }
            
            /* Info box styling */
            .info-box {
                background-color: #e8f4f3;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
                border-left: 4px solid #4ECDC4;
            }
            
            /* Footer styling */
            .footer {
                text-align: center;
                margin-top: 3rem;
                padding-top: 1rem;
                color: #7f8c8d;
                font-size: 0.85rem;
            }
        </style>
    """, unsafe_allow_html=True)

apply_custom_styling()

# ------------------------------
# Sidebar with Information
# ------------------------------
with st.sidebar:
    st.markdown("<h2 style='color: #2c3e50;'>ℹ️ About</h2>", unsafe_allow_html=True)
    st.markdown("""
    This intelligent system uses machine learning to recommend the most suitable crops based on your soil's chemical properties.
    """)
    
    st.markdown("<h3 style='color: #2c3e50;'>📊 Ideal Soil Parameters</h3>", unsafe_allow_html=True)
    st.markdown("""
    - **pH**: 6.0-7.5
    - **Nitrogen (N)**: 20-50 mg/kg
    - **Phosphorus (P)**: 15-30 mg/kg  
    - **Potassium (K)**: 150-250 mg/kg
    - **Zinc (Zn)**: 0.5-2.0 mg/kg
    - **Sulphur (S)**: 10-20 mg/kg
    """)
    
    st.markdown("<h3 style='color: #2c3e50;'>🌾 Crop Database</h3>", unsafe_allow_html=True)
    st.write(f"Our system can recommend from {len(label_encoder.classes_)} different crops")

# ------------------------------
# Application Title and Description
# ------------------------------
st.markdown("<h1 class='main-header'>🌿 Smart Crop Recommendation Engine</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Enter your soil's chemical properties to receive an intelligent crop recommendation</p>", unsafe_allow_html=True)

# ------------------------------
# Input Form
# ------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("<h2 style='color: #2c3e50;'>📋 Soil Analysis Inputs</h2>", unsafe_allow_html=True)

with st.form(key="crop_input_form"):
    # Create three columns for a balanced layout
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("<h3 style='color: #2c3e50; font-size: 1.1rem;'>Primary Nutrients</h3>", unsafe_allow_html=True)
        n = st.slider("🌿 Nitrogen (N) Content", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
        p = st.slider("🌱 Phosphorus (P) Content", min_value=0.0, max_value=50.0, value=15.0, step=1.0)
        k = st.slider("🥔 Potassium (K) Content", min_value=0.0, max_value=300.0, value=150.0, step=1.0)
    
    with col2:
        st.markdown("<h3 style='color: #2c3e50; font-size: 1.1rem;'>Secondary Nutrients & pH</h3>", unsafe_allow_html=True)
        ph = st.slider("⚗️ Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        zn = st.slider("🔬 Zinc (Zn) Content", min_value=0.0, max_value=5.0, value=0.8, step=0.1)
        s = st.slider("🧪 Sulphur (S) Content", min_value=0.0, max_value=30.0, value=12.0, step=0.1)
    
    with col3:
        st.markdown("<h3 style='color: #2c3e50; font-size: 1.1rem;'>Soil Characteristics</h3>", unsafe_allow_html=True)
        soiltype = st.selectbox("🟤 Soil Type", options=list(feature_label_encoders["Soiltype"].classes_))
        
        # Ideal ranges info
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("<h4 style='color: #2c3e50; margin: 0; font-size: 1rem;'>Ideal Ranges</h4>", unsafe_allow_html=True)
        st.markdown("""
        - pH: 6.0-7.5
        - N: 20-50
        - P: 15-30
        - K: 150-250
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Submit button for the form
    submit_button = st.form_submit_button(label="🌱 Recommend Crop")

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# Prediction and Output Display
# ------------------------------
if submit_button:
    with st.spinner('Analyzing soil data and generating recommendations...'):
        # Prepare the feature vector for prediction
        sample = {"Soiltype": soiltype, "Ph": ph, "K": k, "P": p, "N": n, "Zn": zn, "S": s}
        
        encoded_features = []
        for col in feature_names:
            val = sample[col]
            if col in feature_label_encoders:
                le = feature_label_encoders[col]
                try:
                    val = le.transform([val])[0]
                except ValueError:
                    st.error(f"Invalid value '{val}' for Soil Type. Please select a valid option.")
                    st.stop()
            encoded_features.append(val)
        
        # Convert to a 2D NumPy array for the model
        X = np.array(encoded_features).reshape(1, -1)

        # Get prediction probabilities and top 5 results
        pred_proba = model.predict_proba(X)
        top_5_indices = np.argsort(pred_proba[0])[-5:][::-1]
        top_5_crops = label_encoder.inverse_transform(top_5_indices)
        top_5_probs = pred_proba[0][top_5_indices]

    # --- Display Results ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h2 style='color: #2c3e50;'>💡 Recommendation Results</h2>", unsafe_allow_html=True)
    
    # Create columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Top recommendation
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: #2c3e50; text-align: center;'>Top Recommendation</h3>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: #4ECDC4; text-align: center;'>{top_5_crops[0]}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Confidence: {top_5_probs[0]*100:.1f}%</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Other recommendations
        st.markdown("<h3 style='color: #2c3e50;'>Other Suitable Options</h3>", unsafe_allow_html=True)
        for i in range(1, len(top_5_crops)):
            st.markdown(f"**{i+1}. {top_5_crops[i]}** - {top_5_probs[i]*100:.1f}% suitability")
        
        # Display growing tips based on the recommended crop
        st.markdown("<h3 style='color: #2c3e50; margin-top: 1.5rem;'>🌱 Growing Tips</h3>", unsafe_allow_html=True)
        if "rice" in top_5_crops[0].lower():
            st.info("Rice grows best in flooded fields with high nitrogen levels. Maintain proper water management for optimal yield.")
        elif "wheat" in top_5_crops[0].lower():
            st.info("Wheat requires well-drained soil with moderate nitrogen. Rotate crops to prevent disease buildup.")
        elif "corn" in top_5_crops[0].lower() or "maize" in top_5_crops[0].lower():
            st.info("Corn needs plenty of nitrogen and consistent moisture. Plant in blocks for better pollination.")
        else:
            st.info("Ensure proper irrigation and regular soil testing for optimal growth conditions.")

    with col2:
        # Nutrient analysis chart
        st.markdown("<h3 style='color: #2c3e50;'>Nutrient Analysis</h3>", unsafe_allow_html=True)
        
        # Create a bar chart for nutrient levels
        nutrients = ['Nitrogen', 'Phosphorus', 'Potassium', 'Zinc', 'Sulphur']
        values = [n, p, k, zn, s]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=nutrients,
            y=values,
            marker_color='#4ECDC4',
            opacity=0.8
        ))
        
        fig.update_layout(
            height=300,
            showlegend=False,
            yaxis_title="Value (mg/kg)",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Nutrient status indicators
        st.markdown("<h3 style='color: #2c3e50;'>Nutrient Status</h3>", unsafe_allow_html=True)
        
        # Define ideal ranges
        ideal_ranges = {
            'N': (20, 50),
            'P': (15, 30),
            'K': (150, 250),
            'Zn': (0.5, 2.0),
            'S': (10, 20),
            'pH': (6.0, 7.5)
        }
        
        # Create a small table with status indicators
        nutrient_data = {
            'Nutrient': ['N', 'P', 'K', 'Zn', 'S', 'pH'],
            'Value': [n, p, k, zn, s, ph],
            'Status': ['✅ Optimal' if ideal_ranges[nut][0] <= val <= ideal_ranges[nut][1] 
                      else '⚠️ Suboptimal' for nut, val in zip(['N', 'P', 'K', 'Zn', 'S', 'pH'], [n, p, k, zn, s, ph])]
        }
        
        nutrient_df = pd.DataFrame(nutrient_data)
        st.dataframe(nutrient_df, hide_index=True, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a section for saving results
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3 style='color: #2c3e50;'>💾 Save Your Results</h3>", unsafe_allow_html=True)
    
    # Create a dataframe with the input values and recommendations
    results_df = pd.DataFrame({
        'Parameter': ['Soil Type', 'pH', 'Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 'Zinc (Zn)', 'Sulphur (S)'],
        'Value': [soiltype, ph, n, p, k, zn, s],
        'Recommended Crop': [top_5_crops[0], '', '', '', '', '', '']
    })
    
    # Display the results table
    st.dataframe(results_df, hide_index=True, use_container_width=True)
    
    # Add download button
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="crop_recommendation_results.csv",
        mime="text/csv",
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
else:
    # Initial state message before prediction
    st.info("ℹ️ Enter your soil parameters above and click 'Recommend Crop' to get started.")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("© 2023 Smart Crop Recommendation System | For agricultural decision support")
st.markdown("</div>", unsafe_allow_html=True)