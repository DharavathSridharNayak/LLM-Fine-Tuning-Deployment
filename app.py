import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import time
from PIL import Image
import os
import json
import tempfile
from huggingface_hub import notebook_login, HfApi, Repository

# Set page config
st.set_page_config(
    page_title="LLM Fine-Tuning & Deployment",
    page_icon=":robot:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
        .stApp {
            background-color: #000000;
            color: #ffffff;
        }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .stSelectbox>div>div>select {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .stSlider>div>div>div>div {
            background-color: #4CAF50;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .css-1aumxhk {
            background-color: #121212;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            color: #ffffff;
        }
        .header {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .subheader {
            color: #4CAF50;
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }
        .st-bb {
            background-color: transparent;
        }
        .st-at {
            background-color: #1a1a1a;
        }
        .st-bh {
            color: #ffffff;
        }
        .st-ag {
            font-size: 1rem;
            color: #ffffff;
        }
        .st-ae {
            background-color: #1a1a1a;
        }
        .st-af {
            color: #ffffff;
        }
        .stProgress>div>div>div>div {
            background-color: #4CAF50;
        }
        div[data-baseweb="select"]>div {
            background-color: #1a1a1a;
            color: white;
        }
        .st-b7 {
            background-color: transparent;
        }
        .st-b8 {
            color: #ffffff;
        }
        .st-b9 {
            background-color: #1a1a1a;
        }
        .st-ba {
            color: #ffffff;
        }
        .stRadio>div {
            color: #ffffff;
        }
        .stCheckbox>div>label>div {
            color: #ffffff;
        }
        .stFileUploader>div>div>div>div {
            color: #ffffff;
        }
        .stMarkdown {
            color: #ffffff;
        }
        .stAlert {
            background-color: #1a1a1a;
        }
    </style>
""", unsafe_allow_html=True)

# App header
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=100)
with col2:
    st.markdown('<div class="header">LLM Fine-Tuning & Deployment</div>', unsafe_allow_html=True)
    st.markdown("Fine-tune and deploy your large language models with ease")

# Navigation menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Data Preparation", "Model Selection", "Fine-Tuning", "Evaluation", "Deployment", "About"],
        icons=["house", "file-earmark-text", "cpu", "gear", "graph-up", "cloud-upload", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#121212"},
            "icon": {"color": "#4CAF50", "font-size": "18px"},
            "nav-link": {"color": "#ffffff", "font-size": "16px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#4CAF50", "color": "#ffffff"},
        }
    )

# Home Page
if selected == "Home":
    st.markdown('<div class="subheader">Welcome to LLM Fine-Tuning & Deployment</div>', unsafe_allow_html=True)
    st.markdown("""
        This application guides you through the process of fine-tuning large language models (LLMs) 
        and deploying them to Hugging Face Hub.
        
        **Key Features:**
        - Prepare your dataset for fine-tuning
        - Select from popular base models
        - Configure fine-tuning parameters
        - Evaluate model performance
        - Deploy to Hugging Face Hub
        
        Get started by selecting a step from the sidebar menu.
    """)
    
    st.image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hf-libraries.png", 
             caption="Hugging Face Ecosystem", use_column_width=True)

# Data Preparation Page
elif selected == "Data Preparation":
    st.markdown('<div class="subheader">Data Preparation</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Upload Data", "Preview Data", "Data Statistics"])
    
    with tab1:
        st.markdown("### Upload Your Dataset")
        data_file = st.file_uploader("Choose a file (CSV, JSON, or TXT)", type=["csv", "json", "txt"])
        
        if data_file is not None:
            file_details = {"FileName": data_file.name, "FileType": data_file.type, "FileSize": data_file.size}
            st.success("File uploaded successfully!")
            st.json(file_details)
            
            # Save uploaded file to temporary location
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, data_file.name)
            with open(path, "wb") as f:
                f.write(data_file.getbuffer())
            
            st.session_state['data_path'] = path
            st.session_state['data_type'] = data_file.type
    
    with tab2:
        if 'data_path' in st.session_state:
            st.markdown("### Data Preview")
            
            if st.session_state['data_type'] == "text/csv":
                df = pd.read_csv(st.session_state['data_path'])
                st.dataframe(df.head().style.set_properties(**{'background-color': '#1a1a1a', 'color': 'white'}))
            elif st.session_state['data_type'] == "application/json":
                with open(st.session_state['data_path']) as f:
                    data = json.load(f)
                st.json(data)
            else:
                with open(st.session_state['data_path']) as f:
                    data = f.read()
                st.text_area("Text Content", data, height=200)
        else:
            st.warning("Please upload a file first.")
    
    with tab3:
        if 'data_path' in st.session_state:
            st.markdown("### Data Statistics")
            
            if st.session_state['data_type'] == "text/csv":
                df = pd.read_csv(st.session_state['data_path'])
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Samples", len(df))
                col2.metric("Columns", len(df.columns))
                col3.metric("Missing Values", df.isnull().sum().sum())
                
                st.markdown("**Column Types**")
                st.table(df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Type"}).style.set_properties(**{'background-color': '#1a1a1a', 'color': 'white'}))
            else:
                st.info("Detailed statistics available for CSV files only.")

# Model Selection Page
elif selected == "Model Selection":
    st.markdown('<div class="subheader">Model Selection</div>', unsafe_allow_html=True)
    
    model_options = {
        "GPT-like": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        "BERT-like": ["bert-base-uncased", "bert-large-uncased"],
        "RoBERTa": ["roberta-base", "roberta-large"],
        "T5": ["t5-small", "t5-base", "t5-large"],
        "Custom": ["Enter custom model name"]
    }
    
    model_family = st.selectbox("Select Model Family", list(model_options.keys()))
    
    if model_family == "Custom":
        model_name = st.text_input("Enter Hugging Face Model ID")
    else:
        model_name = st.selectbox("Select Model", model_options[model_family])
    
    st.markdown("### Model Information")
    
    if model_name:
        st.info(f"You've selected: **{model_name}**")
        
        # Display model card
        st.markdown(f"View model card on [Hugging Face Hub](https://huggingface.co/{model_name})")
        
        # Show estimated resource requirements
        st.markdown("**Estimated Resource Requirements**")
        
        if "gpt2" in model_name or "large" in model_name:
            st.warning("This model requires significant GPU memory (8GB+ recommended)")
        else:
            st.success("This model can run on modest hardware (4GB GPU memory sufficient for fine-tuning)")
        
        st.session_state['selected_model'] = model_name

# Fine-Tuning Page
elif selected == "Fine-Tuning":
    st.markdown('<div class="subheader">Fine-Tuning Configuration</div>', unsafe_allow_html=True)
    
    if 'selected_model' not in st.session_state:
        st.warning("Please select a model first from the Model Selection page.")
        st.stop()
    
    st.info(f"Fine-tuning model: **{st.session_state['selected_model']}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Training Parameters")
        epochs = st.slider("Number of Epochs", 1, 20, 3)
        batch_size = st.selectbox("Batch Size", [4, 8, 16, 32, 64], index=2)
        learning_rate = st.selectbox("Learning Rate", [1e-5, 3e-5, 5e-5, 1e-4], index=1)
        
    with col2:
        st.markdown("### Advanced Options")
        warmup_steps = st.number_input("Warmup Steps", 0, 1000, 100)
        weight_decay = st.slider("Weight Decay", 0.0, 0.1, 0.01)
        fp16 = st.checkbox("Use Mixed Precision (FP16)", value=True)
    
    st.markdown("### Start Fine-Tuning")
    
    if st.button("Begin Fine-Tuning Process"):
        if 'data_path' not in st.session_state:
            st.error("Please upload your dataset first.")
        else:
            with st.spinner("Setting up fine-tuning environment..."):
                time.sleep(2)
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(1, 101):
                progress_bar.progress(i)
                status_text.text(f"Training progress: {i}%")
                time.sleep(0.05)
                
            st.success("Fine-tuning completed successfully!")
            st.balloons()
            
            st.session_state['fine_tuned'] = True
            st.session_state['model_path'] = f"./models/{st.session_state['selected_model']}-fine-tuned"

# Evaluation Page
elif selected == "Evaluation":
    st.markdown('<div class="subheader">Model Evaluation</div>', unsafe_allow_html=True)
    
    if 'fine_tuned' not in st.session_state:
        st.warning("Please complete the fine-tuning process first.")
        st.stop()
    
    st.success(f"Evaluating fine-tuned model: **{st.session_state['selected_model']}**")
    
    st.markdown("### Evaluation Metrics")
    
    # Simulated metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Training Loss", "0.456", "-0.124 from baseline")
    col2.metric("Validation Loss", "0.512", "-0.098 from baseline")
    col3.metric("Accuracy", "0.872", "+0.15 from baseline")
    
    st.markdown("### Sample Predictions")
    
    sample_text = st.text_area("Enter text to test the model", "The movie was...")
    
    if st.button("Generate Prediction"):
        with st.spinner("Generating response..."):
            time.sleep(2)
            
            # Simulate different responses
            responses = {
                "positive": "The movie was absolutely fantastic! The acting was superb and the storyline kept me engaged throughout.",
                "negative": "The movie was terrible. Poor acting and a predictable plot made it a complete waste of time.",
                "neutral": "The movie was okay. It had some good moments but nothing particularly memorable."
            }
            
            selected_response = np.random.choice(list(responses.values()))
            
            st.markdown("**Model Output:**")
            st.info(selected_response)

# Deployment Page
elif selected == "Deployment":
    st.markdown('<div class="subheader">Model Deployment</div>', unsafe_allow_html=True)
    
    if 'fine_tuned' not in st.session_state:
        st.warning("Please complete the fine-tuning process first.")
        st.stop()
    
    st.info(f"Ready to deploy: **{st.session_state['selected_model']}-fine-tuned**")
    
    st.markdown("### Hugging Face Hub Deployment")
    
    hf_token = st.text_input("Hugging Face Access Token", type="password")
    repo_name = st.text_input("Repository Name", "my-fine-tuned-model")
    privacy = st.radio("Repository Visibility", ["Public", "Private"])
    
    if st.button("Deploy to Hugging Face Hub"):
        if not hf_token:
            st.error("Please provide your Hugging Face access token")
        else:
            with st.spinner("Uploading model to Hugging Face Hub..."):
                time.sleep(3)
                
                st.success(f"Model successfully deployed to Hugging Face Hub!")
                st.markdown(f"Your model is available at: [https://huggingface.co/{repo_name}](https://huggingface.co/{repo_name})")
                
                st.session_state['deployed'] = True

# About Page
elif selected == "About":
    st.markdown('<div class="subheader">About This App</div>', unsafe_allow_html=True)
    
    st.markdown("""
        **LLM Fine-Tuning & Deployment App**
        
        This application provides an intuitive interface for fine-tuning large language models 
        and deploying them to Hugging Face Hub.
        
        **Features:**
        - Streamlined workflow for LLM fine-tuning
        - Support for various model architectures
        - Easy deployment to Hugging Face Hub
        - Beautiful and responsive UI
        
        **Technologies Used:**
        - Streamlit for the web interface
        - Hugging Face Transformers for model handling
        - Hugging Face Hub for model deployment
        
        Developed with ❤️ for the AI community.
    """)
    
    st.markdown("---")
    st.markdown("""
        **Disclaimer:** This is a demo application. For production use, 
        please ensure you have proper hardware resources and follow best practices 
        for model training and deployment.
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #4CAF50; font-size: 0.9rem;">
        LLM Fine-Tuning & Deployment App | Powered by Streamlit and Hugging Face
    </div>
""", unsafe_allow_html=True)
