from ruamel.yaml import YAML
from pathlib import Path
import streamlit as st


@st.cache_data(persist=True)
def get_env_config():
    path = Path("./src/env.yaml")
    yaml = YAML(typ='safe')
    data = yaml.load(path)
    if 'env_config' not in st.session_state:    
        st.session_state['env_config'] = {}
    else:
        st.write("Loading environment configuration...")
        st.session_state['env_config'] = data






