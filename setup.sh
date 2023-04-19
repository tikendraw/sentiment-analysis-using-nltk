#! bin/bash

# installing nltk addons
python3  nltk_essential.py

# setting up streamlit
mkdir -p ~/.streamlit/ 
    echo "\ [server]\n\
    headless = true\n\
    port = $PORT\n\
    enableCORS = false\n\
    \n\" > ~/.streamlit/config.toml
