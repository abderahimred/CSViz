import streamlit as st
import sys
import os
import re
import time
from random import randint
from collections import Counter

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Data Visualization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # Keep sidebar collapsed by default
)

# Set flag to prevent app.py from rendering its UI when imported
os.environ["IMPORTING_ONLY"] = "1"

# Now import other modules that might use Streamlit
from st_aggrid import AgGrid, GridOptionsBuilder
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json
from io import BytesIO
from utils.theme import apply_custom_theme, feature_card
# Import from app.py after setting the environment variable
from app import (
    calculate_column_score,
    calculate_pair_score,
    calculate_triple_score,
    get_visualization_recommendation,
    score_all_columns_and_pairs,
    score_groupby_column,
    score_aggregation_column,
    calculate_groupby_pair_score,
    get_groupby_visualization_recommendation,
    visualize_triple,
    convert_to_datetime,
    visualize_groupby
)
from visualization_decision import (
    is_temporal_column,
    get_vis_type_for_single_column,
    get_vis_type_for_pair,
    get_vis_type_for_triple,
    get_vis_type_for_groupby,
    get_vis_type_for_groupby_pair
)
from utils.translations import get_translation_function

# Apply custom theme
apply_custom_theme()

# Get translation function
t = get_translation_function()

# Define domain keywords dictionary
domain_keywords = {
    "Business & Finance": ["revenue", "profit", "sales", "cost", "margin", "product", "inventory", "customer", "price"],
    "Healthcare": ["patient", "disease", "treatment", "diagnosis", "hospital", "doctor", "medical", "health"],
    "Education": ["student", "course", "grade", "teacher", "school", "university", "education", "learning"],
    "Science & Research": ["experiment", "measurement", "observation", "variable", "correlation", "hypothesis", "sample"],
    "Marketing": ["campaign", "audience", "conversion", "engagement", "channel", "ad", "social", "brand"],
    "Human Resources": ["employee", "salary", "hire", "performance", "department", "role", "position"],
    "Transportation": ["vehicle", "distance", "route", "destination", "travel", "transport", "delivery"],
    "E-commerce": ["product", "order", "customer", "cart", "purchase", "shipping", "review", "item"],
    "Social Media": ["user", "post", "engagement", "follower", "comment", "platform", "share", "like"],
    "Real Estate": ["property", "price", "location", "area", "sale", "rent", "agent", "housing"]
}

# Add CSS for better dropdown formatting
st.markdown("""
<style>
/* Language selector styling - only apply to the language selector */
div.language-selector-container div[data-testid="stSelectbox"] {
    max-width: 200px;
    margin-left: auto;
}

/* Ensure text in buttons is white */
.stButton button {
    color: white !important;
}

.stButton button p, 
.stButton button span, 
.stButton button div {
    color: white !important;
}

/* Tab button styling */
button[data-baseweb="tab"] {
    padding: 10px 15px;
}

/* Make all buttons more visible */
.stButton button {
    margin-bottom: 10px;
}

/* Reduce white space at the top of the page */
.block-container {
    padding-top: 1rem !important;
}

/* Make step indicators more compact */
.step-indicator {
    margin-top: 0 !important;
    margin-bottom: 0.5rem !important;
    padding: 0 !important;
    font-size: 1rem !important;
}

/* Reduce space around progress bar */
.stProgress {
    margin-top: 0.5rem !important;
    margin-bottom: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'dashboard_step' not in st.session_state:
    st.session_state.dashboard_step = 1
    
if 'domain' not in st.session_state:
    st.session_state.domain = None
    
if 'top3' not in st.session_state:
    st.session_state.top3 = []
    
if 'viz_recommendations' not in st.session_state:
    st.session_state.viz_recommendations = None

# Add language selector at the top
language_col1, language_col2 = st.columns([6, 1])
with language_col2:
    # Add CSS to ensure language selector stays properly positioned but doesn't affect other selectboxes
    st.markdown("""
    <style>
    /* Fix language selector positioning */
    div.language-selector-container div[data-testid="stSelectbox"] {
        width: 150px;
        float: right;
        max-width: 100%;
        margin-right: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Wrap the language selector in a div with a specific class for targeting
    st.markdown('<div class="language-selector-container">', unsafe_allow_html=True)
    selected_lang = st.selectbox(
        t('language_selector'),
        options=["English", "Français"],
        index=0 if st.session_state.get("language", "en") == "en" else 1,
        key="language_selector",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Update language when user changes selection
    if (selected_lang == "English" and st.session_state.get("language", "en") != "en") or \
       (selected_lang == "Français" and st.session_state.get("language", "en") != "fr"):
        st.session_state.language = "en" if selected_lang == "English" else "fr"
        st.rerun()

# Add custom CSS for button text
st.markdown("""
    <style>
    /* Force white text in all parts of buttons */
    .stButton button {
        color: white !important;
    }
    
    .stButton button p, 
    .stButton button span, 
    .stButton button div {
        color: white !important;
    }
    
    /* Target button text specifically */
    button[kind="primary"] p,
    [data-testid^="stButton"] p,
    .stButton p {
        color: white !important;
    }
    
    /* Aggregation type styling */
    .mean-bg {
        background-color: #e6f3ff; 
        color: #0066cc;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    
    .sum-bg {
        background-color: #fff2e6; 
        color: #cc6600;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    
    .count-bg {
        background-color: #e6ffe6; 
        color: #006600;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    
    /* Arrow navigation styling */
    .arrow-button {
        background-color: #f0f2f6;
        border: 1px solid #dfe1e6;
        border-radius: 4px;
        color: #36454F;
        cursor: pointer;
        font-size: 18px;
        padding: 8px 12px;
        transition: background-color 0.3s;
    }
    
    .arrow-button:hover:not([disabled]) {
        background-color: #e6e9ef;
    }
    
    .arrow-button[disabled] {
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    /* Styling for navigation buttons */
    button[key="prev_alt"], button[key="next_alt"] {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: white !important;
        font-size: 18px !important;
        width: 45px !important;
        height: 40px !important;
        border-radius: 0.5rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        line-height: 1 !important;
        margin: 0 auto !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    
    /* Add spacing for columns containing navigation buttons */
    [data-testid="column"] button[key="prev_alt"] {
        margin-right: 10px !important;
    }
    
    [data-testid="column"] button[key="next_alt"] {
        margin-left: 10px !important;
    }
    
    /* Target the button's text element to ensure proper centering */
    button[key="prev_alt"] p, button[key="next_alt"] p {
        margin: 0 !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        color: white !important;
    }
    
    button[key="prev_alt"]:hover:not([disabled]), button[key="next_alt"]:hover:not([disabled]) {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
    }
    
    /* Alternative display styling */
    .alternative-display {
        margin: 15px auto;
        padding: 15px;
        background-color: #f0f7ff;
        border-radius: 5px;
        text-align: center;
        max-width: 600px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Consistent button styling - updated to match Home.py */
    .stButton > button {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.025em !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
        color: white !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
        color: white !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%) !important;
    }
    
    /* Wide button for alternatives */
    .wide-button {
        min-width: 200px;
    }
    
    /* Metrics table styling */
    .metrics-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        text-align: center;
        table-layout: fixed;
    }
    
    .metrics-table th {
        background-color: #f0f2f6;
        padding: 12px;
        border-bottom: 2px solid #ddd;
        font-weight: bold;
    }
    
    .metrics-table td {
        padding: 10px;
        border-bottom: 1px solid #eee;
    }
    
    /* Center browse alternatives section */
    .browse-alternatives-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        max-width: 600px;
        width: 100%;
    }
    
    /* Flex container for arrows */
    .arrow-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 30px;
        margin: 10px auto;
        width: 120px;
    }
    
    /* Use Alternative button */
    .use-alternative-btn {
        margin: 15px auto;
        display: block;
    }
    
    /* Centered dropdown */
    .centered-dropdown {
        margin: 0 auto;
        max-width: 400px;
        text-align: center;
    }
    
    /* Override Streamlit's default styles to center elements in alternatives section */
    .browse-alternatives .stButton {
        display: flex;
        justify-content: center;
    }
    
    .stExpander {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Ensure elements inside expander are centered */
    .stExpander > div > div:nth-child(2) {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }
    
    /* Fix button alignment in columns */
    [data-testid="column"] .stButton {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    
    /* Clean up obsolete styles and focus on the simplified approach */
    .alternative-nav {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        margin: 10px auto;
    }
    
    /* Use alternative button styling */
    button[key="use_alternative"] {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        margin: 15px auto !important;
        display: block !important;
        padding: 0.75rem 1.5rem !important;
        letter-spacing: 0.025em !important;
        transition: all 0.3s ease !important;
    }
    
    button[key="use_alternative"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
    }
    
    /* Center the table and give it fixed width */
    .styled-table-container {
        margin: 0 auto;
        max-width: 800px;
        width: 100%;
    }
    
    /* New class for centering visualization header and table */
    .viz-recommendations-container {
        max-width: 900px;
        margin: 0 auto;
        text-align: center;
    }
    
    /* Metrics panel styling - horizontal layout */
    .metrics-panel {
        display: flex;
        flex-direction: row;
        flex-wrap: nowrap;
        justify-content: space-between;
        gap: 10px; /* Reduced from 15px */
        background-color: transparent;
        padding: 0;
        margin-bottom: 10px; /* Reduced from 20px */
        width: 100%;
    }
    
    /* Section titles */
    .section-title {
        font-size: 18px;
        font-weight: 600;
        margin: 8px 0; /* Reduced from 10px 0 */
        padding-bottom: 5px; /* Reduced from 8px */
        border-bottom: 2px solid #f0f2f6;
    }
    
    /* Visualization grid layout */
    .viz-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 15px;
        margin-bottom: 20px;
    }
    
    /* Individual visualization container */
    .viz-container {
        background-color: transparent;
        border-radius: 0;
        padding: 0;
        box-shadow: none;
        margin-bottom: 0;
    }
    
    .viz-container:hover {
        transform: none;
        box-shadow: none;
    }
    
    /* Make plot titles centered */
    .js-plotly-plot .plotly .main-svg .infolayer .g-gtitle {
        text-anchor: middle !important;
    }
    
    /* Visualization title and subtitle */
    .viz-title {
        font-size: 16px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 5px;
        color: #333;
    }
    
    .viz-subtitle {
        font-size: 13px;
        color: #666;
        text-align: center;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid #f0f2f6;
    }
    
    /* Metric card styling */
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 12px 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
        flex: 1 1 0;
        min-width: 0;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .metric-card.mean-bg {
        border-left-color: #2196F3;
    }
    
    .metric-card.sum-bg {
        border-left-color: #FF9800;
    }
    
    .metric-card.count-bg {
        border-left-color: #9C27B0;
    }
    
    .metric-label {
        font-size: 14px;
        color: #666;
        margin-bottom: 5px;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 600;
        color: #333;
    }
    
    .metric-desc {
        font-size: 12px;
        color: #888;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Domain keywords for context selection
domain_keywords = { 
    "General": {},
    "Sales / Ventes": {
        "sales": 2, "revenue": 2, "profit": 2, "quantity": 2, "cost": 1.5, "amount": 1.5, "turnover": 2, "commission": 1.5,
        "vente": 2, "chiffre d'affaires": 2, "bénéfice": 2, "quantité": 2, "coût": 1.5, "montant": 1.5, "commission": 1.5
    },
    "Logistics / Logistique": {
        "weight": 1.5, "distance": 1.5, "time": 1.5, "shipping": 2, "delivery": 2, "transport": 1.5, "stock": 1.5, "warehouse": 1.5,
        "poids": 1.5, "distance": 1.5, "temps": 1.5, "expédition": 2, "livraison": 2, "transport": 1.5, "stock": 1.5, "entrepôt": 1.5
    },
    "Maintenance": {
        "repair": 1.5, "cost": 2, "hours": 1.5, "downtime": 2, "failure": 1.5, "service": 1.5, "inspection": 1.5, "parts": 1.5,
        "réparation": 1.5, "coût": 2, "heures": 1.5, "temps d'arrêt": 2, "panne": 1.5, "service": 1.5, "inspection": 1.5, "pièces": 1.5
    },
    "Finance": {
        "income": 2, "expense": 2, "investment": 2, "loan": 1.5, "interest": 1.5, "tax": 1.5, "credit": 1.5, "debt": 1.5, "salary": 2,
        "revenu": 2, "dépense": 2, "investissement": 2, "prêt": 1.5, "intérêt": 1.5, "impôt": 1.5, "crédit": 1.5, "dette": 1.5, "salaire": 2
    },
    "Education / Éducation": {
        "score": 2, "grade": 2, "attendance": 1.5, "study": 1.5, "tuition": 2, "exam": 1.5, "homework": 1.5, "learning": 2, "GPA": 2,
        "note": 2, "classement": 2, "présence": 1.5, "étude": 1.5, "frais de scolarité": 2, "examen": 1.5, "devoir": 1.5, "apprentissage": 2
    },
    "Health / Santé": {
        "heart rate": 2, "blood pressure": 2, "cholesterol": 1.5, "bmi": 1.5, "steps": 1.5, "calories": 1.5, "exercise": 1.5, "sleep": 2, "weight": 1.5,
        "rythme cardiaque": 2, "pression artérielle": 2, "cholestérol": 1.5, "IMC": 1.5, "pas": 1.5, "calories": 1.5, "exercice": 1.5, "sommeil": 2, "poids": 1.5
    },
    "Social Media / Réseaux Sociaux": {
        "likes": 2, "shares": 2, "followers": 1.5, "comments": 1.5, "engagement": 2, "posts": 1.5, "views": 1.5, "subscribers": 1.5,
        "mentions j'aime": 2, "partages": 2, "abonnés": 1.5, "commentaires": 1.5, "engagement": 2, "publications": 1.5, "vues": 1.5, "inscrits": 1.5
    },
    "Production": {
        "units": 2, "efficiency": 2, "defects": 1.5, "yield": 1.5, "downtime": 1.5, "productivity": 2, "output": 2, "manufacturing": 2,"production":2, "stocks": 1.5,
        "unités": 2, "efficacité": 2, "défauts": 1.5, "rendement": 1.5, "temps d'arrêt": 1.5, "productivité": 2, "production": 2, "fabrication": 2
    },
    "E-commerce": {
        "sales": 2, "orders": 2, "cart": 1.5, "conversion": 2, "return": 1.5, "discount": 1.5, "customer": 1.5, "rating": 1.5, "reviews": 1.5,
        "ventes": 2, "commandes": 2, "panier": 1.5, "conversion": 2, "retour": 1.5, "réduction": 1.5, "client": 1.5, "évaluation": 1.5, "avis": 1.5
    },
    "Energy / Énergie": {
        "power": 2, "consumption": 2, "fuel": 1.5, "electricity": 2, "gas": 1.5, "efficiency": 1.5, "renewable": 2, "solar": 2, "wind": 2,
        "puissance": 2, "consommation": 2, "carburant": 1.5, "électricité": 2, "gaz": 1.5, "efficacité": 1.5, "renouvelable": 2, "solaire": 2, "éolien": 2
    },
    "Real Estate / Immobilier": {
        "property": 2, "price": 2, "rent": 2, "mortgage": 2, "investment": 2, "square footage": 1.5, "valuation": 1.5,
        "propriété": 2, "prix": 2, "loyer": 2, "hypothèque": 2, "investissement": 2, "superficie": 1.5, "évaluation": 1.5
    },
    "Human Resources / Ressources Humaines": {
        "salary": 2, "bonus": 2, "hiring": 2, "promotion": 2, "benefits": 1.5, "training": 1.5, "recruitment": 2,
        "salaire": 2, "prime": 2, "embauche": 2, "promotion": 2, "avantages": 1.5, "formation": 1.5, "recrutement": 2
    },
    "Technology / Informatique": {
        "server": 2, "cloud": 2, "AI": 2, "algorithm": 1.5, "machine learning": 2, "CPU": 1.5, "GPU": 1.5, "latency": 1.5,
        "serveur": 2, "cloud": 2, "IA": 2, "algorithme": 1.5, "apprentissage automatique": 2, "processeur": 1.5, "graphique": 1.5, "latence": 1.5
    }
}

# Helper functions for metrics recommendation
def detect_id_column(column_name):
    id_keywords = ["id", "code", "number", "uuid", "identifier", "reference", "index", "key"]
    return any(keyword in column_name.lower() for keyword in id_keywords)

def calculate_entropy(series):
    binned = pd.qcut(series, q=min(10, len(series.unique())), duplicates='drop')
    counts = binned.value_counts(normalize=True)
    return entropy(counts, base=2)

def clean_column_name(col_name):
    col_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', col_name)
    col_name = col_name.replace("_", " ")
    return col_name.lower().strip()

def business_relevance_boost(col_name, selected_domain):
    if selected_domain in domain_keywords:
        col_name = clean_column_name(col_name)
        boost = sum(weight for keyword, weight in domain_keywords[selected_domain].items() if keyword in col_name)
        return boost
    return 0

def suggest_aggregation(series, col_name):
    unique_ratio = series.nunique() / len(series)
    col_name_clean = clean_column_name(col_name).lower()

    # Rule 1: ID columns → count
    if detect_id_column(col_name):
        return 'count'
    
    # Rule 2: Force sum if column has 'total' or 'sum'
    if 'total' in col_name_clean or 'sum' in col_name_clean:
        return 'sum'
    
    # Rule 3: Force mean if column has 'rate'
    if 'rate' in col_name_clean:
        return 'mean'

    # Rule 4: Special keywords → mean
    if any(keyword in col_name_clean for keyword in ['age', 'price', 'rating', 'score', 'percent']):
        return 'mean'

    # Rule 5: Unique ratio based decisions
    if unique_ratio < 0.05:
        return 'count'
    elif unique_ratio > 0.95:
        return 'mean'
    else:
        # Rule 6: Mostly positive numeric → sum
        if pd.api.types.is_numeric_dtype(series) and (series > 0).mean() > 0.9:
            return 'sum'
        else:
            return 'mean'

def rank_columns(df, selected_domain):
    # Clean column names in the DataFrame
    df.columns = [clean_column_name(col) for col in df.columns]
    
    numerical_cols = [col for col in df.select_dtypes(include=['number']).columns if not detect_id_column(col)]
    scores = []

    for col in numerical_cols:
        series = df[col].dropna()
        if len(series) < 5:
            continue

        col_cleaned = clean_column_name(col)

        cv = series.std() / series.mean() if series.mean() != 0 else 0
        skw = abs(skew(series))
        krt = abs(kurtosis(series))
        ent = calculate_entropy(series)
        uniq = len(series.unique()) / len(series)
        outlier = np.clip((series > series.mean() + 3 * series.std()).sum() / len(series), 0, 1)

        norm_cv = np.tanh(cv)
        norm_skw = np.tanh(skw / 10)
        norm_krt = np.tanh(krt / 15)
        norm_ent = ent / np.log(len(series.unique()) + 1)
        norm_uniq = np.sqrt(uniq)
        norm_outlier = outlier

        score = (
            1.5 * norm_cv +
            1.2 * norm_skw +
            1.0 * norm_krt +
            0.8 * norm_ent +
            0.7 * norm_uniq +
            0.6 * norm_outlier
        )

        boost = business_relevance_boost(col_cleaned, selected_domain)
        score += boost

        suggested_agg = suggest_aggregation(series, col)

        scores.append({
            'Column': col,
            'CV_Score': round(norm_cv * 1.5, 3),
            'Skew_Score': round(norm_skw * 1.2, 3),
            'Kurtosis_Score': round(norm_krt * 1.0, 3),
            'Entropy_Score': round(norm_ent * 0.8, 3),
            'Uniqueness_Score': round(norm_uniq * 0.7, 3),
            'Outlier_Score': round(norm_outlier * 0.6, 3),
            'Business_Relevance_Boost': boost,
            'Final_Score': round(score, 3),
            'Suggested_Aggregation': suggested_agg
        })

    sorted_scores = sorted(scores, key=lambda x: x['Final_Score'], reverse=True)
    return sorted_scores[:3], sorted_scores

# Initialize all required session state variables
if "language" not in st.session_state:
    st.session_state.language = "en"
if "progress" not in st.session_state:
    st.session_state.progress = {'upload': True, 'process': True, 'clean': True, 'visualize': True}
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "cleaned_dataframes" not in st.session_state:
    st.session_state.cleaned_dataframes = {}
if "data_analyzed" not in st.session_state:
    st.session_state.data_analyzed = False
if "data_quality_scores" not in st.session_state:
    st.session_state.data_quality_scores = {}
if "cleaning_mode" not in st.session_state:
    st.session_state.cleaning_mode = None
if "processing_mode" not in st.session_state:
    st.session_state.processing_mode = None

# Dashboard wizard state variables
if "dashboard_step" not in st.session_state:
    st.session_state.dashboard_step = 1
if "selected_domain" not in st.session_state:
    st.session_state.selected_domain = None
if "top3" not in st.session_state:
    st.session_state.top3 = None
if "full_rank" not in st.session_state:
    st.session_state.full_rank = None
if "alternatives" not in st.session_state:
    st.session_state.alternatives = None
if "dummy" not in st.session_state:
    st.session_state.dummy = 0
if 'viz_recommendations' not in st.session_state:
    st.session_state.viz_recommendations = None
if 'selected_visualization' not in st.session_state:
    st.session_state.selected_visualization = None

def advance_step():
    # Clear recommendations when advancing to step 3
    if st.session_state.dashboard_step == 2:
        # Going from step 2 to step 3, clear any previous viz recommendations
        if 'viz_recommendations' in st.session_state:
            st.session_state.viz_recommendations = None
    
    # Increment the step
    st.session_state.dashboard_step += 1

def domain_step():
    st.title(t("Dashboard Configuration"))
    st.markdown(f"<h3 style='text-align: center;'>{t('Step 1: Choose Data Context')}</h3>", unsafe_allow_html=True)
    
    # First check if data exists in any of the possible session state variables
    has_data = False
    if "cleaned_dataframes" in st.session_state and st.session_state.cleaned_dataframes:
        has_data = True
    elif "dataframes" in st.session_state and st.session_state.dataframes:
        has_data = True
    elif "dashboard_uploaded_df" in st.session_state and st.session_state.dashboard_uploaded_df is not None:
        has_data = True
    
    # If no data is available, display warning and return to Home button
    if not has_data:
        st.error(t("No data available. Please upload a CSV file first."))
        
        # Center the button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(t("Go to Home Page"), use_container_width=True):
                st.switch_page("Home.py")
        return
        
    # If we have data, continue with domain selection
    st.markdown(f"<p style='text-align: center;'>{t('Select the domain that best matches your data to get more relevant visualization recommendations.')}</p>", unsafe_allow_html=True)
    
    # Add custom CSS to make the domain selectbox properly centered and contained
    st.markdown("""
        <style>
        /* Domain selection dropdown styles */
        .domain-selection-container {
            max-width: 100%;
            margin: 0 auto;
        }
        
        .domain-selection-container [data-testid="stSelectbox"] {
            max-width: 100%;
            width: 100%;
        }
        
        /* Make select element contained */
        .domain-selection-container [data-testid="stSelectbox"] > div > div {
            max-width: 100%;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
        }
        
        /* Center the label */
        .domain-selection-container [data-testid="stSelectbox"] label {
            text-align: center;
            width: 100%;
            display: block;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Use columns for better layout - larger middle column
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Wrap in a container for styles
        st.markdown('<div class="domain-selection-container">', unsafe_allow_html=True)
    
        domain_options = [t("Select a domain")] + list(domain_keywords.keys())
        selected_domain = st.selectbox(
            t("Select dataset domain"), 
            domain_options, 
            index=0 if st.session_state.selected_domain is None else domain_options.index(st.session_state.selected_domain)
        )
    
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Only show next button when a domain is selected
    if selected_domain != t("Select a domain"):
        st.session_state.selected_domain = selected_domain
        
        # Center the button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(t("Next"), use_container_width=True, type="primary"):
                # Force immediate step advance without requiring a second click
                st.session_state.dashboard_step = 2
                st.rerun()

def metrics_recommendation_step():
    # This should only run when we're on step 2, so let's validate that first
    if st.session_state.dashboard_step != 2:
        return
    
    # Check for data in various possible session state variables
    if "cleaned_dataframes" in st.session_state and st.session_state.cleaned_dataframes:
        # Use the first cleaned dataframe
        df_key = list(st.session_state.cleaned_dataframes.keys())[0]
        df = st.session_state.cleaned_dataframes[df_key]
        st.success(f"{t('Using cleaned data from previous steps')}: {df_key}")
    elif "dataframes" in st.session_state and st.session_state.dataframes:
        # If no cleaned data, try using the raw dataframe
        df_key = list(st.session_state.dataframes.keys())[0]
        df = st.session_state.dataframes[df_key]
        st.info(f"{t('Using raw data from previous steps')}: {df_key}")
    elif "dashboard_uploaded_df" in st.session_state:
        df = st.session_state.dashboard_uploaded_df
        st.info(t("Using previously uploaded file"))
    else:
        # If no data is available, instruct to return to Home
        st.error(t("No data available. Please go back to process data first."))
        if st.button(t("Back to Home"), use_container_width=True):
            st.switch_page("Home.py")
        return
    
    # Clean column names to avoid KeyError
    df.columns = [clean_column_name(col) for col in df.columns]
    
    # Initialize rankings if not already done
    if st.session_state.full_rank is None or st.session_state.top3 is None or st.session_state.alternatives is None:
        top3_candidates, full_ranking = rank_columns(df, st.session_state.selected_domain)
        st.session_state.full_rank = full_ranking
        st.session_state.top3 = top3_candidates
        st.session_state.alternatives = full_ranking[3:8]
    
    # Show Top Metrics in a horizontal table with just column names and aggregation types
    st.markdown(t("### 🏆 Recommended Metrics"))
    
    # Create a horizontal table with HTML
    metrics_html = """
    <table class="metrics-table">
        <thead>
            <tr>
                <th>Metric 1</th>
                <th>Metric 2</th>
                <th>Metric 3</th>
            </tr>
        </thead>
        <tbody>
            <tr>
    """
    
    # Add metric names
    for item in st.session_state.top3:
        metrics_html += f"<td>{item['Column']}</td>"
    
    metrics_html += """
            </tr>
            <tr>
    """
    
    # Add aggregation types with styling
    for item in st.session_state.top3:
        agg_type = item["Suggested_Aggregation"]
        agg_class = f"{agg_type}-bg"
        metrics_html += f'<td><span class="{agg_class}">{agg_type}</span></td>'
    
    metrics_html += """
            </tr>
        </tbody>
    </table>
    """
    
    # Display the metrics table
    st.write(metrics_html, unsafe_allow_html=True)
    
    # Show Alternatives section in an expander
    with st.expander(t("🔄 Browse Alternatives"), expanded=False):
        st.markdown('<div class="alternatives-section">', unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; margin-bottom:15px;'>{t('Select a metric to replace and browse alternatives.')}</div>", unsafe_allow_html=True)
        
        # Select which top metric to replace
        selected_metric = st.selectbox(
            t("Select metric to replace:"), 
            st.session_state.top3, 
            format_func=lambda x: x['Column'],
            key="selected_metric_to_replace"
        )
        
        # Get the position of selected metric
        pos_idx = st.session_state.top3.index(selected_metric)
        
        # Display current metric
        st.markdown(f"""
        <div style='text-align:center; margin:15px 0;'>
            <span style='font-weight:bold;'>{t('Current Metric')}:</span> 
            <span>{selected_metric['Column']}</span> 
            (<span class='{selected_metric['Suggested_Aggregation']}-bg'>{selected_metric['Suggested_Aggregation']}</span>)
        </div>
        """, unsafe_allow_html=True)
        
        # Add custom CSS to make the alternatives section wider
        st.markdown("""
            <style>
            /* Make alternatives section wider */
            .alternative-display {
                width: 100%;
                max-width: 900px;
            }
            
            /* Make the entire expander wider */
            [data-testid="stExpander"] {
                max-width: 900px !important;
                width: 100% !important;
                margin: 0 auto !important;
            }
            
            /* Make the dropdown wider but keep it centered - updated to be more specific */
            .alternatives-section .stSelectbox {
                min-width: 300px !important;
                max-width: 900px !important;
                width: 100% !important;
                margin: 0 auto !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Initialize navigation counter if not in session state
        if "nav_position" not in st.session_state:
            st.session_state.nav_position = {"metrics_alternatives": 0}
            
        # Get position for this section
        alt_pos = st.session_state.nav_position.get("metrics_alternatives", 0)
        
        # Show the alternatives
        if st.session_state.alternatives and len(st.session_state.alternatives) > 0:
            total_alternatives = len(st.session_state.alternatives)
            
            # Ensure position doesn't go out of bounds
            alt_pos = max(0, min(alt_pos, total_alternatives - 1))
            
            alternative_item = st.session_state.alternatives[alt_pos]
            
            st.markdown(f"<div style='text-align:center; margin-bottom:10px;'>{t('Alternative')} {alt_pos+1} {t('of')} {total_alternatives}</div>", unsafe_allow_html=True)
            
            # Add some space
            st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
            
            # Create navigation controls with columns
            nav_cols = st.columns([1, 1, 2, 1, 1])
            
            # Previous button
            prev_disabled = alt_pos <= 0
            with nav_cols[0]:
                if st.button("◀", key="prev_alt_metrics", disabled=prev_disabled, use_container_width=True):
                    st.session_state.nav_position["metrics_alternatives"] = alt_pos - 1
                    st.rerun()
            
            # Position indicator
            with nav_cols[2]:
                st.markdown(f"<div style='text-align: center; font-weight: bold;'>{alt_pos+1}/{total_alternatives}</div>", unsafe_allow_html=True)
            
            # Next button
            next_disabled = alt_pos >= total_alternatives - 1
            with nav_cols[4]:
                if st.button("▶", key="next_alt_metrics", disabled=next_disabled, use_container_width=True):
                    st.session_state.nav_position["metrics_alternatives"] = alt_pos + 1
                    st.rerun()
            
            st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
            
            # Display alternative
            st.markdown(f"""
            <div style='text-align:center; padding:15px; background-color:#f8f9fa; border-radius:5px;'>
                <div style='font-size:18px; font-weight:bold; margin-bottom:8px;'>{alternative_item['Column']}</div>
                <div style='margin-bottom:5px;'>
                    <span class="{alternative_item['Suggested_Aggregation']}-bg">{alternative_item['Suggested_Aggregation']}</span>
                </div>
                <div style='font-size:14px; color:#666; margin-top:10px;'>{t('Score')}: {alternative_item.get('Score', alternative_item.get('Final_Score', 0)):.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add button to use this alternative
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                if st.button(t("Use This Alternative"), key=f"use_metrics_{pos_idx}_{alt_pos}", use_container_width=True, type="primary"):
                    # Replace the selected metric with this alternative
                    # First check if alternative_item is already a dict or needs conversion
                    if hasattr(alternative_item, 'to_dict'):
                        st.session_state.top3[pos_idx] = alternative_item.to_dict()
                    else:
                        # It's already a dict
                        st.session_state.top3[pos_idx] = alternative_item
                    st.success(t("Metric replaced successfully!"))
                    st.rerun()
        else:
            st.info(t("No alternatives available for this metric."))
            
        st.markdown('</div>', unsafe_allow_html=True) # Close alternatives-section
    
    # Add a next button to advance to step 3
    if st.button(t("Next"), use_container_width=True, type="primary"):
        advance_step()
        st.rerun()  # Add explicit rerun to ensure step advances immediately

def extract_columns_from_name(row, df):
    """
    Extract and validate column names from a recommendation row.
    
    Parameters:
    -----------
    row : pandas.Series
        The recommendation row with Name and Type
    df : pandas.DataFrame
        The dataframe to validate column existence
        
    Returns:
    --------
    list
        List of valid column names
    """
    # Check if Name is present and is a string
    if 'Name' not in row:
        print(f"WARNING: 'Name' not found in row: {row}")
        return []
        
    name = row['Name']
    
    # Handle non-string names
    if not isinstance(name, str):
        print(f"WARNING: 'Name' is not a string, type is {type(name)}")
        if isinstance(name, (int, float)):
            # For numeric values, check if any column name matches this value
            matching_cols = [col for col in df.columns if str(name) == col]
            if matching_cols:
                return matching_cols[:1]
            # Otherwise return empty list
            return []
        # Try converting to string if possible
        try:
            name = str(name)
        except:
            return []
    
    row_type = row['Type']
    columns = []
    
    print(f"DEBUG - Extracting columns for {row_type}: {name}")
    
    # Extract columns based on row type
    if row_type == 'Column':
        # Single column - just return it if it exists
        if name in df.columns:
            return [name]
        return []
    
    elif row_type == 'Pair':
        # Try common separators for pairs
        for sep in [' & ', ', ', ' vs ', ' by ', ' and ', ' with ']:
            if sep in name:
                columns = [col.strip() for col in name.split(sep)]
                if len(columns) >= 2:
                    print(f"DEBUG - Found pair columns using separator '{sep}': {columns}")
                    break
        
        # If still no columns and looks like a tuple, try that format
        if len(columns) < 2 and name.startswith('(') and name.endswith(')'):
            try:
                # Remove parentheses and split by comma
                cols_str = name[1:-1]
                extracted = []
                for col in cols_str.split(','):
                    col = col.strip()
                    # Remove quotes if present
                    if (col.startswith("'") and col.endswith("'")) or (col.startswith('"') and col.endswith('"')):
                        col = col[1:-1]
                    extracted.append(col)
                if len(extracted) >= 2:
                    columns = extracted
                    print(f"DEBUG - Found pair columns from tuple: {columns}")
            except:
                # Fallback if tuple parsing fails
                pass
                
        # Last resort - try a simple split on ampersand
        if len(columns) < 2:
            columns = name.split(' & ')
            
        print(f"DEBUG - Extracted pair columns: {columns}")
    
    elif row_type == 'Triple':
        # Try common separators for triples
        for sep in [' & ', ', ']:
            if sep in name:
                columns = [col.strip() for col in name.split(sep)]
                if len(columns) >= 3:
                    break
                    
        # If that didn't work, try more complex formats like "X vs Y by Z"
        if len(columns) < 3:
            if ' vs ' in name and ' by ' in name:
                parts = name.split(' vs ')
                col1 = parts[0].strip()
                parts2 = parts[1].split(' by ')
                col2 = parts2[0].strip()
                col3 = parts2[1].strip()
                columns = [col1, col2, col3]
                
        # Last resort - split on ampersand
        if len(columns) < 3:
            columns = name.split(' & ')
    
    elif row_type == 'GroupBy':
        # Try extracting GroupBy columns with different formats
        if ' [by] ' in name:
            columns = name.split(' [by] ', 1)
        elif ' grouped by ' in name:
            columns = name.split(' grouped by ', 1)
        elif ' by ' in name:
            columns = name.split(' by ', 1)
        else:
            columns = [name]
    
    # Validate that columns exist in the dataframe
    valid_columns = [col for col in columns if col in df.columns]
    
    # If we don't have enough columns for the type, try fallbacks
    if row_type == 'Pair' and len(valid_columns) < 2 and len(df.columns) >= 2:
        # For pairs, we need at least 2 columns
        if len(valid_columns) == 1:
            # We have one valid column, find another one
            other_cols = [col for col in df.columns if col != valid_columns[0]]
            if other_cols:
                valid_columns.append(other_cols[0])
                print(f"DEBUG - Added complementary numeric column: {other_cols[0]}")
            else:
                # If no numeric column, use any other column
                other_col = next((col for col in df.columns if col != columns[0]), None)
                if other_col:
                    columns.append(other_col)
                    print(f"DEBUG - Added any complementary column: {other_col}")
        elif len(columns) == 0:
            # No valid column, use first two numeric columns as fallback
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            if len(numeric_cols) >= 2:
                columns = numeric_cols[:2]
                print(f"DEBUG - Using first two numeric columns: {columns}")
            else:
                columns = list(df.columns[:2])
                print(f"DEBUG - Using first two columns: {columns}")
            
        print(f"DEBUG - Final columns for pair visualization: {columns}")
    
    # Check if all specified columns exist in the dataframe
    existing_columns = [col for col in columns if col in df.columns]
    if not existing_columns:
        print(f"WARNING: None of the columns {columns} exist in the dataframe")
        # Use first columns as fallback
        if len(df.columns) > 0:
            existing_columns = [df.columns[0]]
            if row_type == 'Pair' and len(df.columns) > 1:
                existing_columns.append(df.columns[1])
            print(f"DEBUG - Using fallback columns: {existing_columns}")
    
    # Final log of columns to be used
    print(f"DEBUG - Final columns for visualization: {existing_columns}")
    
    return existing_columns

def visualization_recommendation_step():
    """Generate and display visualization recommendations based on the data."""
    
    # This should only run when we're on step 3, so let's validate that first
    if st.session_state.dashboard_step != 3:
        return
    
    # Get the dataframe from session state (handle different possible storage locations)
    if "dataframes" in st.session_state and st.session_state.dataframes:
        df_key = list(st.session_state.dataframes.keys())[0]
        df = st.session_state.dataframes[df_key]
    elif "dashboard_uploaded_df" in st.session_state:
        df = st.session_state.dashboard_uploaded_df
    else:
        st.error("No data found. Please return to previous steps.")
        return

    # Display centered header
    st.markdown("""
    <div class="viz-recommendations-container">
      <h3>🏆 Recommended Visualisations</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate the file ID for tracking changes
    current_file_id = None
    if "dataframes" in st.session_state and st.session_state.dataframes:
        df_key = list(st.session_state.dataframes.keys())[0]
        current_file_id = hash(df_key + str(df.shape) + str(list(df.columns)))
    elif "dashboard_uploaded_df" in st.session_state:
        current_file_id = hash("dashboard_df" + str(df.shape) + str(list(df.columns)))
    
    # Initialize file_id in session state if it doesn't exist
    if 'file_id' not in st.session_state:
        st.session_state.file_id = None
    
    # Check if we need to recalculate scores (new file upload)
    new_file_upload = st.session_state.file_id != current_file_id
    
    # Calculate scores only if this is a new file or first run
    if new_file_upload or 'all_scores' not in st.session_state:
        # Update file ID
        st.session_state.file_id = current_file_id
        
        # Calculate scores for all columns, pairs, and triples
        with st.spinner("Analyzing data and calculating visualization scores..."):
            column_scores, pair_scores, triple_scores, groupby_scores = score_all_columns_and_pairs(df)
        
        # Prepare data for display
        column_data = []
        for column, scores in column_scores.items():
            column_data.append({
                'Name': column,
                'Type': 'Column',
                'Distribution': round(scores['distribution_score'], 2),
                'Type Match': round(scores['data_type_score'], 2),
                'Quality': round(scores['data_quality_score'], 2),
                'Predictive': round(scores['predictive_power_score'], 2),
                'Semantic': round(scores['semantic_content_score'], 2),
                'Dimensional': round(scores['dimensional_analysis_score'], 2),
                'Variance': round(scores['variance_info_ratio_score'], 2),
                'Total Score': round(scores['total_score'], 2)
            })
        
        # Column pairs
        pair_data = []
        for pair, scores in pair_scores.items():
            pair_data.append({
                'Name': pair,
                'Type': 'Pair',
                'Association': round(scores['statistical_association'], 2),
                'Complexity': round(scores['visualization_complexity'], 2),
                'Pattern': round(scores['pattern_detection'], 2),
                'Anomaly': round(scores['anomaly_highlighting'], 2),
                'Complementarity': round(scores['information_complementarity'], 2),
                'Redundancy': round(scores['redundancy_penalization'], 2),
                'Utility': round(scores['practical_utility_score'], 2),
                'Total Score': round(scores['total_score'], 2)
            })
        
        # Column triples
        triple_data = []
        for triple, scores in triple_scores.items():
            triple_data.append({
                'Name': triple,
                'Type': 'Triple',
                'Dimensional Balance': round(scores['dimensional_balance'], 2),
                'Information Density': round(scores['information_density'], 2),
                'Visualization Feasibility': round(scores['visualization_feasibility'], 2),
                'Insight Potential': round(scores['insight_potential'], 2),
                'Interaction Synergy': round(scores['interaction_synergy'], 2),
                'Complexity Penalty': round(scores['complexity_penalty'], 2),
                'Total Score': round(scores['total_score'], 2)
            })
        
        # GroupBy pairs
        groupby_data = []
        for pair, scores in groupby_scores.items():
            # For app.py compatibility, format the name with [by] separator
            try:
                groupby_col, agg_col = pair
                pair_name = f"{groupby_col} [by] {agg_col}"
            except:
                # Handle the case where the pair might be formatted differently
                pair_name = str(pair)
            
            groupby_data.append({
                'Name': pair_name,
                'Type': 'GroupBy',
                'Group Differentiation': round(scores['group_differentiation'], 2),
                'Aggregation Meaningfulness': round(scores['aggregation_meaningfulness'], 2),
                'Group Size Balance': round(scores['group_size_balance'], 2),
                'Outlier Robustness': round(scores['outlier_robustness'], 2),
                'Visualization Potential': round(scores['visualization_potential'], 2),
                'Total Score': round(scores['total_score'], 2)
            })
        
        # Combine all scores
        all_scores = pd.DataFrame(column_data + pair_data + triple_data + groupby_data)
        all_scores = all_scores.sort_values('Total Score', ascending=False)
        
        # Create a recommended visualization type column
        all_scores['Recommended Visualization'] = all_scores.apply(
            lambda row: get_visualization_recommendation(row, df, 
                                                       {**column_scores, **pair_scores, **triple_scores}),
            axis=1
        )
        
        # Update visualization recommendations for GroupBy pairs
        for idx, row in all_scores[all_scores['Type'] == 'GroupBy'].iterrows():
            try:
                parts = row['Name'].split(" [by] ")
                if len(parts) == 2:
                    groupby_col, agg_col = parts
                    all_scores.at[idx, 'Recommended Visualization'] = get_groupby_visualization_recommendation(
                        df, groupby_col, agg_col, groupby_scores[(groupby_col, agg_col)]
                    )
            except:
                # Handle cases where the format is different or key error
                continue
        
        # Store all results in session state
        st.session_state.all_scores = all_scores
        st.session_state.column_scores = column_scores
        st.session_state.pair_scores = pair_scores
        st.session_state.triple_scores = triple_scores
        st.session_state.groupby_scores = groupby_scores
    else:
        # Use cached scores from session state
        all_scores = st.session_state.all_scores
        column_scores = st.session_state.column_scores
        pair_scores = st.session_state.pair_scores
        triple_scores = st.session_state.triple_scores
        groupby_scores = st.session_state.groupby_scores
    
    # Only create initial recommendations once when a new file is loaded
    if new_file_upload or 'top_recommendations' not in st.session_state:
        # Get the top items of each type separately
        top_columns = all_scores[all_scores['Type'] == 'Column'].head(2)
        top_pairs = all_scores[all_scores['Type'] == 'Pair'].head(2)
        top_triples = all_scores[all_scores['Type'] == 'Triple'].head(1)  # Limit to 1
        top_groupby = all_scores[all_scores['Type'] == 'GroupBy'].head(1)  # Limit to 1
        
        # Combine and re-sort to get the top 5 overall
        candidates = pd.concat([top_columns, top_pairs, top_triples, top_groupby])
        candidates = candidates.sort_values('Total Score', ascending=False)
        
        # Now create a more balanced selection
        winners = []
        
        # Check if we have each type available
        has_columns = not top_columns.empty
        has_pairs = not top_pairs.empty
        has_triples = not top_triples.empty
        has_groupby = not top_groupby.empty
        
        # Always include highest scoring column and pair if available
        if has_columns:
            best_column = candidates[candidates['Type'] == 'Column'].iloc[0]
            winners.append(best_column)
            
        if has_pairs:
            best_pair = candidates[candidates['Type'] == 'Pair'].iloc[0]
            winners.append(best_pair)
            
        # Include one triple if available
        if has_triples:
            best_triple = candidates[candidates['Type'] == 'Triple'].iloc[0]
            winners.append(best_triple)
            
        # Include one groupby if available
        if has_groupby:
            best_groupby = candidates[candidates['Type'] == 'GroupBy'].iloc[0]
            winners.append(best_groupby)
            
        # Fill remaining slots with highest scored items not already included
        winners_names = [w.name for w in winners]
        remaining = candidates[~candidates.index.isin(winners_names)]
        
        # Add items until we reach 5 total or run out of candidates
        for _, row in remaining.iterrows():
            winners.append(row)
            if len(winners) >= 5:
                break
            
        # Convert to DataFrame
        top_recommendations = pd.DataFrame(winners)
        
        # Ensure we have exactly 5 visualizations
        if len(top_recommendations) < 5:
            print(f"WARNING: Only have {len(top_recommendations)} recommendations, need 5")
            # If we have fewer than 5, duplicate some to reach 5
            while len(top_recommendations) < 5:
                # Add the top scoring visualization again
                top_recommendations = pd.concat([top_recommendations, pd.DataFrame([top_recommendations.iloc[0]])])
                print(f"Added duplicate to reach {len(top_recommendations)} visualizations")
        
        # Ensure we have at most 5 visualizations
        if len(top_recommendations) > 5:
            print(f"WARNING: Have {len(top_recommendations)} recommendations, trimming to 5")
            top_recommendations = top_recommendations.iloc[:5]
        
        # Create recommendations with column information
        top_recommendations['columns'] = top_recommendations.apply(
            lambda row: extract_columns_from_name(row, df), axis=1
        )
        
        # Store in session state
        st.session_state.top_recommendations = top_recommendations.copy()
        st.session_state.current_recommendations = top_recommendations.copy()
        
        # Store only the top alternatives for each type (limit to top 5)
        max_alternatives = 5
        st.session_state.column_candidates = all_scores[all_scores['Type'] == 'Column'].head(max_alternatives+2).iloc[2:].copy() if len(all_scores[all_scores['Type'] == 'Column']) > 2 else pd.DataFrame()
        st.session_state.pair_candidates = all_scores[all_scores['Type'] == 'Pair'].head(max_alternatives+2).iloc[2:].copy() if len(all_scores[all_scores['Type'] == 'Pair']) > 2 else pd.DataFrame()
        st.session_state.triple_candidates = all_scores[all_scores['Type'] == 'Triple'].head(max_alternatives+1).iloc[1:].copy() if len(all_scores[all_scores['Type'] == 'Triple']) > 1 else pd.DataFrame()
        st.session_state.groupby_candidates = all_scores[all_scores['Type'] == 'GroupBy'].head(max_alternatives+1).iloc[1:].copy() if len(all_scores[all_scores['Type'] == 'GroupBy']) > 1 else pd.DataFrame()
    
    # Initialize a counter for retry clicks if it doesn't exist
    if 'retry_counter' not in st.session_state:
        st.session_state.retry_counter = 0
    
    # Use current recommendations from session state
    displayed_recommendations = st.session_state.current_recommendations
    
    # Display the table using Streamlit's native table functionality instead of HTML
    # Create a simplified table for display
    display_df = displayed_recommendations.copy().reset_index(drop=True)
    display_df.index = range(1, len(display_df) + 1)  # 1-based indexing for user
    
    # Create a DataFrame for display with desired columns
    table_data = []
    for idx, row in display_df.iterrows():
        type_val = row['Type']
        # Define type-specific styling similar to metrics step
        type_class = ""
        if type_val == 'Column':
            type_class = "mean-bg"  # Blue styling
        elif type_val == 'Pair':
            type_class = "sum-bg"   # Orange styling
        elif type_val == 'Triple':
            type_class = "count-bg" # Purple styling
        elif type_val == 'GroupBy':
            type_class = "count-bg" # Purple styling (can be changed)
            
        table_data.append({
            "#": idx,
            "Column(s)": row['Name'],
            "Recommended Visualization": row['Recommended Visualization'],
            "Type": f'<span class="{type_class}">{type_val}</span>'  # Apply styling to type
        })
    
    # Convert to DataFrame for display
    vis_table = pd.DataFrame(table_data)
    
    # Display the table using Streamlit with HTML formatting
    st.write(vis_table.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # Add a container for the Browse Alternatives section
    st.markdown("""
    <style>
    .browse-alternatives-container {
        max-width: 900px;
        margin: 20px auto;
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Make visualization alternatives wider */
    .alternative-display {
        width: 100%;
        max-width: 900px;
    }
    
    /* Make the entire expander wider */
    [data-testid="stExpander"] {
        max-width: 900px !important;
        width: 100% !important;
        margin: 0 auto !important;
    }
    
    /* Make the dropdown wider but keep it centered - updated to be more specific */
    .viz-alternatives-section .stSelectbox {
        min-width: 300px !important;
        max-width: 900px !important;
        width: 100% !important;
        margin: 0 auto !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Show Alternatives section in an expander, just like in metrics page
    with st.expander(t("🔄 Browse Alternatives"), expanded=False):
        st.markdown('<div class="viz-alternatives-section">', unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; margin-bottom:15px;'>{t('Select a recommendation to replace and browse alternatives.')}</div>", unsafe_allow_html=True)
        
        # Dropdown to select which recommendation to modify
        selected_idx = st.selectbox(
            t("Select recommendation to replace:"), 
            range(1, len(displayed_recommendations)+1),
            format_func=lambda x: f"#{x}: {displayed_recommendations.iloc[x-1]['Name']}",
            key="selected_recommendation_to_replace"
        )
            
        # Get the type of selected recommendation (using 0-based index)
        row_idx = selected_idx - 1
        row_type = displayed_recommendations.iloc[row_idx]['Type']
        
        # Create a unique key for this recommendation
        nav_key = f"{row_idx}_{row_type}"
        
        current_item = displayed_recommendations.iloc[row_idx]
        
        # Add type-specific styling
        type_color_map = {
            'Column': '#0066cc',
            'Pair': '#cc6600',
            'Triple': '#006600',
            'GroupBy': '#6600cc'
        }
        type_color = type_color_map.get(row_type, 'black')
            
        # Display current metric
        st.markdown(f"""
        <div style='text-align:center; margin:15px 0;'>
            <span style='font-weight:bold;'>{t('Current Metric')}:</span> 
            <span>{current_item['Name']}</span> 
            (<span style='color: {type_color}; font-weight: bold;'>{current_item['Type']}</span>)
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize navigation counter if not in session state
        if "nav_position" not in st.session_state:
            st.session_state.nav_position = {}
            
        # Initialize counter for this recommendation if needed
        if nav_key not in st.session_state.nav_position:
            st.session_state.nav_position[nav_key] = 0
        
        # Get the alternative position for this recommendation
        alt_pos = st.session_state.nav_position[nav_key]
        
        # Determine how many alternatives we have
        if row_type == 'Column':
            candidates = st.session_state.column_candidates
        elif row_type == 'Pair':
            candidates = st.session_state.pair_candidates
        elif row_type == 'Triple':
            candidates = st.session_state.triple_candidates
        elif row_type == 'GroupBy':
            candidates = st.session_state.groupby_candidates
        else:
            candidates = pd.DataFrame()
            
        total_alternatives = len(candidates)
        
        if total_alternatives > 0:
            # Show alternative count 
            st.markdown(f"<div style='text-align:center; margin-bottom:10px;'>Alternative {alt_pos+1} of {total_alternatives}</div>", unsafe_allow_html=True)
            
            # Use custom HTML for spacing
            st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
            
            # Use 5 columns with balanced width for better spacing - make the buttons further apart
            nav_cols = st.columns([1, 1, 2, 1, 1])
            
            # Previous button
            with nav_cols[1]:
                prev_disabled = alt_pos <= 0
                if st.button("◀", key=f"prev_alt_viz", disabled=prev_disabled, use_container_width=True):
                    if alt_pos > 0:
                        st.session_state.nav_position[nav_key] -= 1
                        st.rerun()
            
            # Center column for spacing
            with nav_cols[2]:
                st.markdown(f"<div style='text-align: center; font-weight: bold;'>{alt_pos+1}/{total_alternatives}</div>", unsafe_allow_html=True)
            
            # Next button
            with nav_cols[3]:
                next_disabled = alt_pos >= total_alternatives - 1
                if st.button("▶", key=f"next_alt_viz", disabled=next_disabled, use_container_width=True):
                    if alt_pos < total_alternatives - 1:
                        st.session_state.nav_position[nav_key] += 1
                        st.rerun()
            
            st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
            
            # Make sure alt_pos is within valid range
            alt_pos = min(alt_pos, total_alternatives - 1)
            alternative = candidates.iloc[alt_pos]
                
            # Show the alternative with styling (matching metrics page style)
            st.markdown(f"""
            <div class='alternative-display' style='margin: 15px auto; padding: 20px; background-color: #f0f7ff; border-radius: 8px; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.1);'>
                <span style='font-size: 16px; font-weight: 500;'>{alternative['Name']}</span><br>
                <span style='color: {type_color}; font-weight: 500; margin-top: 8px; display: inline-block;'>
                    {alternative['Type']}
                </span> | 
                <span style='font-style: italic;'>
                    {alternative['Recommended Visualization']}
                </span>
                <span style='font-style: italic; margin-left: 10px;'>
                    (Score: {alternative['Total Score']:.2f})
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Center the Use button (visualization preview removed)
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                if st.button("Use This Alternative", key=f"use_viz_{row_idx}_{alt_pos}", use_container_width=True, type="primary"):
                    # Debug before replacement
                    print(f"\n====== BEFORE ALTERNATIVE SELECTION DEBUG ======")
                    print(f"displayed_recommendations shape: {displayed_recommendations.shape}")
                    print(f"displayed_recommendations index: {displayed_recommendations.index.tolist()}")
                    print(f"Replacing row at index {row_idx}")
                    print(f"Current visualization: {displayed_recommendations.iloc[row_idx]['Name']} (Type: {displayed_recommendations.iloc[row_idx]['Type']})")
                    print(f"New visualization: {alternative['Name']} (Type: {alternative['Type']})")
                    print(f"======= END BEFORE DEBUG =======\n")
                    
                    # More robust replacement approach - create a new DataFrame
                    # Convert alternative to Series if it's a dictionary
                    if not hasattr(alternative, 'name'):
                        alternative = pd.Series(alternative)
                    
                    # Extract columns for the alternative
                    alternative_row = alternative.copy()
                    extracted_columns = extract_columns_from_name(alternative_row, df)
                    alternative_row['columns'] = extracted_columns
                    
                    # Replace the row in displayed_recommendations
                    temp_df = displayed_recommendations.copy()
                    temp_df.iloc[row_idx] = alternative_row
                    displayed_recommendations = temp_df
                    
                    # Print debug info for columns
                    print(f"DEBUG - Alternative selected: {alternative_row['Name']}")
                    print(f"DEBUG - Extracted columns: {extracted_columns}")
                    
                    # Debug after replacement
                    print(f"\n====== AFTER ALTERNATIVE SELECTION DEBUG ======")
                    print(f"displayed_recommendations shape: {displayed_recommendations.shape}")
                    print(f"displayed_recommendations index: {displayed_recommendations.index.tolist()}")
                    print(f"Row at index {row_idx} now contains: {displayed_recommendations.iloc[row_idx]['Name']} (Type: {displayed_recommendations.iloc[row_idx]['Type']})")
                    print(f"======= END AFTER DEBUG =======\n")
                    
                    # Update session_state.viz_recommendations with the new selection
                    st.session_state.viz_recommendations = displayed_recommendations.copy()
                    
                    # Update displayed_recommendations in the current function scope
                    st.session_state.current_recommendations = displayed_recommendations.copy()
                    
                    # Ensure we have at most 5 recommendations
                    if len(st.session_state.current_recommendations) > 5:
                        print(f"WARNING: Trimming current_recommendations from {len(st.session_state.current_recommendations)} to 5")
                        st.session_state.current_recommendations = st.session_state.current_recommendations.iloc[:5]
                        st.session_state.viz_recommendations = st.session_state.current_recommendations.copy()
                    
                    # Force a complete refresh of the display
                    # Add a timestamp to session state to force recomputation
                    st.session_state.last_alternative_update = time.time()
                    
                    # Don't remove from candidates so user can go back to previous choices
                    # Just add a success message and keep the current navigation position
                    st.success("Metric replaced successfully!")
                    
                    # Force a rerun to update the table
                    st.rerun()
        else:
            st.info(t("No alternatives available for this recommendation."))
    
    st.markdown('</div>', unsafe_allow_html=True) # Close viz-alternatives-section
    
    # Store recommendations for dashboard
    if 'viz_recommendations' not in st.session_state or st.session_state.viz_recommendations is None:
        st.session_state.viz_recommendations = displayed_recommendations
    
    # Add a reset button to revert to original recommendations
    if st.button("↻ Reset to Original Recommendations", use_container_width=True):
        # Reset to original top recommendations
        st.session_state.current_recommendations = st.session_state.top_recommendations.copy()
        st.session_state.viz_recommendations = st.session_state.top_recommendations.copy()
        st.session_state.last_alternative_update = time.time()
        st.rerun()
    
    # Add navigation buttons
    cols = st.columns([2, 2, 2])
    with cols[0]:
        if st.button("← Previous", use_container_width=True):
            st.session_state.dashboard_step = 2
            st.rerun()
    with cols[2]:
        if st.button("Approve & View Dashboard", use_container_width=True, type="primary"):
            # Debug before updating
            print(f"\n====== APPROVE BUTTON DEBUG ======")
            print(f"current_recommendations shape: {st.session_state.current_recommendations.shape}")
            print(f"current_recommendations index: {st.session_state.current_recommendations.index.tolist()}")
            if len(st.session_state.current_recommendations) > 0:
                print(f"First recommendation: {st.session_state.current_recommendations.iloc[0]['Name']}")
            print(f"======= END APPROVE DEBUG =======\n")
            
            # Update the final visualizations before advancing
            # Ensure we only have 5 visualizations
            if len(st.session_state.current_recommendations) > 5:
                print(f"WARNING: Trimming visualizations from {len(st.session_state.current_recommendations)} to 5")
                st.session_state.viz_recommendations = st.session_state.current_recommendations.iloc[:5]
            else:
                st.session_state.viz_recommendations = st.session_state.current_recommendations
                
            # Debug after updating
            print(f"\n====== APPROVE BUTTON AFTER DEBUG ======")
            print(f"viz_recommendations shape: {st.session_state.viz_recommendations.shape}")
            print(f"viz_recommendations index: {st.session_state.viz_recommendations.index.tolist()}")
            if len(st.session_state.viz_recommendations) > 0:
                print(f"First recommendation: {st.session_state.viz_recommendations.iloc[0]['Name']}")
            print(f"======= END APPROVE AFTER DEBUG =======\n")
            
            advance_step()
            st.rerun()
    
    # Initialize show_all_viz if not present
    if 'show_all_viz' not in st.session_state:
        st.session_state.show_all_viz = False

def dashboard_layout_step():
    """Show the final dashboard with metrics at the top and visualizations below."""
    # This should only run when we're on step 4
    if st.session_state.dashboard_step != 4:
        return
        
    if 'filtered_df' not in st.session_state or st.session_state.filtered_df is None:
        if "cleaned_dataframes" in st.session_state and st.session_state.cleaned_dataframes:
            df_key = list(st.session_state.cleaned_dataframes.keys())[0]
            df = st.session_state.cleaned_dataframes[df_key]
        elif "dataframes" in st.session_state and st.session_state.dataframes:
            df_key = list(st.session_state.dataframes.keys())[0]
            df = st.session_state.dataframes[df_key]
        elif "dashboard_uploaded_df" in st.session_state:
            df = st.session_state.dashboard_uploaded_df
        else:
            st.error("No data found. Please return to previous steps.")
            return
    else:
        df = st.session_state.filtered_df
    
    # Add enhanced dashboard CSS for a professional look
    st.markdown("""
    <style>
    /* Dashboard container styling */
    .dashboard-container {
        padding: 0;
        background-color: transparent;
        margin-bottom: 10px; /* Reduced from 20px */
    }
    
    /* Metrics panel styling - horizontal layout */
    .metrics-panel {
        display: flex;
        flex-direction: row;
        flex-wrap: nowrap;
        justify-content: space-between;
        gap: 10px; /* Reduced from 15px */
        background-color: transparent;
        padding: 0;
        margin-bottom: 10px; /* Reduced from 20px */
        width: 100%;
    }
    
    /* Section titles */
    .section-title {
        font-size: 18px;
        font-weight: 600;
        margin: 8px 0; /* Reduced from 10px 0 */
        padding-bottom: 5px; /* Reduced from 8px */
        border-bottom: 2px solid #f0f2f6;
    }
    
    /* Individual visualization container */
    .viz-container {
        background-color: transparent;
        border-radius: 0;
        padding: 0;
        box-shadow: none;
        margin-bottom: 0;
    }
    
    .viz-container:hover {
        transform: none;
        box-shadow: none;
    }
    
    /* Make plot titles centered */
    .js-plotly-plot .plotly .main-svg .infolayer .g-gtitle {
        text-anchor: middle !important;
    }
    
    /* Metric card styling */
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 12px 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
        flex: 1 1 0;
        min-width: 0;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .metric-card.mean-bg {
        border-left-color: #2196F3;
    }
    
    .metric-card.sum-bg {
        border-left-color: #FF9800;
    }
    
    .metric-card.count-bg {
        border-left-color: #9C27B0;
    }
    
    .metric-label {
        font-size: 14px;
        color: #666;
        margin-bottom: 5px;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 600;
        color: #333;
    }
    
    .metric-desc {
        font-size: 12px;
        color: #888;
        margin-top: 5px;
    }
    </style>
    
    <div class="dashboard-container">
    """, unsafe_allow_html=True)
    
    # Display metrics in a horizontal panel at the top with no title
    if 'top3' in st.session_state and st.session_state.top3:
        # Create columns for metrics - one column per metric
        metric_cols = st.columns(len(st.session_state.top3))
        
        for i, metric in enumerate(st.session_state.top3):
            column = metric['Column']
            agg_type = metric['Suggested_Aggregation']
            
            # Skip if column doesn't exist in dataframe
            if column not in df.columns:
                continue
                
            # Calculate metric value
            try:
                if agg_type == 'mean':
                    value = df[column].mean()
                    bg_class = "mean-bg"
                    description = f"Average value across {len(df)} records"
                elif agg_type == 'sum':
                    value = df[column].sum()
                    bg_class = "sum-bg"
                    description = f"Total sum across {len(df)} records"
                elif agg_type == 'count':
                    value = df[column].nunique()
                    bg_class = "count-bg"
                    description = f"Number of unique values"
                else:
                    continue
                    
                # Format number for display
                if isinstance(value, (int, float)):
                    if value >= 1000000:
                        formatted_value = f"{value/1000000:.1f}M"
                    elif value >= 1000:
                        formatted_value = f"{value/1000:.1f}K"
                    elif isinstance(value, float):
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                else:
                    formatted_value = str(value)
                
                # Display the metric in its column
                with metric_cols[i]:
                    st.markdown(f"""
                    <div class="metric-card {bg_class}">
                        <div class="metric-label">{agg_type.capitalize()} of {column}</div>
                        <div class="metric-value">{formatted_value}</div>
                        <div class="metric-desc">{description}</div>
                    </div>
                    """, unsafe_allow_html=True)
            except:
                # Skip metrics that can't be calculated
                pass
    
    # Add a divider line under metrics
    st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    
    # Display visualizations from recommendations - show all of them
    if 'viz_recommendations' in st.session_state and not st.session_state.viz_recommendations.empty:
        print(f"\n---------- BEGIN DASHBOARD VISUALIZATION RECOMMENDATIONS ----------")
        # (Keep debug printing)
        print(f"---------- END DASHBOARD VISUALIZATION RECOMMENDATIONS ----------\n")
        
        # Create the dashboard layout with left and right sections (Restore)
        left_col, right_col = st.columns([2, 1])
        
        viz_list = []
        # Ensure we only have 5 recommendations (Keep validation)
        if len(st.session_state.viz_recommendations) > 5:
            viz_recommendations = st.session_state.viz_recommendations.iloc[:5]
        else:
            viz_recommendations = st.session_state.viz_recommendations
            
        for idx, row in viz_recommendations.iterrows():
            viz_dict = row.to_dict()
            # (Keep viz_dict preparation logic as modified before - trusting Step 3 columns)
            if 'Recommended Visualization' in viz_dict and 'vis_type' not in viz_dict: viz_dict['vis_type'] = viz_dict['Recommended Visualization']
            if 'columns' not in viz_dict or viz_dict['columns'] is None or not isinstance(viz_dict['columns'], list):
                 viz_dict['columns'] = extract_columns_from_name(row, df)
            else:
                 original_columns = viz_dict['columns']
                 valid_columns = [col for col in original_columns if col in df.columns]
                 if len(valid_columns) != len(original_columns): viz_dict['columns'] = valid_columns
            if not viz_dict.get('columns'): viz_dict['columns'] = [df.columns[0]] if len(df.columns) > 0 else []
            
            viz_list.append(viz_dict)
            
        # Ensure we have exactly 5 visualizations for the layout (Restore)
        if len(viz_list) == 5:
            # Calculate heights (Restore)
            left_viz_height = 350  
            right_viz_height = 700 
            
            # Add minimal CSS (Keep existing or restore original)
            st.markdown("""
            <style>
            /* ... Keep styles ... */
            </style>
            """, unsafe_allow_html=True)
            
            # Left column - top row (Restore)
            with left_col:
                top_cols = st.columns(2)
                for col_idx, rec_idx in enumerate([0, 1]):
                    with top_cols[col_idx]:
                        create_viz_container(df, viz_list[rec_idx], height=left_viz_height, chart_id=f"left_top_{col_idx}")
            
                # Left column - bottom row (Restore)
                bottom_cols = st.columns(2)
                for col_idx, rec_idx in enumerate([2, 3]):
                    with bottom_cols[col_idx]:
                        create_viz_container(df, viz_list[rec_idx], height=left_viz_height, chart_id=f"left_bottom_{col_idx}")
            
            # Right column - single visualization (Restore)
            with right_col:
                create_viz_container(df, viz_list[4], height=right_viz_height, chart_id="right_main")
        else:
             st.warning("Could not display dashboard layout: Incorrect number of visualizations.")
             
    else:
        st.warning("No visualization recommendations available. Please complete step 3 first.")
    
    # Close dashboard container
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add navigation buttons (Keep)
    cols = st.columns([2, 2, 2])
    with cols[0]:
        if st.button("← Previous", use_container_width=True):
            st.session_state.dashboard_step = 3
            st.rerun()

def create_visualization(df, viz_type, columns, is_pair=False, rec=None):
    """Create a plotly visualization based on visualization type and columns..."""
    try:
        print(f"\n---------- BEGIN CREATE_VISUALIZATION ----------")
        print(f"DEBUG - Requested viz_type: '{viz_type}', is_pair: {is_pair}")
        print(f"DEBUG - Columns provided: {columns}")
        print(f"DEBUG - Recommendation type: {rec.get('Type') if rec else 'None'}")

        # --- Input Validation ---
        if not isinstance(viz_type, str):
            print(f"WARNING: viz_type is not a string: {viz_type} (type: {type(viz_type)}). Converting.")
            viz_type = str(viz_type)

        if not isinstance(columns, list):
            print(f"WARNING: columns parameter is not a list: {type(columns)}. Setting to empty.")
            columns = []

        # Ensure all columns exist in the dataframe and are usable
        original_columns_input = columns.copy()
        columns = [col for col in columns if col in df.columns]
        if len(columns) != len(original_columns_input):
            missing = set(original_columns_input) - set(columns)
            print(f"WARNING: Some provided columns don't exist in dataframe: {missing}. Using valid subset: {columns}")

        if not columns:
            print(f"ERROR: No valid columns available for visualization.")
            raise ValueError("No valid columns provided for visualization.")

        # --- Determine Target Visualization Type ---
        exact_viz_map = {
            "Scatter Plot Matrix (Splom)": "scatter_matrix",
            "Scatter Plot Matrix": "scatter_matrix",
            "Splom": "scatter_matrix",
            "Scatter Matrix": "scatter_matrix",
            "Scatter Plot (px.scatter)": "scatter",
            "Scatter Plot": "scatter",
            "Scatter Plot with Colors (px.scatter)": "scatter_color",
            "Scatter Plot with Color (px.scatter)": "scatter_color",
            "Bar Chart (px.bar)": "bar",
            "Bar Chart": "bar",
            "Grouped Bar Chart (px.bar)": "bar_grouped",
            "Grouped Bar Chart": "bar_grouped",
            "Stacked Bar Chart (px.bar)": "bar_stacked",
            "Stacked Bar Chart": "bar_stacked",
            "Pie Chart (px.pie)": "pie",
            "Pie Chart": "pie",
            "Sunburst Chart (px.sunburst)": "sunburst",
            "Sunburst (px.sunburst)": "sunburst",
            "Sunburst Chart": "sunburst",
            "Treemap (px.treemap)": "treemap",
            "Treemap": "treemap",
            "Line Chart (px.line)": "line",
            "Line Chart": "line",
            "Line Plot (px.line)": "line",
            "Multi-line Chart (px.line)": "line_multi",
            "Area Chart (px.area)": "area",
            "Area Chart": "area",
            "Stacked Area Chart (px.area)": "area_stacked",
            "Box Plot (px.box)": "box",
            "Box Plot": "box",
            "Strip Plot (px.box with points)": "box_strip",
            "Violin Plot (px.violin)": "violin",
            "Violin Plot": "violin",
            "Histogram (px.histogram)": "histogram",
            "Histogram": "histogram",
            "Heatmap (px.imshow)": "heatmap",
            "Heatmap (px.heatmap)": "heatmap",
            "Heatmap": "heatmap",
            "Density Heatmap (px.density_heatmap)": "density_heatmap",
            "Calendar Heatmap (px.heatmap)": "calendar_heatmap",
            "Calendar Heatmap": "calendar_heatmap",
            "Choropleth Map (px.choropleth)": "choropleth",
            "Choropleth Map": "choropleth",
            "Parallel Categories (px.parallel_categories)": "parallel_categories",
            "Parallel Categories": "parallel_categories",
            "Candlestick Chart (px.candlestick)": "candlestick",
            "Candlestick Chart": "candlestick",
            "Table (go.Table)": "table",
            "Timeline (px.scatter)": "timeline",
            "Timeline": "timeline",
            "3D Scatter Plot (px.scatter3d)": "scatter3d",
            "3D Scatter Plot": "scatter3d",
            "Animated Scatter (px.scatter with frames)": "scatter_animated",
            "Animated Scatter": "scatter_animated",
            "Faceted Scatter Plots (px.subplots)": "scatter_faceted",
            "Triple Visualization": "triple" # Added triple type
        }

        simple_viz_type = None
        if viz_type in exact_viz_map:
            simple_viz_type = exact_viz_map[viz_type]
            print(f"DEBUG - Using exact mapping for '{viz_type}' -> '{simple_viz_type}'")
        else:
            # Fallback to keyword extraction if no exact match
            viz_lower = viz_type.lower()
            if 'scatter' in viz_lower:
                if 'matrix' in viz_lower or 'splom' in viz_lower: simple_viz_type = 'scatter_matrix'
                elif '3d' in viz_lower: simple_viz_type = 'scatter3d'
                elif 'animate' in viz_lower or 'frame' in viz_lower: simple_viz_type = 'scatter_animated'
                elif 'facet' in viz_lower: simple_viz_type = 'scatter_faceted'
                elif 'color' in viz_lower: simple_viz_type = 'scatter_color'
                else: simple_viz_type = 'scatter'
            elif 'bar' in viz_lower:
                if 'grouped' in viz_lower: simple_viz_type = 'bar_grouped'
                elif 'stacked' in viz_lower: simple_viz_type = 'bar_stacked'
                else: simple_viz_type = 'bar'
            elif 'line' in viz_lower: simple_viz_type = 'line'
            elif 'area' in viz_lower: simple_viz_type = 'area'
            elif 'heatmap' in viz_lower: simple_viz_type = 'heatmap'
            elif 'pie' in viz_lower: simple_viz_type = 'pie'
            elif 'box' in viz_lower: simple_viz_type = 'box'
            elif 'hist' in viz_lower: simple_viz_type = 'histogram'
            elif 'violin' in viz_lower: simple_viz_type = 'violin'
            elif 'tree' in viz_lower: simple_viz_type = 'treemap'
            elif 'sunburst' in viz_lower: simple_viz_type = 'sunburst'
            elif 'triple' in viz_lower: simple_viz_type = 'triple' # Added triple keyword
            # Add more keyword checks here if needed

            if simple_viz_type:
                 print(f"DEBUG - Inferred viz type from keywords: '{simple_viz_type}'")
            else:
                 # If still no type, use a default based on column count
                 print(f"WARNING: Could not determine viz type from '{viz_type}'. Using default.")
                 if len(columns) == 1: simple_viz_type = 'histogram'
                 elif len(columns) == 2: simple_viz_type = 'scatter'
                 elif len(columns) >= 3: simple_viz_type = 'scatter_matrix'
                 else: raise ValueError("Cannot determine default viz type.")

        # --- Create Visualization based on Determined Type ---
        fig = None
        col1 = columns[0] if len(columns) > 0 else None
        col2 = columns[1] if len(columns) > 1 else None
        col3 = columns[2] if len(columns) > 2 else None

        print(f"DEBUG - Attempting to create: '{simple_viz_type}' with columns: {columns}")

        try:
            # --- Handle Specific Visualization Types --- 
            if simple_viz_type == "histogram":
                if col1: fig = px.histogram(df, x=col1, title=f"Distribution of {col1}")
                else: raise ValueError("Histogram requires at least one column.")

            elif simple_viz_type == "bar":
                if col1 and col2:
                    is_cat1 = not pd.api.types.is_numeric_dtype(df[col1]); is_num1 = not is_cat1
                    is_cat2 = not pd.api.types.is_numeric_dtype(df[col2]); is_num2 = not is_cat2
                    if is_cat1 and is_num2: fig = px.bar(df, x=col1, y=col2, title=f"{col2} by {col1}")
                    elif is_num1 and is_cat2: fig = px.bar(df, x=col2, y=col1, title=f"{col1} by {col2}")
                    else:
                        agg_func = 'sum' if is_num1 and is_num2 else 'count'
                        grouped = df.groupby(col1)[col2].agg(agg_func).reset_index()
                        fig = px.bar(grouped, x=col1, y=col2, title=f"{agg_func.capitalize()} of {col2} by {col1}")
                elif col1:
                    value_counts = df[col1].value_counts().reset_index()
                    value_counts.columns = ['category', 'count']
                    fig = px.bar(value_counts, x='category', y='count', title=f"Count by {col1}")
                else: raise ValueError("Bar chart requires at least one column.")

            elif simple_viz_type == "scatter":
                 if col1 and col2: fig = px.scatter(df, x=col1, y=col2, title=f"{col2} vs {col1}")
                 else: raise ValueError("Scatter plot requires two columns.")

            elif simple_viz_type == "scatter_color":
                 if col1 and col2 and col3:
                     fig = px.scatter(df, x=col1, y=col2, color=col3, title=f"{col2} vs {col1} by {col3}")
                 elif col1 and col2:
                      print("WARNING: scatter_color needs 3 columns, falling back to scatter.")
                      fig = px.scatter(df, x=col1, y=col2, title=f"{col2} vs {col1}")
                 else: raise ValueError("Scatter plot with color requires at least two columns, preferably three.")

            elif simple_viz_type == "line":
                 if col1 and col2:
                     is_temporal1 = pd.api.types.is_datetime64_any_dtype(df[col1]) or is_temporal_column(df, col1)
                     is_temporal2 = pd.api.types.is_datetime64_any_dtype(df[col2]) or is_temporal_column(df, col2)
                     x_col, y_col = (col1, col2) if is_temporal1 or not is_temporal2 else (col2, col1)
                     print(f"DEBUG - Line chart using x={x_col}, y={y_col}")
                     df_copy = df.copy()
                     if is_temporal_column(df_copy, x_col):
                         df_copy[x_col] = convert_to_datetime(df_copy[x_col])
                     if pd.api.types.is_datetime64_any_dtype(df_copy[x_col]) or pd.api.types.is_numeric_dtype(df_copy[x_col]):
                         df_copy = df_copy.sort_values(x_col)
                     fig = px.line(df_copy, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                 else: raise ValueError("Line chart requires two columns.")

            elif simple_viz_type == "box":
                 if col1 and col2:
                     is_cat1 = not pd.api.types.is_numeric_dtype(df[col1]); is_num1 = not is_cat1
                     is_cat2 = not pd.api.types.is_numeric_dtype(df[col2]); is_num2 = not is_cat2
                     if is_cat1 and is_num2: fig = px.box(df, x=col1, y=col2, title=f"Distribution of {col2} by {col1}")
                     elif is_num1 and is_cat2: fig = px.box(df, x=col2, y=col1, title=f"Distribution of {col1} by {col2}")
                     else:
                         y_col = col1 if is_num1 else (col2 if is_num2 else None)
                         if y_col:
                             print(f"WARNING: Boxplot needs Cat/Num pair. Plotting distribution of {y_col}.")
                             fig = px.box(df, y=y_col, title=f"Distribution of {y_col}")
                         else: raise ValueError("Boxplot requires one numeric column, preferably with a categorical column.")
                 elif col1:
                     if pd.api.types.is_numeric_dtype(df[col1]): fig = px.box(df, y=col1, title=f"Box Plot of {col1}")
                     else: raise ValueError("Single column boxplot requires a numeric column.")
                 else: raise ValueError("Boxplot requires at least one numeric column.")

            elif simple_viz_type == "violin":
                 if col1 and col2:
                     is_cat1 = not pd.api.types.is_numeric_dtype(df[col1]); is_num1 = not is_cat1
                     is_cat2 = not pd.api.types.is_numeric_dtype(df[col2]); is_num2 = not is_cat2
                     if is_cat1 and is_num2: fig = px.violin(df, x=col1, y=col2, title=f"Distribution of {col2} by {col1}")
                     elif is_num1 and is_cat2: fig = px.violin(df, x=col2, y=col1, title=f"Distribution of {col1} by {col2}")
                     else:
                         y_col = col1 if is_num1 else (col2 if is_num2 else None)
                         if y_col:
                             print(f"WARNING: Violin needs Cat/Num pair. Plotting distribution of {y_col}.")
                             fig = px.violin(df, y=y_col, title=f"Distribution of {y_col}")
                         else: raise ValueError("Violin plot requires one numeric column, preferably with a categorical column.")
                 elif col1:
                     if pd.api.types.is_numeric_dtype(df[col1]): fig = px.violin(df, y=col1, title=f"Violin Plot of {col1}")
                     if pd.api.types.is_numeric_dtype(df[col1]):
                         fig = px.violin(df, y=col1, title=f"Violin Plot of {col1}")
                     else: raise ValueError("Single column violin plot requires a numeric column.")
                 else: raise ValueError("Violin plot requires at least one numeric column.")

            elif simple_viz_type == "heatmap":
                 if col1 and col2:
                     if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                         # Correlation heatmap for two numeric columns
                         corr_df = df[[col1, col2]].corr()
                         fig = px.imshow(corr_df, text_auto=True, title=f"Correlation between {col1} and {col2}")
                     else:
                         # Crosstab heatmap for categorical columns
                         crosstab = pd.crosstab(df[col1], df[col2])
                         fig = px.imshow(crosstab, text_auto=True, title=f"Heatmap of {col1} vs {col2}")
                 else: raise ValueError("Heatmap requires two columns.")

            elif simple_viz_type == "pie":
                 if col1:
                     # Limit categories for readability
                     if df[col1].nunique() <= 20: 
                         value_counts = df[col1].value_counts().reset_index()
                         value_counts.columns = ['category', 'count']
                         fig = px.pie(value_counts, names='category', values='count', title=f"Pie Chart of {col1}")
                     else:
                         print(f"WARNING: Too many categories for pie chart ({df[col1].nunique()}). Using bar chart.")
                         value_counts = df[col1].value_counts().nlargest(20).reset_index()
                         value_counts.columns = ['category', 'count']
                         fig = px.bar(value_counts, x='category', y='count', title=f"Top 20 Categories of {col1}")
                 else: raise ValueError("Pie chart requires one column.")

            elif simple_viz_type == "scatter_matrix":
                if len(columns) >= 2:
                    fig = px.scatter_matrix(df, dimensions=columns, title=f"Scatter Plot Matrix")
                else: raise ValueError("Scatter matrix requires at least two columns.")
            
            elif simple_viz_type == "treemap":
                if col1 and col2: # Path + Values
                    if not pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                         grouped = df.groupby(col1)[col2].sum().reset_index()
                         fig = px.treemap(grouped, path=[px.Constant("all"), col1], values=col2, title=f"Treemap of {col2} by {col1}")
                    elif pd.api.types.is_numeric_dtype(df[col1]) and not pd.api.types.is_numeric_dtype(df[col2]): # Swap if needed
                         grouped = df.groupby(col2)[col1].sum().reset_index()
                         fig = px.treemap(grouped, path=[px.Constant("all"), col2], values=col1, title=f"Treemap of {col1} by {col2}")
                    elif not pd.api.types.is_numeric_dtype(df[col1]) and not pd.api.types.is_numeric_dtype(df[col2]): # Two categoricals (hierarchy)
                         if col3 and pd.api.types.is_numeric_dtype(df[col3]): # Use third numeric as value
                              fig = px.treemap(df, path=[px.Constant("all"), col1, col2], values=col3, title=f"Treemap of {col3} by {col1}, {col2}")
                         else: # Use counts
                              grouped = df.groupby([col1, col2]).size().reset_index(name='count')
                              fig = px.treemap(grouped, path=[px.Constant("all"), col1, col2], values='count', title=f"Treemap of Counts by {col1}, {col2}")
                    else: # Fallback to single path if types dont match well
                         print("WARNING: Treemap combination not ideal, using single path.")
                         value_counts = df[col1].value_counts().reset_index()
                         value_counts.columns = ['category', 'count']
                         fig = px.treemap(value_counts, path=[px.Constant("all"), 'category'], values='count', title=f"Treemap of {col1}")
                elif col1: # Single column treemap (counts)
                     value_counts = df[col1].value_counts().reset_index()
                     value_counts.columns = ['category', 'count']
                     fig = px.treemap(value_counts, path=[px.Constant("all"), 'category'], values='count', title=f"Treemap of {col1}")
                else: raise ValueError("Treemap requires at least one column.")
            
            elif simple_viz_type == "triple":
                 if col1 and col2 and col3:
                     print(f"DEBUG - Calling visualize_triple for {col1}, {col2}, {col3}")
                     # Ensure visualize_triple exists and handles the plotting
                     if 'visualize_triple' in globals():
                         fig = visualize_triple(df, col1, col2, col3)
                     else:
                         print("ERROR: visualize_triple function not found. Falling back to 3D scatter.")
                         fig = px.scatter_3d(df, x=col1, y=col2, z=col3, title=f"{col1} vs {col2} vs {col3}")
                 else: raise ValueError("Triple visualization requires three columns.")

            # --- Add more specific types as needed --- 
            
            # elif simple_viz_type == "some_other_type":
            #    ... implementation ...

            else:
                # If the specific type is not implemented above, raise error to trigger fallback
                raise NotImplementedError(f"Visualization type '{simple_viz_type}' is recognized but not implemented.")

        except (ValueError, NotImplementedError, Exception) as e:
            print(f"WARNING: Failed to create requested visualization '{simple_viz_type}': {str(e)}. Attempting fallback.")
            
            # --- Fallback Logic --- 
            try:
                if len(columns) == 1:
                    col = columns[0]
                    print(f"DEBUG - Fallback: Single column ({col})")
                    if pd.api.types.is_numeric_dtype(df[col]):
                        fig = px.histogram(df, x=col, title=f"Distribution of {col} (Fallback)")
                    else:
                        value_counts = df[col].value_counts().reset_index()
                        value_counts.columns = ['category', 'count']
                        fig = px.bar(value_counts, x='category', y='count', title=f"Counts for {col} (Fallback)")
                elif len(columns) == 2:
                    col1, col2 = columns[0], columns[1]
                    print(f"DEBUG - Fallback: Two columns ({col1}, {col2})")
                    is_num1 = pd.api.types.is_numeric_dtype(df[col1]); is_cat1 = not is_num1
                    is_num2 = pd.api.types.is_numeric_dtype(df[col2]); is_cat2 = not is_num2
                    if is_num1 and is_num2: # Num x Num -> Scatter
                        fig = px.scatter(df, x=col1, y=col2, title=f"{col2} vs {col1} (Fallback)")
                    elif (is_cat1 and is_num2) or (is_num1 and is_cat2): # Cat x Num -> Bar
                        x_ax, y_ax = (col1, col2) if is_cat1 else (col2, col1)
                        fig = px.bar(df, x=x_ax, y=y_ax, title=f"{y_ax} by {x_ax} (Fallback)")
                    else: # Cat x Cat -> Heatmap (count)
                        crosstab = pd.crosstab(df[col1], df[col2])
                        fig = px.imshow(crosstab, text_auto=True, title=f"Counts of {col1} vs {col2} (Fallback)")
                elif len(columns) >= 3:
                     print(f"DEBUG - Fallback: Three+ columns ({columns})")
                     # Use scatter matrix for 3+ columns
                     fig = px.scatter_matrix(df, dimensions=columns[:min(len(columns), 5)], title="Scatter Matrix (Fallback)") # Limit dimensions
                else:
                    # This case should be prevented by earlier checks
                    print("ERROR: Fallback failed - No columns.")
                    fig = None 

            except Exception as fallback_e:
                print(f"ERROR: Fallback visualization creation also failed: {fallback_e}")
                fig = None # Ensure fig is None if fallback fails

        # --- Final Check and Return ---
        if fig is None:
            print(f"ERROR: Could not create any visualization for type '{viz_type}' with columns {columns}")
            # Create an empty figure with error message
            fig = go.Figure()
            fig.add_annotation(text=f"Could not create visualization ({viz_type})",
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title="Visualization Unavailable")
        else:
             # Ensure title is set, using a generic one if needed
             if not fig.layout.title or not fig.layout.title.text:
                 fig.update_layout(title_text=f"Visualization of {', '.join(columns)}")
             print(f"---------- END CREATE_VISUALIZATION: SUCCESS ({simple_viz_type}) ----------\n")
             
        return fig

    except Exception as e:
        print(f"ERROR in create_visualization (Outer Scope): {str(e)}")
        import traceback
        traceback.print_exc()
        # Create an empty figure with error message
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="red"))
        print(f"---------- END CREATE_VISUALIZATION: ERROR ----------\n")
        return fig

def create_viz_container(df, rec, height=300, chart_id=None):
    """Create a visualization container with title and visualization"""
    try:
        # More detailed debug for diagnosing visualization issues
        print(f"\n---------- BEGIN CREATE_VIZ_CONTAINER ----------")
        print(f"DEBUG - Creating visualization for recommendation: {rec.get('Name')}")
        print(f"DEBUG - Recommendation type: {rec.get('Type')}")
        print(f"DEBUG - Raw recommendation columns: {rec.get('columns')}")
        print(f"DEBUG - Visualization type: {rec.get('vis_type')}")
        
        viz_type = rec.get('vis_type')
        is_pair = rec.get('Type') == 'Pair'
        columns = rec.get('columns', []) # Get columns directly from recommendation

        # Ensure vis_type is a string
        if not isinstance(viz_type, str):
            print(f"DEBUG - Converting non-string viz_type to string: {viz_type} (type: {type(viz_type)})")
            viz_type = str(viz_type)

        # Ensure columns is a list and exists (minimal check)
        if not isinstance(columns, list):
            print(f"WARNING: Columns in recommendation is not a list: {type(columns)}. Attempting extraction.")
            columns = extract_columns_from_name(rec, df)
            
        # Verify columns exist in the dataframe (minimal check)
        original_columns = columns.copy()
        columns = [col for col in columns if col in df.columns]
        if len(columns) != len(original_columns):
            missing = set(original_columns) - set(columns)
            print(f"WARNING: Some recommended columns are missing from DataFrame: {missing}. Using valid subset: {columns}")
        
        # If no columns remain after validation, cannot proceed
        if not columns:
            print(f"ERROR: No valid columns available for visualization: {rec.get('Name')}")
            st.error(t(f"Could not display visualization for {rec.get('Name')}: No valid columns found."))
            print(f"---------- END CREATE_VIZ_CONTAINER: FAILED (NO VALID COLUMNS) ----------\n")
            return False
            
        print(f"DEBUG - Using columns from recommendation: {columns}")
        
        container_id = f"viz_container_{chart_id}" if chart_id else f"viz_container_{randint(1000, 9999)}"
        
        viz = create_visualization(df, viz_type, columns, is_pair=is_pair, rec=rec)
        
        if viz is not None:
            # Restore original layout update logic based on chart_id
            if 'right_main' in str(chart_id): # Use chart_id for check
                viz.update_layout(
                    height=height, 
                    margin=dict(t=40, b=30, l=30, r=30), # Original right margins 
                    autosize=True,
                    title_font_size=16, # Original right font size
                    title_y=1.0,        
                    title={'text': viz.layout.title.text, 'x': 0.5}, 
                    title_xanchor='center' 
                )
            else: # For left plots
                viz.update_layout(
                    height=height,
                    margin=dict(t=40, b=20, l=25, r=25), # Original left margins
                    autosize=True,
                    title_font_size=14, # Original left font size
                    title_y=1.0,        
                    title={'text': viz.layout.title.text, 'x': 0.5},
                    title_xanchor='center' 
                )
            st.plotly_chart(viz, use_container_width=True, key=container_id)
            print(f"---------- END CREATE_VIZ_CONTAINER: SUCCESS ----------\n")
            return True
        else:
            st.error(t(f"Could not create {viz_type} with columns {', '.join(columns)}"))
            print(f"---------- END CREATE_VIZ_CONTAINER: FAILED (NULL VISUALIZATION) ----------\n")
            return False
    except Exception as e:
        # ... (Keep existing exception handling) ...
        return False

def main():
    # Display step progress at the top
    step_col1, step_col2, step_col3, step_col4 = st.columns(4)
    with step_col1:
        st.markdown(f"### {t('🔵' if st.session_state.dashboard_step == 1 else '✅')} {t('Step 1: Domain')}")
    with step_col2:
        if st.session_state.dashboard_step >= 2:
            st.markdown(f"### {t('🔵' if st.session_state.dashboard_step == 2 else '✅')} {t('Step 2: Metrics')}")
        else:
            st.markdown(f"### {t('⚪')} {t('Step 2: Metrics')}")
    with step_col3:
        if st.session_state.dashboard_step >= 3:
            st.markdown(f"### {t('🔵' if st.session_state.dashboard_step == 3 else '✅')} {t('Step 3: Visualization')}")
        else:
            st.markdown(f"### {t('⚪')} {t('Step 3: Visualization')}")
    with step_col4:
        if st.session_state.dashboard_step >= 4:
            st.markdown(f"### {t('🔵')} {t('Step 4: Dashboard')}")
        else:
            st.markdown(f"### {t('⚪')} {t('Step 4: Dashboard')}")
    
    # Simple progress bar
    st.progress(st.session_state.dashboard_step / 4)
    
    # Display the current step
    if st.session_state.dashboard_step == 1:
        domain_step()
    elif st.session_state.dashboard_step == 2:
        metrics_recommendation_step()
    elif st.session_state.dashboard_step == 3:
        visualization_recommendation_step()
    elif st.session_state.dashboard_step == 4:
        dashboard_layout_step()

if __name__ == "__main__":
    main() 