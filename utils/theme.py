import streamlit as st

# Visualization theme with color palette
viz_theme = {
    'primary': '#3366CC',    # Main blue
    'secondary': '#FF9933',  # Orange
    'tertiary': '#33AA33',   # Green
    'light': '#E6EEF8',      # Light blue
    'dark': '#0D2B4D',       # Dark blue
    'accent': '#CC3366',     # Pink/Purple
    'palette': [
        '#3366CC',  # Blue
        '#FF9933',  # Orange
        '#33AA33',  # Green
        '#CC3366',  # Pink/Purple
        '#9966CC',  # Purple
        '#FF6666',  # Red
        '#66CCCC',  # Teal
        '#FFCC33',  # Yellow
        '#666699',  # Slate
        '#CC9966',  # Brown
    ]
}

def get_theme():
    """Return the visualization theme dictionary"""
    return viz_theme

def apply_custom_theme():
    """Apply custom SaaS theme to the Streamlit app"""
    st.markdown("""
        <style>
        /* Global styling */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Main container styling */
        .main {
            background-color: #F9FAFB;
            padding: 2rem 3rem;
        }
        
        /* Custom container for cards */
        .stCard {
            background-color: white;
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            margin-bottom: 1.5rem;
            border: 1px solid #F3F4F6;
            transition: all 0.2s ease;
        }
        
        .stCard:hover {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
            transform: translateY(-2px);
        }
        
        /* Modern button styling */
        .stButton > button {
            background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
            color: white !important;
            border: none;
            border-radius: 0.5rem;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            letter-spacing: 0.025em;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
            color: white !important;
        }
        
        .stButton > button:active {
            transform: translateY(0px);
            color: white !important;
        }
        
        /* Secondary button */
        .secondary-button button {
            background: white !important;
            color: #4F46E5 !important;
            border: 1px solid #E5E7EB !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }
        
        .secondary-button button:hover {
            background: #F9FAFB !important;
            border-color: #D1D5DB !important;
        }
        
        /* File uploader styling */
        .uploadedFile {
            border: 2px dashed #4F46E5;
            border-radius: 1rem;
            padding: 2rem;
            text-align: center;
            background: rgba(79, 70, 229, 0.05);
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .uploadedFile:hover {
            background: rgba(79, 70, 229, 0.1);
            border-color: #7C3AED;
        }
        
        /* Header styling */
        h1 {
            color: #111827;
            font-weight: 800;
            letter-spacing: -0.025em;
            font-size: 2.25rem;
            margin-bottom: 0.5rem;
        }
        
        h2 {
            color: #1F2937;
            font-weight: 700;
            letter-spacing: -0.025em;
            font-size: 1.875rem;
        }
        
        h3 {
            color: #1F2937;
            font-weight: 600;
            font-size: 1.5rem;
        }
        
        h4 {
            color: #374151;
            font-weight: 600;
            font-size: 1.25rem;
        }
        
        p {
            color: #4B5563;
            line-height: 1.625;
        }
        
        /* Success message styling */
        .success-message, .stSuccess {
            background-color: #ECFDF5;
            color: #065F46;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #10B981;
            margin: 1rem 0;
            display: flex;
            align-items: center;
        }
        
        /* Error message styling */
        .error-message, .stError {
            background-color: #FEF2F2;
            color: #991B1B;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #EF4444;
            margin: 1rem 0;
            display: flex;
            align-items: center;
        }
        
        /* Info message styling */
        .info-message, .stInfo {
            background-color: #EFF6FF;
            color: #1E40AF;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #3B82F6;
            margin: 1rem 0;
            display: flex;
            align-items: center;
        }
        
        /* Warning message styling */
        .warning-message, .stWarning {
            background-color: #FFFBEB;
            color: #92400E;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #F59E0B;
            margin: 1rem 0;
            display: flex;
            align-items: center;
        }
        
        /* Sidebar styling */
        .css-1d391kg, [data-testid="stSidebar"] {
            background-color: white;
            border-right: 1px solid #E5E7EB;
        }
        
        /* Custom upload zone */
        .upload-zone {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 250px;
            border: 2px dashed #D1D5DB;
            border-radius: 0.75rem;
            background-color: white;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .upload-zone:hover {
            border-color: #4F46E5;
            background-color: #F9FAFB;
        }
        
        .upload-zone-icon {
            font-size: 3rem;
            color: #6B7280;
            margin-bottom: 1rem;
        }
        
        /* Data table styling */
        .dataframe {
            border-collapse: collapse;
            width: 100%;
            border-radius: 0.5rem;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .dataframe thead th {
            background-color: #F3F4F6;
            padding: 0.75rem 1rem;
            text-align: left;
            font-weight: 600;
            color: #374151;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .dataframe tbody tr {
            border-bottom: 1px solid #E5E7EB;
            transition: background-color 0.2s;
        }
        
        .dataframe tbody tr:hover {
            background-color: #F9FAFB;
        }
        
        .dataframe tbody td {
            padding: 0.75rem 1rem;
            font-size: 0.875rem;
            color: #4B5563;
        }
        
        /* Progress bar styling */
        .stProgress > div > div > div {
            background-color: #4F46E5;
        }
        
        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-weight: 700;
            color: #111827;
        }
        
        [data-testid="stMetricDelta"] {
            font-size: 0.875rem;
        }
        
        /* Badge styling */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        
        .badge-blue {
            background-color: #EFF6FF;
            color: #1E40AF;
        }
        
        .badge-green {
            background-color: #ECFDF5;
            color: #065F46;
        }
        
        .badge-red {
            background-color: #FEF2F2;
            color: #991B1B;
        }
        
        .badge-yellow {
            background-color: #FFFBEB;
            color: #92400E;
        }
        
        .badge-purple {
            background-color: #F5F3FF;
            color: #5B21B6;
        }
        
        /* Animated loading */
        @keyframes pulse {
            0%, 100% {
                opacity: 1;
            }
            50% {
                opacity: 0.5;
            }
        }
        
        .animate-pulse {
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        
        /* Custom tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1px;
            background-color: #F3F4F6;
            border-radius: 0.5rem;
            padding: 0.25rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 2.5rem;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 0.375rem;
            margin: 0.125rem;
            padding: 0.5rem 1rem;
            color: #4B5563;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: white !important;
            color: #4F46E5 !important;
            font-weight: 600;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }
        </style>
    """, unsafe_allow_html=True)

def section_header(title, description=None):
    """Display a styled section header with optional description"""
    st.markdown(f"<h2>{title}</h2>", unsafe_allow_html=True)
    if description:
        st.markdown(f"<p style='margin-top: -0.5rem; margin-bottom: 1.5rem; color: #6B7280;'>{description}</p>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 1.5rem 0; border: 0; height: 1px; background-color: #E5E7EB;'>", unsafe_allow_html=True)

def card(content, title=None, icon=None):
    """Display content in a styled card container"""
    card_html = "<div class='stCard'>"
    
    # Add header if title is provided
    if title:
        card_header = f"<div style='display: flex; align-items: center; margin-bottom: 1rem;'>"
        if icon:
            card_header += f"<div style='margin-right: 0.75rem; font-size: 1.5rem;'>{icon}</div>"
        card_header += f"<h3 style='margin: 0;'>{title}</h3></div>"
        card_html += card_header
    
    card_html += "</div>"
    
    # Render the card container
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Render the content inside the card
    with st.container():
        content()

def hero_section(title, subtitle):
    """Display a modern hero section"""
    st.markdown(f"""
        <div style='text-align: center; padding: 3rem 1rem; margin-bottom: 2rem;'>
            <h1 style='font-size: 3rem; margin-bottom: 1rem;'>{title}</h1>
            <p style='font-size: 1.25rem; color: #6B7280; max-width: 600px; margin: 0 auto;'>{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

def feature_card(icon, title, description, button_text=None, on_click=None):
    """Display a feature card with icon, title, description and optional button"""
    col1, col2, col3 = st.columns([1, 10, 2])
    
    with col1:
        st.markdown(f"<div style='font-size: 2rem; color: #4F46E5;'>{icon}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<h4 style='margin: 0; margin-bottom: 0.5rem;'>{title}</h4>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin: 0; color: #6B7280;'>{description}</p>", unsafe_allow_html=True)
    
    if button_text and on_click:
        with col3:
            st.button(button_text, on_click=on_click, key=f"feature_{title}")

def upload_zone(message="Drag and drop your CSV files here or click to browse"):
    """Display a modern upload zone"""
    st.markdown(f"""
        <div class='upload-zone'>
            <div class='upload-zone-icon'>📄</div>
            <p style='margin-bottom: 0.5rem; font-weight: 500; color: #374151;'>{message}</p>
            <p style='margin: 0; font-size: 0.875rem; color: #6B7280;'>Supports CSV files up to 200MB</p>
        </div>
    """, unsafe_allow_html=True)

def badge(text, color="blue"):
    """Display a styled badge with text"""
    return f"<span class='badge badge-{color}'>{text}</span>"

def stat_card(value, label, delta=None, delta_color="normal"):
    """Display a stat card with value, label and optional delta"""
    delta_html = ""
    if delta:
        icon = "↑" if delta_color == "positive" else "↓" if delta_color == "negative" else ""
        color = "#10B981" if delta_color == "positive" else "#EF4444" if delta_color == "negative" else "#6B7280"
        delta_html = f"<p style='margin: 0; font-size: 0.875rem; color: {color};'>{icon} {delta}</p>"
    
    st.markdown(f"""
        <div style='background-color: white; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); border: 1px solid #F3F4F6;'>
            <p style='margin: 0; margin-bottom: 0.5rem; font-size: 0.875rem; font-weight: 500; color: #6B7280; text-transform: uppercase; letter-spacing: 0.05em;'>{label}</p>
            <h3 style='margin: 0; font-size: 1.875rem; font-weight: 700; color: #111827;'>{value}</h3>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)

def loading_placeholder(text="Loading..."):
    """Display a loading placeholder with animation"""
    return st.markdown(f"""
        <div style='display: flex; align-items: center; padding: 1rem; background-color: #F9FAFB; border-radius: 0.5rem; margin: 1rem 0;'>
            <div style='width: 1rem; height: 1rem; border-radius: 50%; background-color: #4F46E5; margin-right: 0.75rem;' class='animate-pulse'></div>
            <p style='margin: 0; color: #4B5563;'>{text}</p>
        </div>
    """, unsafe_allow_html=True)

def load_custom_theme():
    """Apply the custom theme to the app."""
    apply_custom_theme()
    return viz_theme 