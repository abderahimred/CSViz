import streamlit as st
import pandas as pd
import os
from utils.theme import apply_custom_theme, card, section_header
from utils.translations import get_translation
from utils.config import get_openai_api_key

# Initialize language in session state if not present
if "language" not in st.session_state:
    st.session_state.language = "en"  # Default to English

# Initialize OpenAI API key from environment
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = get_openai_api_key()
    if st.session_state.openai_api_key:
        st.session_state.openai_connected = True

# Function to translate text
def t(text_id):
    return get_translation(text_id, st.session_state.language)

# Set page config
st.set_page_config(
    page_title="CSViz",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom theme
apply_custom_theme()

# Add language selector at the top
language_col1, language_col2 = st.columns([6, 1])
with language_col2:
    selected_lang = st.selectbox(
        t('language_selector'),
        options=["English", "Français"],
        index=0 if st.session_state.language == "en" else 1,
        key="language_selector",
        label_visibility="collapsed"
    )
    
    # Update language when user changes selection
    if (selected_lang == "English" and st.session_state.language != "en") or \
       (selected_lang == "Français" and st.session_state.language != "fr"):
        st.session_state.language = "en" if selected_lang == "English" else "fr"
        st.rerun()

# Add custom CSS to force white text in buttons - note: we're adding these after the theme is applied
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
    
    /* Language selector styling */
    div[data-testid="stSelectbox"] {
        width: 120px;
        float: right;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}
if "cleaned_dataframes" not in st.session_state:
    st.session_state.cleaned_dataframes = {}
if "file_errors" not in st.session_state:
    st.session_state.file_errors = {}
if "total_rows_processed" not in st.session_state:
    st.session_state.total_rows_processed = 0
if "progress" not in st.session_state:
    st.session_state.progress = {'upload': False, 'process': False, 'clean': False, 'visualize': False}
# Reset data analysis state
if "data_analyzed" in st.session_state:
    st.session_state.data_analyzed = False
if "data_quality_scores" in st.session_state:
    st.session_state.data_quality_scores = {}

def handle_file_upload():
    """Handle file upload with improved validation and feedback"""
    files_changed = False
    all_success = True
    
    for file in st.session_state.uploaded_files_widget:
        try:
            if file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files[file.name] = file
                # Try to read the file and validate it's a proper CSV
                try:
                    df = pd.read_csv(file)
                    if len(df) == 0:
                        raise ValueError(t('empty_csv_file'))
                    # File is valid, add to dataframes
                    st.session_state.dataframes[file.name] = df
                    files_changed = True
                    # Update total rows processed count
                    st.session_state.total_rows_processed += len(df)
                    # Set upload progress to true when files are successfully added
                    st.session_state.progress['upload'] = True
                except Exception as e:
                    # Remove the file if it can't be processed
                    if file.name in st.session_state.uploaded_files:
                        del st.session_state.uploaded_files[file.name]
                    st.session_state.file_errors[file.name] = str(e)
                    all_success = False
        except Exception as e:
            st.error(f"{t('error_processing_file')} {file.name}: {str(e)}")
            all_success = False
    
    if files_changed:
        if all_success:
            st.success(f"✅ {len(st.session_state.uploaded_files_widget)} {t('files_uploaded_successfully')}")
        else:
            st.warning(t('some_files_had_errors'))

def get_file_size_str(file):
    """Get a human-readable file size string"""
    file_size = len(file.getvalue())
    if file_size < 1024:
        return f"{file_size} {t('bytes')}"
    elif file_size < 1024 * 1024:
        return f"{file_size / 1024:.1f} {t('kb')}"
    else:
        return f"{file_size / (1024 * 1024):.1f} {t('mb')}"

# A custom button using HTML/CSS for complete control
def custom_button(label, key=None, on_click=None):
    """Create a custom button with guaranteed white text"""
    # Generate a unique key if not provided
    if key is None:
        key = f"custom_btn_{label}"
    
    # Create a unique ID for the button
    button_id = f"btn_{key}"
    
    # Insert custom styled button HTML
    st.markdown(f"""
        <style>
        #{button_id} {{
            background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
            color: white !important;
            border: none;
            border-radius: 0.5rem;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            letter-spacing: 0.025em;
            cursor: pointer;
            width: 100%;
            text-align: center;
            font-family: 'Inter', sans-serif;
            font-size: 1rem;
        }}
        #{button_id}:hover {{
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        }}
        </style>
        <div id="{button_id}">{label}</div>
    """, unsafe_allow_html=True)
    
    # Return standard button which will be hidden but captures the click
    return st.button("Hidden Button", key=key, on_click=on_click, help=label, label_visibility="collapsed")

def main():
    # Show workflow progress as the first element, exactly matching other pages
    cols = st.columns(5)
    with cols[0]:
        st.markdown(f"#### 1. {t('upload')} 🔄")
    with cols[1]:
        st.markdown(f"#### 2. {t('Analysis')}")
    with cols[2]:
        st.markdown(f"#### 3. {t('process')}")
    with cols[3]:
        st.markdown(f"#### 4. {t('clean')}")
    with cols[4]:
        st.markdown(f"#### 5. {t('visualize')}")
    
    st.progress(0)  # 0% through the workflow
    
    # Main page title after progress bar
    st.markdown("# CSViz")
    st.markdown("---")
    
    # File upload section
    st.markdown(f"### {t('upload_csv_files')}")
    
    # File uploader
    uploaded_files = st.file_uploader(
        t('drop_files_here'), 
        type=["csv"], 
        accept_multiple_files=True,
        key="uploaded_files_widget",
        on_change=handle_file_upload
    )
    
    # Show errors if any
    for filename, error in st.session_state.file_errors.items():
        st.error(f"{t('error_with')} {filename}: {error}")
    
    # Display uploaded files as a simple table
    if st.session_state.dataframes:
        st.markdown("---")
        st.markdown(f"### {t('uploaded_files')}")
        
        # Create columns for the table header
        cols = st.columns([3, 2, 2, 1])
        cols[0].markdown(f"**{t('filename')}**")
        cols[1].markdown(f"**{t('size')}**")
        cols[2].markdown(f"**{t('rows_columns')}**")
        cols[3].markdown(f"**{t('action')}**")
        
        # Add a separator line
        st.markdown("<hr style='margin-top: 0; margin-bottom: 1rem'>", unsafe_allow_html=True)
        
        # Display file information in rows
        for filename, df in st.session_state.dataframes.items():
            file = st.session_state.uploaded_files[filename]
            
            cols = st.columns([3, 2, 2, 1])
            cols[0].write(filename)
            cols[1].write(get_file_size_str(file))
            cols[2].write(f"{len(df)} × {len(df.columns)}")
            
            # Delete button
            if cols[3].button("🗑️", key=f"delete_{filename}"):
                if filename in st.session_state.uploaded_files:
                    del st.session_state.uploaded_files[filename]
                if filename in st.session_state.dataframes:
                    del st.session_state.dataframes[filename]
                st.rerun()
        
        # Show action buttons based on number of files
        st.markdown("---")
        st.markdown(f"### {t('next_steps')}")
        
        # Show different buttons based on the number of files
        if len(st.session_state.dataframes) > 1:
            # For multiple files, show the Process button
            process_btn = st.button(
                f"🔗 {t('process_multiple_files')}", 
                use_container_width=True,
                key="process_btn",
                type="primary"
            )
            if process_btn:
                # Set progress flag
                st.session_state.progress['upload'] = True
                st.session_state.processing_mode = None  # Show method selection screen
                st.switch_page("pages/2_Process.py")
        else:
            # For a single file, show the Analyze button
            analyze_btn = st.button(
                "📊 Analyze Data Quality", 
                use_container_width=True,
                key="analyze_btn_white",
                type="primary"
            )
            if analyze_btn:
                # Reset analysis state to ensure fresh analysis
                if "data_analyzed" in st.session_state:
                    st.session_state.data_analyzed = False
                if "data_quality_scores" in st.session_state:
                    st.session_state.data_quality_scores = {}
                    
                # Set progress flag
                st.session_state.progress['upload'] = True
                st.switch_page("pages/1_Analysis.py")
    else:
        # Show instructions when no files are uploaded
        st.info(t('upload_instructions'))

if __name__ == "__main__":
    main() 