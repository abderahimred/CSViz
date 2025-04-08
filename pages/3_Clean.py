import streamlit as st
import pandas as pd
import numpy as np
import re
from word2number import w2n
from utils.theme import apply_custom_theme, hero_section, feature_card
from utils.cleaning import clean_data_basic
from utils.ai import clean_dataframe_with_ai
from utils.translations import get_translation_function

# Get translation function
t = get_translation_function()

# Initialize session state variables needed before page config
if "cleaning_mode" not in st.session_state:
    st.session_state.cleaning_mode = None
if "language" not in st.session_state:
    st.session_state.language = "en"  # Default to English

# Set page config - expand sidebar if in manual cleaning mode
initial_sidebar = "expanded" if st.session_state.cleaning_mode == "standard" else "collapsed"
st.set_page_config(
    page_title=f"{t('clean')} - Data Cleaning App",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state=initial_sidebar
)

# Apply custom theme
apply_custom_theme()

# Add language selector at the top
language_col1, language_col2 = st.columns([6, 1])
with language_col2:
    # Wrap language selector in a dedicated container
    st.markdown('<div class="language-selector-container">', unsafe_allow_html=True)
    selected_lang = st.selectbox(
        t('language_selector'),
        options=["English", "Français"],
        index=0 if st.session_state.language == "en" else 1,
        key="language_selector",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Update language when user changes selection
    if (selected_lang == "English" and st.session_state.language != "en") or \
       (selected_lang == "Français" and st.session_state.language != "fr"):
        st.session_state.language = "en" if selected_lang == "English" else "fr"
        st.rerun()

# Add custom CSS to force white text in buttons
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
    
    /* Language selector styling - only apply to language selector container */
    .language-selector-container div[data-testid="stSelectbox"] {
        width: 120px;
        float: right;
    }
    </style>
    """, unsafe_allow_html=True)

def handle_outliers_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return column.clip(lower_bound, upper_bound)

def handle_outliers_winsorization(column, lower_percentile=0.05, upper_percentile=0.95):
    lower_bound = column.quantile(lower_percentile)
    upper_bound = column.quantile(upper_percentile)
    return column.clip(lower_bound, upper_bound)

def clean_column(column):
    """Convert words to numbers and ensure numeric typing"""
    def convert_word_to_number(value):
        try:
            return w2n.word_to_num(str(value))
        except:
            return value
    cleaned_column = column.apply(convert_word_to_number)
    cleaned_column = pd.to_numeric(cleaned_column, errors='coerce')
    return cleaned_column

# Track page history
if "page_history" not in st.session_state:
    st.session_state.page_history = []
st.session_state.page_history.append("pages/3_Clean.py")

# Initialize session state with consistent names
if "cleaning_mode" not in st.session_state:
    st.session_state.cleaning_mode = None
if "clean_processed_df" not in st.session_state:
    st.session_state.clean_processed_df = None
if "has_cleaned" not in st.session_state:
    st.session_state.has_cleaned = False
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "rename_mappings" not in st.session_state:
    st.session_state.rename_mappings = {}
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "openai_connected" not in st.session_state:
    st.session_state.openai_connected = False
if "cleaned_dataframes" not in st.session_state:
    st.session_state.cleaned_dataframes = {}
if "cleaning_reports" not in st.session_state:
    st.session_state.cleaning_reports = {}
if 'progress' not in st.session_state:
    st.session_state.progress = {'upload': True, 'process': True, 'clean': True, 'visualize': False}
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "data_analyzed" not in st.session_state:
    st.session_state.data_analyzed = False
if "data_quality_scores" not in st.session_state:
    st.session_state.data_quality_scores = {}
if "processing_mode" not in st.session_state:
    st.session_state.processing_mode = None
if "combined_df" not in st.session_state:
    st.session_state.combined_df = None
if "merged_df" not in st.session_state:
    st.session_state.merged_df = None

def reset_cleaning_state():
    """Reset cleaning-specific state when starting fresh"""
    st.session_state.cleaning_mode = None
    st.session_state.has_cleaned = False

def show_manual_cleaning_options():
    """Show all manual cleaning options in the sidebar"""
    with st.sidebar:
        st.header(t('data_cleaning_options'))
        
        # Unified Text Cleaning Section with checkboxes
        with st.expander(t('text_cleaning_formatting'), expanded=True):
            object_columns = st.session_state.clean_processed_df.select_dtypes(include=['object']).columns.tolist()
            if object_columns:
                leading_spaces = st.checkbox(t('delete_leading_spaces'))
                trailing_spaces = st.checkbox(t('delete_trailing_spaces'))
                extra_spaces = st.checkbox(t('delete_extra_spaces'))
                remove_punctuation = st.checkbox(t('remove_punctuation'))
                capitalize_text = st.checkbox(t('capitalize_text'))
        
                if st.button(t('apply_text_cleaning')):
                    try:
                        for col in object_columns:
                            if leading_spaces:
                                st.session_state.clean_processed_df[col] = st.session_state.clean_processed_df[col].str.lstrip()
                            if trailing_spaces:
                                st.session_state.clean_processed_df[col] = st.session_state.clean_processed_df[col].str.rstrip()
                            if extra_spaces:
                                st.session_state.clean_processed_df[col] = st.session_state.clean_processed_df[col].str.replace(r'\s+', ' ', regex=True)
                            if remove_punctuation:
                                st.session_state.clean_processed_df[col] = st.session_state.clean_processed_df[col].apply(
                                    lambda x: re.sub(r'[^\w\s]', '', x) if isinstance(x, str) else x
                                )
                            if capitalize_text:
                                st.session_state.clean_processed_df[col] = st.session_state.clean_processed_df[col].apply(
                                    lambda x: x.title() if isinstance(x, str) else x
                                )
                        st.session_state.has_cleaned = True
                        st.success(t("text_cleaning_applied"))
                        st.rerun()  # Refresh to show changes
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Split Column by Delimiter Section
        with st.expander(t("split_column_delimiter")):
            column_to_split = st.selectbox(t("select_column"), st.session_state.clean_processed_df.columns, key="split_col")
            delimiter = st.text_input(t("delimiter"), key="split_delim")
            fill_value = st.text_input(t("fill_missing_values"), key="split_fill")
            drop_original = st.checkbox(t("drop_original"), key="split_drop")
            
            if st.button(t("apply_split"), key="split_btn"):
                try:
                    split_cols = st.session_state.clean_processed_df[column_to_split].str.split(delimiter, expand=True)
                    if fill_value:
                        split_cols = split_cols.fillna(fill_value)
                    new_names = [f"{column_to_split}_{i+1}" for i in range(split_cols.shape[1])]
                    split_cols.columns = new_names
                    st.session_state.clean_processed_df = pd.concat([st.session_state.clean_processed_df, split_cols], axis=1)
                    if drop_original:
                        st.session_state.clean_processed_df.drop(columns=[column_to_split], inplace=True)
                    st.session_state.has_cleaned = True
                    st.success(t("split_successful"))
                    st.rerun()  # Refresh to show changes
                except Exception as e:
                    st.error(f"Error: {e}")

        # Rename Columns Section
        with st.expander(t("rename_columns")):
            if "rename_mappings" not in st.session_state:
                st.session_state.rename_mappings = {}

            old_name = st.selectbox(t("select_column"), st.session_state.clean_processed_df.columns, key="rename_select")
            new_name = st.text_input(t("new_name"), key="rename_input")
            
            if st.button(t("queue_rename"), key="rename_queue"):
                if new_name.strip():
                    st.session_state.rename_mappings[old_name] = new_name
                    st.success(t("rename_queued"))
            
            if st.session_state.rename_mappings:
                st.write(t("pending_renames"))
                for old, new in st.session_state.rename_mappings.items():
                    st.write(f"{old} → {new}")
            
            if st.button(t("apply_renames"), key="rename_apply"):
                try:
                    st.session_state.clean_processed_df.rename(columns=st.session_state.rename_mappings, inplace=True)
                    st.session_state.rename_mappings = {}
                    st.session_state.has_cleaned = True
                    st.success(t("renames_applied"))
                    st.rerun()  # Refresh to show changes
                except Exception as e:
                    st.error(f"Error: {e}")

        # Drop Columns Section
        with st.expander(t("drop_columns")):
            cols_to_drop = st.multiselect(t("select_columns_to_drop"), st.session_state.clean_processed_df.columns, key="drop_cols")
            if st.button(t("drop_columns_btn"), key="drop_btn"):
                if cols_to_drop:
                    st.session_state.clean_processed_df.drop(columns=cols_to_drop, inplace=True)
                    st.session_state.has_cleaned = True
                    st.success(f"{t('dropped')} {len(cols_to_drop)} {t('columns')}")
                    st.rerun()  # Refresh to show changes
                else:
                    st.warning(t("no_columns_selected"))

        # Change Data Types Section
        with st.expander(t("change_data_types")):
            convert_cols = st.multiselect(t("select_columns"), st.session_state.clean_processed_df.columns, key="dtype_select")
            new_type = st.selectbox(t("new_type"), ["int", "float", "str", "bool", "datetime"], key="dtype_type")
            
            if st.button(t("convert"), key="dtype_btn"):
                for col in convert_cols:
                    try:
                        if new_type == "int":
                            # Use clean_column function to handle word-to-number conversion
                            st.session_state.clean_processed_df[col] = clean_column(st.session_state.clean_processed_df[col]).astype("Int64")
                        elif new_type == "float":
                            st.session_state.clean_processed_df[col] = clean_column(st.session_state.clean_processed_df[col]).astype(float)
                        elif new_type == "str":
                            st.session_state.clean_processed_df[col] = st.session_state.clean_processed_df[col].astype(str)
                        elif new_type == "bool":
                            st.session_state.clean_processed_df[col] = st.session_state.clean_processed_df[col].astype(bool)
                        elif new_type == "datetime":
                            st.session_state.clean_processed_df[col] = pd.to_datetime(st.session_state.clean_processed_df[col], errors='coerce')
                        st.session_state.has_cleaned = True
                        st.success(f"{t('converted')} {col} {t('to')} {new_type}")
                    except Exception as e:
                        st.error(f"{t('error_converting')} {col}: {e}")
                if convert_cols:
                    st.rerun()  # Refresh to show changes

        # Handle Outliers Section
        with st.expander(t("handle_outliers")):
            numeric_cols = st.session_state.clean_processed_df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                selected_cols = st.multiselect(t("select_columns"), numeric_cols, key="outlier_select")
                method = st.radio(t("method"), ["IQR", t("winsorization")], key="outlier_method")
                
                if method == t("winsorization"):
                    lower = st.slider(t("lower_percentile"), 0.0, 0.25, 0.05, 0.01)
                    upper = st.slider(t("upper_percentile"), 0.75, 1.0, 0.95, 0.01)
                
                if st.button(t("fix_outliers"), key="outlier_btn"):
                    for col in selected_cols:
                        try:
                            original_dtype = st.session_state.clean_processed_df[col].dtype
                            series = st.session_state.clean_processed_df[col].copy()
                            
                            if method == "IQR":
                                if pd.api.types.is_integer_dtype(original_dtype):
                                    series = series.astype(float)
                                processed = handle_outliers_iqr(series)
                            else:  # Winsorization
                                if pd.api.types.is_integer_dtype(original_dtype):
                                    series = series.astype(float)
                                processed = handle_outliers_winsorization(series, lower_percentile=lower, upper_percentile=upper)
                            
                            # Convert back to original type if integer
                            if pd.api.types.is_integer_dtype(original_dtype):
                                processed = processed.round().astype(original_dtype)
                            
                            st.session_state.clean_processed_df[col] = processed
                            st.session_state.has_cleaned = True
                            st.success(f"{t('outliers_handled')} {col}")
                        except Exception as e:
                            st.error(f"{t('error_with')} {col}: {e}")
                    if selected_cols:
                        st.rerun()  # Refresh to show changes

        # Handle Missing Values Section
        with st.expander(t("handle_missing_values")):
            na_cols = st.session_state.clean_processed_df.columns[st.session_state.clean_processed_df.isna().any()].tolist()
            if na_cols:
                selected_cols = st.multiselect(t("select_columns"), na_cols, key="missing_select")
                method = st.selectbox(
                    t("method"), 
                    [t("drop"), t("fill_mean"), t("fill_median"), 
                     t("fill_mode"), t("fill_sequential"), t("fill_constant")], 
                    key="missing_method"
                )
                
                if method == t("fill_constant"):
                    fill_value = st.text_input(t("fill_value"), key="constant_fill")
                
                if st.button(t("handle_missing"), key="missing_btn"):
                    for col in selected_cols:
                        try:
                            original_dtype = st.session_state.clean_processed_df[col].dtype
                            
                            if method == t("drop"):
                                original_len = len(st.session_state.clean_processed_df)
                                st.session_state.clean_processed_df.dropna(subset=[col], inplace=True)
                                new_len = len(st.session_state.clean_processed_df)
                                st.success(f"{t('dropped')} {original_len - new_len} {t('rows_with_missing')}")
                            elif method == t("fill_sequential"):
                                if not pd.api.types.is_integer_dtype(original_dtype):
                                    raise ValueError(f"{t('column')} {col} {t('must_be_integer')}")
                                
                                temp_series = st.session_state.clean_processed_df[col].astype(float)
                                interpolated = temp_series.interpolate(method='linear')
                                filled = interpolated.round().astype('Int64')
                                st.session_state.clean_processed_df[col] = filled.ffill().bfill().astype('Int64')
                                st.success(f"{t('filled_sequential')} {col}")
                            elif method == t("fill_constant"):
                                if not fill_value:
                                    st.error(t("enter_fill_value"))
                                    continue
                                
                                try:
                                    # Try to convert fill_value to match column type
                                    if pd.api.types.is_numeric_dtype(original_dtype):
                                        fill_val = float(fill_value)
                                        if pd.api.types.is_integer_dtype(original_dtype):
                                            fill_val = int(fill_val)
                                    else:
                                        fill_val = fill_value
                                    
                                    st.session_state.clean_processed_df[col].fillna(fill_val, inplace=True)
                                    st.success(f"{t('filled_missing')} {col} {t('with')} {fill_val}")
                                except ValueError:
                                    st.error(f"{t('fill_value_match')} {col}")
                            else:  # Fill with Mean, Median, or Mode
                                if method in [t("fill_mean"), t("fill_median")]:
                                    if not pd.api.types.is_numeric_dtype(original_dtype):
                                        st.error(f"{t('cannot_calc_mean')} {col}")
                                        continue
                                        
                                    if pd.api.types.is_integer_dtype(original_dtype):
                                        fill_val = round(st.session_state.clean_processed_df[col].mean() if method == t("fill_mean") else st.session_state.clean_processed_df[col].median())
                                        fill_val = int(fill_val)
                                    else:
                                        fill_val = st.session_state.clean_processed_df[col].mean() if method == t("fill_mean") else st.session_state.clean_processed_df[col].median()
                                else:  # Fill with Mode
                                    fill_val = st.session_state.clean_processed_df[col].mode()[0]
                                
                                st.session_state.clean_processed_df[col].fillna(fill_val, inplace=True)
                                st.success(f"{t('filled_missing')} {col} {t('with')} {fill_val}")
                            
                            # Convert back to original type if needed
                            if method != t("fill_sequential") and pd.api.types.is_integer_dtype(original_dtype):
                                st.session_state.clean_processed_df[col] = st.session_state.clean_processed_df[col].astype(original_dtype)
                            
                            st.session_state.has_cleaned = True
                        except Exception as e:
                            st.error(f"{t('error_with')} {col}: {e}")
                    if selected_cols:
                        st.rerun()  # Refresh to show changes
            else:
                st.write(t("no_missing_columns"))

        # Drop Duplicates Section
        with st.expander(t("drop_duplicates")):
            if st.session_state.clean_processed_df.duplicated().any():
                subset_cols = st.multiselect(
                    t("select_columns_duplicates"),
                    st.session_state.clean_processed_df.columns,
                    key="dup_subset_cols"
                )
                
                keep_option = st.radio(t("keep_option"), ["first", "last", "none"], key="dup_keep")
                
                if st.button(t("drop_duplicates_btn"), key="drop_dupes_btn"):
                    try:
                        before = len(st.session_state.clean_processed_df)
                        st.session_state.clean_processed_df.drop_duplicates(
                            subset=subset_cols if subset_cols else None,
                            keep=keep_option,
                            inplace=True
                        )
                        after = len(st.session_state.clean_processed_df)
                        st.session_state.has_cleaned = True
                        st.success(f"{t('removed')} {before - after} {t('duplicates')}")
                        st.rerun()  # Refresh to show changes
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.write(t("no_duplicates"))
                
        # Reorder Columns Section
        with st.expander(t("reorder_columns")):
            if not st.session_state.clean_processed_df.empty:
                current_columns = st.session_state.clean_processed_df.columns.tolist()
                col_order_str = st.text_input(
                    t("edit_column_order"), 
                    value=", ".join(current_columns),
                    key="col_order_editor"
                )
                
                if st.button(t("apply_column_order")):
                    try:
                        new_order = [col.strip() for col in col_order_str.split(",")]
                        missing = set(current_columns) - set(new_order)
                        extra = set(new_order) - set(current_columns)
                        
                        if missing:
                            st.error(f"{t('missing_columns')}: {', '.join(missing)}")
                        elif extra:
                            st.error(f"{t('unknown_columns')}: {', '.join(extra)}")
                        else:
                            st.session_state.clean_processed_df = st.session_state.clean_processed_df[new_order]
                            st.session_state.has_cleaned = True
                            st.success(t("column_order_updated"))
                            st.rerun()  # Refresh to show changes
                    except Exception as e:
                        st.error(f"Error: {e}")

def main():
    # Check if there are uploaded files
    if "dataframes" not in st.session_state or not st.session_state.dataframes:
        st.info(t('no_files_uploaded'))
        if st.button(t('back_to_upload'), key="no_files_back", use_container_width=True):
            reset_cleaning_state()
            st.switch_page("Home.py")
        return
    
    # Debug info to help diagnose issues
    st.sidebar.markdown("### Debug Info")
    st.sidebar.write(f"Available files: {list(st.session_state.dataframes.keys())}")
    
    # Verify processed datasets are available
    processed_file = None
    if 'combined_dataset.csv' in st.session_state.dataframes:
        st.sidebar.write("✅ Combined dataset is available")
        processed_file = 'combined_dataset.csv'
        # Ensure it's not empty
        if len(st.session_state.dataframes['combined_dataset.csv']) > 0:
            st.sidebar.write(f"  - Rows: {len(st.session_state.dataframes['combined_dataset.csv'])}")
        else:
            st.sidebar.write("  - Warning: Dataset is empty")
    
    if 'merged_dataset.csv' in st.session_state.dataframes:
        st.sidebar.write("✅ Merged dataset is available")
        if not processed_file:
            processed_file = 'merged_dataset.csv'
        # Ensure it's not empty
        if len(st.session_state.dataframes['merged_dataset.csv']) > 0:
            st.sidebar.write(f"  - Rows: {len(st.session_state.dataframes['merged_dataset.csv'])}")
        else:
            st.sidebar.write("  - Warning: Dataset is empty")
    
    # Set up progress bar
    cols = st.columns(5)
    with cols[0]:
        st.markdown(f"#### 1. {t('upload')} ✅")
    with cols[1]:
        st.markdown(f"#### 2. {t('Analysis')} ✅")
    with cols[2]:
        st.markdown(f"#### 3. {t('process')} ✅")
    with cols[3]:
        st.markdown(f"#### 4. {t('clean')} 🔄")
    with cols[4]:
        st.markdown(f"#### 5. {t('visualize')}")
    
    st.progress(0.6)  # 60% through the workflow
    
    # Mode selection - simplified to just two clear options
    if st.session_state.cleaning_mode is None:
        st.markdown(f"# {t('choose_cleaning_method')}")
        st.markdown("---")
        
        # Show file summary
        file_count = len(st.session_state.dataframes)
        total_rows = sum(len(df) for df in st.session_state.dataframes.values())
        
        st.markdown(f"**Ready to clean {file_count} file(s) with {total_rows:,} total rows**")
        
        # Ensure processed dataset is available
        processed_file = None
        if 'combined_dataset.csv' in st.session_state.dataframes:
            processed_file = 'combined_dataset.csv'
        elif 'merged_dataset.csv' in st.session_state.dataframes:
            processed_file = 'merged_dataset.csv'
        
        if processed_file:
            st.success(f"Processed dataset found: {processed_file}")
        else:
            st.warning("No processed dataset found. You may need to go back and process your data first.")
        
        # Simple two-button layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {t('ai_cleaning')}")
            st.markdown(t('let_ai_clean'))
            
            # Add four bullet points in the same style as manual side
            st.markdown(f"• {t('detect_fix_issues')}")
            st.markdown(f"• {t('convert_data_types')}")
            st.markdown(f"• {t('fill_missing_values')}")
            st.markdown(f"• {t('remove_outliers')}")
            
            # Check if API key is already in session state
            api_key_available = st.session_state.get("openai_api_key") is not None
            
            if not api_key_available:
                # If no API key is configured, let the user input one
                openai_api_key = st.text_input(
                    "OpenAI API Key (optional)",
                    type="password",
                    value=""
                )
                
                if openai_api_key:
                    st.session_state.openai_api_key = openai_api_key
                    st.session_state.openai_connected = True
            
            # AI cleaning button
            if st.button(t('clean_with_ai'), use_container_width=True, type="primary"):
                st.session_state.cleaning_mode = "ai"
                st.rerun()
        
        with col2:
            st.markdown(f"### {t('manual_cleaning')}")
            st.markdown(t('clean_data_using_tools'))
            
            # List key features without too much detail
            st.markdown(f"{t('text_cleaning')}")
            st.markdown(f"{t('handle_missing_values')}")
            st.markdown(f"{t('remove_duplicates')}")
            st.markdown(f"{t('column_operations')}")
            
            # Manual cleaning button
            if st.button(t('clean_manually'), use_container_width=True, type="primary"):
                st.session_state.cleaning_mode = "standard"
                
                # Initialize the working dataframe with the processed file first, if available
                if "dataframes" in st.session_state and st.session_state.dataframes:
                    # Prioritize the processed dataset if available
                    if 'combined_dataset.csv' in st.session_state.dataframes:
                        first_file = 'combined_dataset.csv'
                    elif 'merged_dataset.csv' in st.session_state.dataframes:
                        first_file = 'merged_dataset.csv'
                    else:
                        first_file = list(st.session_state.dataframes.keys())[0]
                        
                    st.session_state.clean_processed_df = st.session_state.dataframes[first_file].copy()
                    st.session_state.current_file = first_file
                
                st.rerun()
        
        # Add back to upload button
        st.markdown("---")
        if st.button(t('back_to_upload'), use_container_width=True):
            # Reset any cleaning-specific state
            reset_cleaning_state()
            st.switch_page("Home.py")
    
    # Handle AI cleaning mode
    elif st.session_state.cleaning_mode == "ai":
        hero_section(
            t('ai_powered_data_cleaning'),
            t('let_ai_clean')
        )
        
        # File selection
        st.subheader(t('select_dataset_to_clean'))
        # Organize files with processed ones at the top
        file_options = list(st.session_state.dataframes.keys())
        # Prioritize processed datasets
        if 'combined_dataset.csv' in file_options:
            file_options.remove('combined_dataset.csv')
            file_options.insert(0, 'combined_dataset.csv')
        if 'merged_dataset.csv' in file_options:
            file_options.remove('merged_dataset.csv')
            file_options.insert(0, 'merged_dataset.csv')

        selected_file = st.selectbox(
            t('choose_file_to_clean'),
            file_options,
            key="ai_file_selector",
            format_func=lambda x: f"{x} ({'PROCESSED - ' if x in ['combined_dataset.csv', 'merged_dataset.csv'] else ''}{len(st.session_state.dataframes[x])} rows, {len(st.session_state.dataframes[x].columns)} columns)"
        )
        
        df = st.session_state.dataframes[selected_file]
        
        # Check if we have already cleaned this file in this session
        already_cleaned = selected_file in st.session_state.cleaned_dataframes
        
        # Show original dataset preview
        st.subheader(t('original_dataset'))
        st.dataframe(df.head(10), use_container_width=True, height=300)
        
        # If already cleaned, show the cleaned version
        if already_cleaned:
            cleaned_df = st.session_state.cleaned_dataframes[selected_file]
            st.success("✅ This dataset has already been cleaned!")
            
            # Show the cleaned dataset
            st.subheader(t('cleaned_dataset'))
            st.dataframe(cleaned_df, use_container_width=True, height=300)
            
            # Simplified navigation - only keep the analyze button
            st.markdown("---")
            if st.button(t("analyze_cleaned_data"), use_container_width=True, type="primary"):
                # Save the cleaned version in place of the original to ensure it's used
                st.session_state.dataframes[selected_file] = cleaned_df.copy()
                
                # Update the processed data variables
                if selected_file == 'combined_dataset.csv':
                    st.session_state.combined_df = cleaned_df.copy()
                elif selected_file == 'merged_dataset.csv':
                    st.session_state.merged_df = cleaned_df.copy()
                
                # Set analyzed flag to false to trigger re-analysis
                st.session_state.data_analyzed = False
                
                # Update progress and navigate to analysis page
                st.session_state.progress['clean'] = True
                st.session_state.progress['visualize'] = False
                st.switch_page("pages/1_Analysis.py")
        
        # If not cleaned yet, show the clean button
        else:
            # AI Cleaning button - simple and direct
            if st.button(t('clean_with_ai'), type="primary", use_container_width=True):
                with st.spinner("Cleaning data with AI... This may take a minute."):
                    try:
                        # Use the AI cleaning function
                        cleaned_df, report = clean_dataframe_with_ai(
                            df, 
                            openai_api_key=st.session_state.openai_api_key
                        )
                        
                        # Save results
                        st.session_state.clean_processed_df = cleaned_df
                        st.session_state.cleaned_dataframes[selected_file] = cleaned_df
                        st.session_state.cleaning_reports[selected_file] = report
                        st.session_state.has_cleaned = True
                        
                        # Rerun to show the cleaned version UI
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"Error during cleaning: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc(), language="python")
        
        # Display back button for mode selection
        st.markdown("---")
        if st.button(t('select_different_cleaning_method'), use_container_width=True):
            reset_cleaning_state()
            st.rerun()
    
    # Handle standard cleaning mode
    elif st.session_state.cleaning_mode == "standard":
        # Add a message to draw attention to the sidebar for cleaning options
        st.sidebar.info(t('use_sidebar_options'))
        
        hero_section(
            t('standard_data_cleaning'),
            t('clean_data_using_tools')
        )
        
        # File selection
        st.subheader(t('select_dataset_to_clean'))
        file_col1, file_col2 = st.columns([3, 1])
        with file_col1:
            # Organize files with processed ones at the top
            file_options = list(st.session_state.dataframes.keys())
            # Prioritize processed datasets
            if 'combined_dataset.csv' in file_options:
                file_options.remove('combined_dataset.csv')
                file_options.insert(0, 'combined_dataset.csv')
            if 'merged_dataset.csv' in file_options:
                file_options.remove('merged_dataset.csv')
                file_options.insert(0, 'merged_dataset.csv')
            
            selected_file = st.selectbox(
                t('choose_file_to_clean'),
                file_options,
                key="file_selector",
                format_func=lambda x: f"{x} ({'PROCESSED - ' if x in ['combined_dataset.csv', 'merged_dataset.csv'] else ''}{len(st.session_state.dataframes[x])} rows, {len(st.session_state.dataframes[x].columns)} columns)"
            )
        
        # Update working dataframe if file changed
        if st.session_state.current_file != selected_file:
            st.session_state.clean_processed_df = st.session_state.dataframes[selected_file].copy()
            st.session_state.current_file = selected_file
        
        # Quick stats about the dataset
        stats_cols = st.columns(4)
        with stats_cols[0]:
            st.metric(t('rows'), len(st.session_state.clean_processed_df))
        with stats_cols[1]:
            st.metric(t('columns'), len(st.session_state.clean_processed_df.columns))
        with stats_cols[2]:
            st.metric(t('missing_values'), st.session_state.clean_processed_df.isna().sum().sum())
        with stats_cols[3]:
            st.metric(t('duplicates'), st.session_state.clean_processed_df.duplicated().sum())
        
        # Original Dataset Preview
        with st.expander(t('original_dataset'), expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{t('original_data_preview')}**")
                st.dataframe(
                    st.session_state.dataframes[selected_file],
                    height=300,
                    use_container_width=True,
                    hide_index=True
                )
            with col2:
                st.write(f"**{t('original_data_types')}**")
                st.write(st.session_state.dataframes[selected_file].dtypes)
        
        # Working Dataset Preview
        if st.session_state.clean_processed_df is not None:
            st.write(f"### {t('working_dataset')}")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{t('current_data_preview')}**")
                st.dataframe(
                    st.session_state.clean_processed_df,
                    height=400,
                    use_container_width=True,
                    hide_index=True
                )
            with col2:
                st.write(f"**{t('current_data_types')}**")
                st.write(st.session_state.clean_processed_df.dtypes)
        
        # Add instructions for sidebar with improved wording
        st.info(t('all_cleaning_tools'))
        
        # Show manual cleaning options
        show_manual_cleaning_options()
        
        # Navigation buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(t('select_different_cleaning_method'), use_container_width=True):
                reset_cleaning_state()
                st.rerun()
        
        with col2:
            if st.button(t('analyze_cleaned_data'), use_container_width=True, type="primary"):
                if st.session_state.has_cleaned:
                    # Auto-save the dataframe
                    if selected_file in st.session_state.dataframes:
                        # Save the cleaned version in place of the original
                        st.session_state.dataframes[selected_file] = st.session_state.clean_processed_df.copy()
                        st.session_state.cleaned_dataframes[selected_file] = st.session_state.clean_processed_df.copy()
                        
                        # Also update the processed data variables that might be used by Analysis page
                        if selected_file == 'combined_dataset.csv':
                            st.session_state.combined_df = st.session_state.clean_processed_df.copy()
                        elif selected_file == 'merged_dataset.csv':
                            st.session_state.merged_df = st.session_state.clean_processed_df.copy()
                        
                        st.success(f"{t('saved_cleaned_version')} {selected_file}")
                        
                        # Set analyzed flag to false to trigger re-analysis
                        st.session_state.data_analyzed = False
                        
                        # Update progress and navigate to analysis page
                        st.session_state.progress['clean'] = True
                        st.session_state.progress['visualize'] = False
                        st.switch_page("pages/1_Analysis.py")
                else:
                    st.warning(t('no_changes_made'))

if __name__ == "__main__":
    main() 