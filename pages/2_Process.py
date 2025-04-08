import streamlit as st
import pandas as pd
from collections import defaultdict
import json
from utils.theme import apply_custom_theme, hero_section, feature_card
from utils.ai import process_dataframes_with_ai
from utils.translations import get_translation_function

# Get translation function
t = get_translation_function()

# Set page config
st.set_page_config(
    page_title=f"{t('process')} - Data Cleaning App",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="collapsed"
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

# Track page history
if "page_history" not in st.session_state:
    st.session_state.page_history = []
st.session_state.page_history.append("pages/1_Process.py")

# ========== Session State ==========
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = None
if 'merged_df' not in st.session_state:
    st.session_state.merged_df = None
if 'renamed_columns' not in st.session_state:
    st.session_state.renamed_columns = {}
if 'column_mappings' not in st.session_state:
    st.session_state.column_mappings = defaultdict(list)
if 'process_completed' not in st.session_state:
    st.session_state.process_completed = False
if 'last_operation' not in st.session_state:
    st.session_state.last_operation = None
if 'processing_mode' not in st.session_state:
    st.session_state.processing_mode = None
if 'progress' not in st.session_state:
    st.session_state.progress = {'upload': True, 'process': False, 'clean': False, 'visualize': False}

def reset_process_state(keep_completed=False):
    """Reset process-specific session state when starting fresh"""
    if not keep_completed:
        st.session_state.combined_df = None
        st.session_state.merged_df = None
        st.session_state.renamed_columns = {}
        
        # Remove the combined dataset from dataframes
        if 'dataframes' in st.session_state and 'combined_dataset.csv' in st.session_state.dataframes:
            del st.session_state.dataframes['combined_dataset.csv']
        
        # Remove the merged dataset from dataframes
        if 'dataframes' in st.session_state and 'merged_dataset.csv' in st.session_state.dataframes:
            del st.session_state.dataframes['merged_dataset.csv']
            
        st.session_state.process_completed = False
        st.session_state.last_operation = None
        st.session_state.processing_mode = None
    else:
        # Just reset the mode selection while keeping the data
        st.session_state.processing_mode = None

def main():
    # Check if there are uploaded files
    if "dataframes" not in st.session_state or not st.session_state.dataframes:
        st.info(t('no_files_uploaded'))
        if st.button(t('back_to_upload'), key="no_files_back", use_container_width=True):
            reset_process_state(keep_completed=False)
            st.switch_page("Home.py")
        return
    
    # Show workflow progress
    cols = st.columns(5)
    with cols[0]:
        st.markdown(f"#### 1. {t('upload')} ✅")
    with cols[1]:
        st.markdown(f"#### 2. {t('Analysis')} ✅")  
    with cols[2]:
        st.markdown(f"#### 3. {t('process')} 🔄")
    with cols[3]:
        st.markdown(f"#### 4. {t('clean')}")
    with cols[4]:
        st.markdown(f"#### 5. {t('visualize')}")
    
    st.progress(0.4)  # 40% through the workflow
    
    # Mode selection - simplified to just two clear options
    if st.session_state.processing_mode is None:
        st.markdown(f"# {t('choose_processing_method')}")
        st.markdown("---")
        
        # Show file summary
        file_count = len(st.session_state.dataframes)
        
        if file_count < 2:
            st.warning(t('need_at_least_two_files'))
            if st.button(t('back_to_upload'), key="not_enough_files_back"):
                reset_process_state(keep_completed=False)
                st.switch_page("Home.py")
            return
        
        # Show file stats
        total_rows = sum(len(df) for df in st.session_state.dataframes.values())
        st.markdown(f"**Ready to process {file_count} files with {total_rows:,} total rows**")
        
        # Simple two-button layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {t('ai_processing')}")
            st.markdown(t('let_ai_analyze'))
            
            # Add four bullet points in the same style as manual side
            st.markdown(f"• {t('intelligently_analyze')}")
            st.markdown(f"• {t('identify_common_fields')}")
            st.markdown(f"• {t('handle_column_mapping')}")
            st.markdown(f"• {t('optimize_data_structure')}")
            
            # Check if API key is already in session state
            api_key_available = st.session_state.get("openai_api_key") is not None
            
            if not api_key_available:
                # If no API key is configured, let the user input one
                openai_api_key = st.text_input(
                    t('openai_api_key'),
                    type="password",
                    value=""
                )
                
                if openai_api_key:
                    st.session_state.openai_api_key = openai_api_key
                    st.session_state.openai_connected = True
            
            # AI processing button - enabled if API key is available in session state
            if st.session_state.get("openai_api_key"):
                if st.button(t('process_with_ai'), key="method_choice_ai", use_container_width=True, type="primary"):
                    st.session_state.processing_mode = "ai"
                    st.rerun()
            else:
                # Disabled button with explanation if no API key
                st.button(t('process_with_ai'), key="method_choice_ai_disabled", use_container_width=True, disabled=True)
                st.info(t('enter_api_key'))
        
        with col2:
            st.markdown(f"### {t('manual_processing')}")
            st.markdown(t('process_data_using_tools'))
            
            # List key features without too much detail
            st.markdown(f"{t('vertical_concatenation')}")
            st.markdown(f"{t('horizontal_concatenation')}")
            st.markdown(f"{t('join_merge')}")
            st.markdown(f"{t('column_mapping')}")
            
            # Manual processing button
            if st.button(t('process_manually'), key="method_choice_manual", use_container_width=True, type="primary"):
                st.session_state.processing_mode = "standard"
                st.rerun()
                
        # Add a Back to Upload button at the bottom of the method selection page
        st.markdown("---")
        if st.button(t('back_to_upload'), key="method_selection_back", use_container_width=True):
            reset_process_state(keep_completed=False)
            # Also reset any temporary datasets
            if 'result_dataset' in st.session_state:
                del st.session_state.result_dataset
            st.switch_page("Home.py")
    
    # Handle AI processing mode
    elif st.session_state.processing_mode == "ai":
        hero_section(
            t("ai_powered_data_processing"),
            t("let_ai_analyze")
        )
        
        # Display available datasets
        st.subheader(t("original_datasets"))
        
        # Show the datasets in a grid
        cols = st.columns(min(len(st.session_state.dataframes), 3))
        for i, (fname, df) in enumerate(st.session_state.dataframes.items()):
            with cols[i % 3]:
                st.write(f"**{fname}**")
                st.dataframe(df.head(5), height=200)
                st.write(f"{t('rows')}: {len(df)} | {t('columns')}: {len(df.columns)}")
        
        # AI Processing button - simple and direct
        if st.button(t("process_with_ai_button"), key="ai_process_button", type="primary", use_container_width=True):
            with st.spinner(t("processing_with_ai")):
                try:
                    # Process the dataframes with AI
                    result_df, report = process_dataframes_with_ai(
                        st.session_state.dataframes,
                        api_key=st.session_state.openai_api_key
                    )
                    
                    if 'error' in report:
                        st.error(f"{t('ai_processing_failed')}: {report['error']}")
                    else:
                        # Save the processed dataframe
                        operation = report.get('operation', 'processing')
                        result_filename = f"combined_dataset.csv"
                        st.session_state.dataframes[result_filename] = result_df.copy(deep=True)
                        
                        # Update session state
                        st.session_state.combined_df = result_df.copy(deep=True)
                        st.session_state.last_operation = 'concat'
                        st.session_state.process_completed = True
                        
                        # Display success message
                        st.success(t("processing_completed"))
                        
                        # Show the result
                        st.subheader(t("processing_result"))
                        
                        # Show basic metrics
                        metrics_cols = st.columns(3)
                        with metrics_cols[0]:
                            st.metric(t("files_processed"), len(report['files_processed']))
                        with metrics_cols[1]:
                            st.metric(t("original_rows"), report['total_rows_before'])
                        with metrics_cols[2]:
                            st.metric(t("result_rows"), report['total_rows_after'])
                        
                        # Show the processed dataframe
                        st.dataframe(result_df, use_container_width=True, height=400)
                        
                        # Navigation buttons
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(t("process_another_dataset"), key="ai_process_another", use_container_width=True):
                                reset_process_state(keep_completed=True)
                                st.rerun()
                        with col2:
                            if st.button(t("proceed_to_cleaning"), key="ai_proceed", use_container_width=True, type="primary"):
                                st.session_state.progress['process'] = True
                                st.session_state.progress['clean'] = True
                                # Ensure the result is saved in the dataframes dictionary
                                st.session_state.dataframes[result_filename] = result_df.copy(deep=True)
                                # Make sure the result is saved to combined_df for consistency
                                st.session_state.combined_df = result_df.copy(deep=True)
                                st.switch_page("pages/3_Clean.py")
                
                except Exception as e:
                    st.error(f"{t('error_during_ai_processing')}: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
        
        # Back button
        st.markdown("---")
        if st.button(t("select_different_processing_method"), key="ai_mode_select_different", use_container_width=True):
            reset_process_state(keep_completed=True)
            st.session_state.processing_mode = None
            st.rerun()
    
    # Handle standard processing mode
    elif st.session_state.processing_mode == "standard":
        hero_section(
            t("standard_data_processing"),
            t("manually_combine_datasets")
        )
        
        # Display available files
        st.subheader(t("available_datasets"))
        files_container = st.container()
        with files_container:
            cols = st.columns(min(len(st.session_state.dataframes), 3))
            for i, (fname, df) in enumerate(st.session_state.dataframes.items()):
                with cols[i % 3]:
                    st.write(f"**{fname}**")
                    st.dataframe(df.head(5), height=200)
                    st.write(f"{t('columns_list')}: {', '.join(df.columns)}")
        
        # ========== Tab System with improved styling ==========
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.2rem;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            border-radius: 5px;
        }
        </style>""", unsafe_allow_html=True)
        
        concat_tab, merge_tab = st.tabs([t("concatenation_tab"), t("merging_tab")])

        with concat_tab:
            if len(st.session_state.dataframes) >= 2:
                st.subheader(t("concatenation_options"))
                
                # Clear any previous operation if switching tabs
                if st.session_state.last_operation == "merge":
                    # Reset but keep completion status
                    was_completed = st.session_state.process_completed
                    reset_process_state(keep_completed=True)
                    st.session_state.process_completed = was_completed
                    st.session_state.processing_mode = "standard"
                
                # Display data preview like in the reference UI
                st.subheader("Uploaded Data Preview")
                cols = st.columns(min(len(st.session_state.dataframes), 3))
                for i, (fname, df) in enumerate(st.session_state.dataframes.items()):
                    with cols[i % 3]:
                        st.write(f"**{fname}**")
                        st.dataframe(df.head(5), height=200)
                
                # Select files to concatenate
                files_to_concat = st.multiselect(
                    t("select_files_to_concat"),
                    options=list(st.session_state.dataframes.keys()),
                    default=list(st.session_state.dataframes.keys())[:2]
                )
                
                if len(files_to_concat) >= 2:
                    selected_dfs = {f: st.session_state.dataframes[f] for f in files_to_concat}
                    
                    # Configuration options
                    reset_index = st.checkbox(t("reset_index"), value=True, key='concat_reset')
                    
                    all_columns = [set(df.columns) for df in selected_dfs.values()]
                    common_columns = set.intersection(*all_columns)
                    all_match = all(cols == all_columns[0] for cols in all_columns)
                    no_common = len(common_columns) == 0
                    
                    if all_match:
                        st.success(t("all_columns_match"))
                        if st.button(t("vertical_concatenation_btn")):
                            with st.spinner(t("processing")):
                                combined = pd.concat(selected_dfs.values(), axis=0)
                                st.session_state.combined_df = combined.reset_index(drop=True) if reset_index else combined
                                # Auto-save with default name
                                result_name = "combined_dataset.csv"
                                st.session_state.dataframes[result_name] = st.session_state.combined_df.copy(deep=True)
                                st.session_state.process_completed = True
                                st.session_state.last_operation = "concat"
                                st.success(f"{t('dataset_auto_saved')} '{result_name}'")
                                # Refresh page to show navigation buttons
                                st.rerun()
                    
                    elif no_common:
                        st.warning(t("no_common_columns"))
                        if st.button(t("horizontal_concatenation_btn")):
                            with st.spinner(t("processing")):
                                try:
                                    combined = pd.concat(selected_dfs.values(), axis=1)
                                    if reset_index:
                                        combined = combined.reset_index(drop=True)
                                    st.session_state.combined_df = combined
                                    # Auto-save with default name
                                    result_name = "combined_dataset.csv"
                                    st.session_state.dataframes[result_name] = st.session_state.combined_df.copy(deep=True)
                                    st.session_state.process_completed = True
                                    st.session_state.last_operation = "concat"
                                    st.success(f"{t('dataset_auto_saved')} '{result_name}'")
                                    # Refresh page to show navigation buttons
                                    st.rerun()
                                except ValueError as e:
                                    st.error(f"{t('concatenation_error')}: {e}")
                    
                    else:
                        concat_method = st.radio(t("concatenation_method"), [t("vertical"), t("horizontal")])
                        
                        if concat_method == t("vertical"):
                            st.info(t("map_matching_columns"))
                            all_columns_flat = list({col for cols in all_columns for col in cols})
                            
                            selected_cols = st.multiselect(t("select_columns_to_unify"), all_columns_flat)
                            new_name = st.text_input(t("new_unified_column_name"))
                            
                            if st.button(t("apply_column_unification"), disabled=not (new_name and selected_cols)):
                                st.session_state.renamed_columns[new_name] = selected_cols
                                st.success(f"{t('unified_columns')} {len(selected_cols)} {t('into')} '{new_name}'")
                                st.rerun()
                            
                            if st.session_state.renamed_columns:
                                st.write("**Active Mappings:**")
                                for new, old in st.session_state.renamed_columns.items():
                                    st.write(f"{new} ← {', '.join(old)}")
                                
                                if st.button(t("clear_all_mappings")):
                                    st.session_state.renamed_columns = {}
                                    st.rerun()
                            
                            modified_dfs = []
                            for df in selected_dfs.values():
                                modified_df = df.rename(columns={
                                    old: new for new, olds in st.session_state.renamed_columns.items() 
                                    for old in olds if old in df.columns
                                })
                                modified_dfs.append(modified_df)
                            
                            modified_columns = [set(df.columns) for df in modified_dfs]
                            if all(cols == modified_columns[0] for cols in modified_columns):
                                if st.button(t("perform_vertical_concat")):
                                    with st.spinner(t("processing")):
                                        combined = pd.concat(modified_dfs, axis=0)
                                        st.session_state.combined_df = combined.reset_index(drop=True) if reset_index else combined
                                        # Auto-save with default name
                                        result_name = "combined_dataset.csv"
                                        st.session_state.dataframes[result_name] = st.session_state.combined_df.copy(deep=True)
                                        st.session_state.process_completed = True
                                        st.session_state.last_operation = "concat"
                                        st.success(f"{t('dataset_auto_saved')} '{result_name}'")
                                        # Refresh page to show navigation buttons
                                        st.rerun()
                            else:
                                st.error(t("columns_still_dont_match"))
                        
                        else:  # Horizontal concatenation
                            st.info(t("rename_common_columns"))
                            common_cols = list(common_columns)
                            
                            if common_cols:
                                st.write("**Common Columns to Rename:**")
                                for col in common_cols:
                                    if st.checkbox(f"Show rename options for '{col}'", key=f"chk_{col}"):
                                        cols = st.columns(len(selected_dfs))
                                        new_names = []
                                        for i, (fname, df) in enumerate(selected_dfs.items()):
                                            with cols[i]:
                                                if col in df.columns:
                                                    new_name = st.text_input(
                                                        f"New name for {col} in {fname}",
                                                        value=f"{col}_{i+1}",
                                                        key=f"h_{col}_{i}"
                                                    )
                                                    new_names.append((i, fname, new_name))
                                        
                                        if st.button(f"Apply Renames for {col}", key=f"btn_{col}"):
                                            for dataset_idx, fname, new_name in new_names:
                                                key = f"dataset_{dataset_idx}_renames"
                                                if key not in st.session_state:
                                                    st.session_state[key] = {}
                                                st.session_state[key][col] = new_name
                                                st.success(f"Renamed '{col}' to '{new_name}' in {fname}")
                                            st.rerun()
                            
                            if st.button(t("perform_horizontal_concat")):
                                with st.spinner(t("processing")):
                                    try:
                                        renamed_dfs = []
                                        for i, (fname, df) in enumerate(selected_dfs.items()):
                                            renames = st.session_state.get(f"dataset_{i}_renames", {})
                                            renamed_df = df.rename(columns=renames)
                                            renamed_dfs.append(renamed_df)
                                        
                                        combined = pd.concat(renamed_dfs, axis=1)
                                        if reset_index:
                                            combined = combined.reset_index(drop=True)
                                        st.session_state.combined_df = combined
                                        # Auto-save with default name
                                        result_name = "combined_dataset.csv"
                                        st.session_state.dataframes[result_name] = st.session_state.combined_df.copy(deep=True)
                                        st.session_state.process_completed = True
                                        st.session_state.last_operation = "concat"
                                        st.success(f"{t('dataset_auto_saved')} '{result_name}'")
                                        st.success("Concatenation completed successfully!")
                                        # Refresh page to show navigation buttons
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"{t('horizontal_concat_failed')}: {str(e)}")

                    if st.session_state.combined_df is not None and st.session_state.last_operation == "concat":
                        st.markdown("---")
                        st.subheader(t("concatenation_result"))
                        st.dataframe(st.session_state.combined_df)
                        st.download_button(
                            t("download_concatenated_csv"),
                            st.session_state.combined_df.to_csv(index=False),
                            "concatenated.csv"
                        )
                        
                        # Ensure process completed flag is set
                        st.session_state.process_completed = True
                else:
                    st.warning(t("select_at_least_two_files"))
            else:
                st.warning(t("upload_at_least_two_files"))

        with merge_tab:
            if len(st.session_state.dataframes) >= 2:
                st.subheader(t("merge_configuration"))
                
                # Clear any previous operation if switching tabs
                if st.session_state.last_operation == "concat":
                    # Reset but keep completion status
                    was_completed = st.session_state.process_completed
                    reset_process_state(keep_completed=True)
                    st.session_state.process_completed = was_completed
                    st.session_state.processing_mode = "standard"
                
                # Data preview in two columns exactly like reference UI
                st.subheader("Uploaded Data Preview")
                files = list(st.session_state.dataframes.keys())
                cols = st.columns(2)
                with cols[0]:
                    st.write(f"**{files[0]}**")
                    st.dataframe(st.session_state.dataframes[files[0]], height=200)
                with cols[1]:
                    st.write(f"**{files[1] if len(files) > 1 else ''}**")
                    if len(files) > 1:
                        st.dataframe(st.session_state.dataframes[files[1]], height=200)
                
                st.subheader("Merge Configuration")
                left_file = st.selectbox("Left Dataset", files)
                right_file = st.selectbox("Right Dataset", [f for f in files if f != left_file])
                
                left_df = st.session_state.dataframes[left_file]
                right_df = st.session_state.dataframes[right_file]
                
                st.markdown("**Key Pairing**")
                num_pairs = st.number_input("Number of key pairs", 1, 5, 1)
                key_pairs = []
                for i in range(num_pairs):
                    cols = st.columns(2)
                    with cols[0]:
                        left_key = st.selectbox(f"Left key {i+1}", left_df.columns, key=f"lk_{i}")
                    with cols[1]:
                        right_key = st.selectbox(f"Right key {i+1}", right_df.columns, key=f"rk_{i}")
                    key_pairs.append((left_key, right_key))
                
                join_type = st.selectbox("Join Type", ["inner", "left", "right", "outer"])
                suffixes = st.text_input("Suffixes", "_x,_y")
                
                if st.button("Execute Merge"):
                    with st.spinner(t("merging_datasets")):
                        try:
                            merged = pd.merge(
                                left_df,
                                right_df,
                                left_on=[pair[0] for pair in key_pairs],
                                right_on=[pair[1] for pair in key_pairs],
                                how=join_type,
                                suffixes=suffixes.split(",")
                            )
                            st.session_state.merged_df = merged
                            # Auto-save with default name
                            result_name = "merged_dataset.csv"
                            st.session_state.dataframes[result_name] = st.session_state.merged_df.copy(deep=True)
                            st.session_state.process_completed = True
                            st.session_state.last_operation = "merge"
                            st.success(f"{t('merge_completed')} '{result_name}'")
                            st.success("Merge completed successfully!")
                            
                            # Rerun to show the "Proceed to Cleaning" button
                            st.rerun()
                        except Exception as e:
                            st.error(f"{t('merge_failed')}: {str(e)}")
                
                if st.session_state.merged_df is not None and st.session_state.last_operation == "merge":
                    st.markdown("---")
                    st.subheader(t("merge_result"))
                    st.dataframe(st.session_state.merged_df)
                    st.download_button(
                        t("download_merged_csv"),
                        st.session_state.merged_df.to_csv(index=False),
                        "merged.csv"
                    )
            else:
                st.warning(t("upload_at_least_two_files"))
    
    # Add navigation button to select a different method if no processing has been done yet
    if not st.session_state.process_completed and st.session_state.processing_mode is not None:
        st.markdown("---")
        if st.button(t("select_different_processing_method"), key="bottom_select_different", use_container_width=True):
            reset_process_state(keep_completed=True)
            st.session_state.processing_mode = None
            st.rerun()

    # Navigation buttons for completed processing
    if st.session_state.process_completed:
        st.markdown("---")
        st.success(t("processing_completed_proceed"))
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(t("back_to_upload"), key="completed_back", use_container_width=True):
                # Make sure to reset all processing state before going back to upload
                reset_process_state(keep_completed=False)
                # Also reset any temporary datasets
                if 'result_dataset' in st.session_state:
                    del st.session_state.result_dataset
                st.switch_page("Home.py")
        with col2:
            if st.button(t('analyze_processed_data'), key="analyze_processed", use_container_width=True, type="primary"):
                # Ensure we have properly saved the processed dataset
                if st.session_state.last_operation == "concat" and st.session_state.combined_df is not None:
                    # Make sure the combined dataset is in st.session_state.dataframes
                    result_name = "combined_dataset.csv"
                    st.session_state.dataframes[result_name] = st.session_state.combined_df.copy(deep=True)
                elif st.session_state.last_operation == "merge" and st.session_state.merged_df is not None:
                    # Make sure the merged dataset is in st.session_state.dataframes
                    result_name = "merged_dataset.csv"
                    st.session_state.dataframes[result_name] = st.session_state.merged_df.copy(deep=True)
                
                # Reset analysis state to ensure fresh analysis
                if "data_analyzed" in st.session_state:
                    st.session_state.data_analyzed = False
                if "data_quality_scores" in st.session_state:
                    st.session_state.data_quality_scores = {}
                
                # Update progress state
                st.session_state.progress['upload'] = True
                st.session_state.progress['process'] = True
                
                # Navigate to analysis page
                st.switch_page("pages/1_Analysis.py")

    # If coming from Analysis page directly with standard mode and fewer than 2 files
    elif st.session_state.processing_mode == "standard" and len(st.session_state.dataframes) < 2:
        st.warning(t("need_at_least_two_files"))
        if st.button(t("return_to_upload"), use_container_width=True):
            reset_process_state(keep_completed=False)
            # Also reset any temporary datasets
            if 'result_dataset' in st.session_state:
                del st.session_state.result_dataset
            st.session_state.processing_mode = None
            st.switch_page("Home.py")
        return

if __name__ == "__main__":
    main() 