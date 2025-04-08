import streamlit as st
import pandas as pd
import numpy as np
from utils.translations import get_translation_function
from utils.theme import apply_custom_theme, card, section_header
from utils.ai import analyze_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set page config
st.set_page_config(
    page_title="Data Analysis - CSV Master",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom theme
apply_custom_theme()

# Get translation function
t = get_translation_function()

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

    /* Readiness score styling */
    .readiness-score {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
    }
    
    .readiness-high {
        color: #10B981;
    }
    
    .readiness-medium {
        color: #F59E0B;
    }
    
    .readiness-low {
        color: #EF4444;
    }

    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.25em 0.75em;
        font-size: 0.75em;
        font-weight: 500;
        border-radius: 9999px;
        text-transform: uppercase;
        margin-right: 0.5em;
    }
    
    .badge-green {
        background-color: #ECFDF5;
        color: #065F46;
    }
    
    .badge-yellow {
        background-color: #FFFBEB;
        color: #92400E;
    }
    
    .badge-red {
        background-color: #FEF2F2;
        color: #991B1B;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for data analysis
if "data_analyzed" not in st.session_state:
    st.session_state.data_analyzed = False
if "data_quality_scores" not in st.session_state:
    st.session_state.data_quality_scores = {}
if "progress" not in st.session_state:
    st.session_state.progress = {'upload': True, 'process': False, 'clean': False, 'visualize': False}
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "language" not in st.session_state:
    st.session_state.language = "en"

def detect_mixed_data_columns(df):
    """Detect columns that have a mix of numeric and text values"""
    mixed_columns = []
    
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            # Check if column has some numeric values but isn't fully numeric
            try:
                # Count how many values can be converted to numbers
                series = df[col].dropna()
                if not series.empty:
                    # Try to convert each value to float
                    numeric_mask = series.apply(lambda x: isinstance(x, (int, float)) or 
                                               (isinstance(x, str) and x.replace('.', '', 1).isdigit()))
                    numeric_count = numeric_mask.sum()
                    total_count = len(series)
                    
                    # If column has between 20% and 80% numeric values, it's likely mixed
                    numeric_percent = numeric_count / total_count * 100
                    if 20 <= numeric_percent <= 80:
                        mixed_columns.append({
                            'column': col,
                            'numeric_percent': numeric_percent,
                            'text_percent': 100 - numeric_percent
                        })
            except:
                pass
    
    return mixed_columns

def detect_date_format_inconsistencies(df):
    """Detect columns with inconsistent date formats"""
    date_format_issues = []
    
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            series = df[col].dropna().astype(str)
            if len(series) > 0:
                # Check for potential date patterns
                date_patterns = {
                    'yyyy-mm-dd': series.str.match(r'^\d{4}-\d{2}-\d{2}$').sum(),
                    'mm/dd/yyyy': series.str.match(r'^\d{1,2}/\d{1,2}/\d{4}$').sum(),
                    'dd/mm/yyyy': series.str.match(r'^\d{1,2}/\d{1,2}/\d{4}$').sum(),
                    'mm-dd-yyyy': series.str.match(r'^\d{1,2}-\d{1,2}-\d{4}$').sum(),
                    'dd-mm-yyyy': series.str.match(r'^\d{1,2}-\d{1,2}-\d{4}$').sum(),
                    'dd.mm.yyyy': series.str.match(r'^\d{1,2}\.\d{1,2}\.\d{4}$').sum()
                }
                
                # Count total potential date values
                total_date_values = sum(date_patterns.values())
                
                # If at least 30% of values look like dates but multiple formats exist
                if total_date_values / len(series) > 0.3:
                    # Get formats with at least 5% of the values
                    multiple_formats = [
                        fmt for fmt, count in date_patterns.items() 
                        if count > 0 and count / total_date_values >= 0.05
                    ]
                    
                    if len(multiple_formats) > 1:
                        date_format_issues.append({
                            'column': col,
                            'formats': multiple_formats,
                            'inconsistency_score': len(multiple_formats) - 1  # Higher score = more inconsistent
                        })
    
    return date_format_issues

def detect_decimal_separator_inconsistencies(df):
    """Detect columns with inconsistent decimal separators (. vs ,)"""
    decimal_issues = []
    
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            series = df[col].dropna().astype(str)
            if len(series) > 0:
                # Count values with . as decimal separator
                dot_decimals = series.str.match(r'^-?\d+\.\d+$').sum()
                
                # Count values with , as decimal separator
                comma_decimals = series.str.match(r'^-?\d+,\d+$').sum()
                
                # If both types are present and they make up a significant portion of the data
                total_numeric = dot_decimals + comma_decimals
                if dot_decimals > 0 and comma_decimals > 0 and total_numeric / len(series) > 0.3:
                    decimal_issues.append({
                        'column': col,
                        'dot_count': dot_decimals,
                        'comma_count': comma_decimals,
                        'inconsistency_score': min(dot_decimals, comma_decimals) / total_numeric  # Higher score = more balanced inconsistency
                    })
    
    return decimal_issues

def detect_numeric_with_units(df):
    """Detect columns with numeric values mixed with units (e.g., '10kg', '100m')"""
    unit_issues = []
    
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            series = df[col].dropna().astype(str)
            if len(series) > 0:
                # Look for patterns like number followed by text
                with_units = series.str.match(r'^\d+\.?\d*[a-zA-Z]+$').sum()
                
                # Look for numeric values without units
                pure_numeric = series.str.match(r'^\d+\.?\d*$').sum()
                
                # If we have a mix of values with and without units
                total_values = with_units + pure_numeric
                if with_units > 0 and pure_numeric > 0 and total_values / len(series) > 0.3:
                    unit_issues.append({
                        'column': col,
                        'with_units': with_units,
                        'pure_numeric': pure_numeric,
                        'inconsistency_score': min(with_units, pure_numeric) / total_values  # Higher score = more balanced inconsistency
                    })
    
    return unit_issues

def detect_string_case_inconsistencies(df):
    """Detect columns with inconsistent string casing"""
    case_issues = []
    
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            series = df[col].dropna().astype(str)
            # Only check columns with string values (skip if mostly numeric)
            numeric_values = series.str.match(r'^\d+\.?\d*$').sum()
            if len(series) > 0 and numeric_values / len(series) < 0.5:
                # Count different case patterns
                lowercase = series.str.islower().sum()
                uppercase = series.str.isupper().sum()
                titlecase = (series == series.str.title()).sum()
                mixed_case = len(series) - lowercase - uppercase - titlecase
                
                # If no single case format dominates (80%+)
                total = len(series)
                case_counts = [lowercase, uppercase, titlecase, mixed_case]
                max_case_percent = max(case_counts) / total * 100
                
                if max_case_percent < 80 and len(series) >= 5:
                    case_issues.append({
                        'column': col,
                        'lowercase_pct': lowercase / total * 100,
                        'uppercase_pct': uppercase / total * 100,
                        'titlecase_pct': titlecase / total * 100,
                        'mixedcase_pct': mixed_case / total * 100,
                        'inconsistency_score': (100 - max_case_percent) / 100  # Higher score = more inconsistent
                    })
    
    return case_issues

def detect_potential_enum_inconsistencies(df):
    """Detect columns that appear to be enumerations with inconsistent values"""
    enum_issues = []
    
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            series = df[col].dropna()
            if len(series) > 0:
                # Check if this looks like an enumeration (limited number of unique values)
                value_counts = series.value_counts()
                unique_count = len(value_counts)
                
                # Only consider as potential enum if it has between 2-15 unique values
                # and these unique values make up at least 90% of the data
                if 2 <= unique_count <= 15:
                    # Look for similar but not identical values (potential inconsistencies)
                    similar_values = []
                    unique_values = value_counts.index.tolist()
                    
                    # Convert all to lowercase for comparison
                    lowercase_values = [str(v).lower() for v in unique_values]
                    
                    # Find values that are similar (differ only by case, spacing, or minor spelling)
                    for i, val1 in enumerate(lowercase_values):
                        for j, val2 in enumerate(lowercase_values[i+1:], i+1):
                            # Check for case differences
                            if val1 == val2 and unique_values[i] != unique_values[j]:
                                similar_values.append((unique_values[i], unique_values[j], 'case'))
                            
                            # Check for spacing/punctuation differences
                            elif val1.replace(' ', '').replace('-', '').replace('_', '') == \
                                 val2.replace(' ', '').replace('-', '').replace('_', ''):
                                similar_values.append((unique_values[i], unique_values[j], 'spacing/punctuation'))
                            
                            # Check for very similar values using string distance
                            elif len(val1) > 3 and len(val2) > 3:
                                # Simple check: if they have 80% characters in common
                                common_chars = sum(1 for c in val1 if c in val2)
                                if common_chars / len(val1) > 0.8:
                                    similar_values.append((unique_values[i], unique_values[j], 'similar'))
                    
                    if similar_values:
                        enum_issues.append({
                            'column': col,
                            'similar_values': similar_values,
                            'unique_count': unique_count,
                            'inconsistency_score': len(similar_values) / unique_count  # Higher score = more inconsistencies
                        })
    
    return enum_issues

def calculate_readiness_score(df):
    """Calculate data readiness score based on various quality factors from Main.py logic"""
    # Get metrics from Main.py approach
    missing_values = df.isna().sum()
    missing_total = missing_values.sum()
    missing_percent = (missing_total / df.size * 100) if df.size > 0 else 0
    
    duplicate_rows = df.duplicated().sum()
    duplicate_percent = (duplicate_rows / len(df) * 100) if len(df) > 0 else 0
    
    text_issues = detect_text_issues(df)
    text_issues_total = text_issues.sum().sum()
    
    outlier_counts = df.apply(detect_outliers)
    total_outliers = outlier_counts.sum()
    outlier_percent = (total_outliers / df.size * 100) if df.size > 0 else 0
    
    # Enhanced detection for data type and format issues
    
    # 1. Check for data type issues (numeric data stored as strings)
    type_issues = 0
    type_issues_columns = []
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            # Check if column might be numeric
            try:
                numeric_conversion = pd.to_numeric(df[col], errors='coerce')
                if numeric_conversion.notna().mean() > 0.8:  # More than 80% convertible
                    type_issues += 1
                    type_issues_columns.append(col)
            except:
                pass
    type_issues_percent = (type_issues / len(df.columns) * 100) if len(df.columns) > 0 else 0
    
    # 2. Check for potential date columns stored as text
    date_type_issues = 0
    date_columns = []
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            # Try to convert to datetime
            try:
                datetime_conversion = pd.to_datetime(df[col], errors='coerce')
                if datetime_conversion.notna().mean() > 0.8:  # More than 80% convertible
                    date_type_issues += 1
                    date_columns.append(col)
            except:
                pass
    
    # 3. Check for boolean values stored as strings
    bool_type_issues = 0
    bool_columns = []
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            # Check if column contains boolean-like values
            series = df[col].dropna().astype(str).str.lower()
            if len(series) > 0:
                bool_values = series.isin(['true', 'false', 'yes', 'no', 'y', 'n', '1', '0', 't', 'f'])
                if bool_values.mean() > 0.8:  # More than 80% boolean-like
                    bool_type_issues += 1
                    bool_columns.append(col)
    
    # 4. Check for mixed data columns
    mixed_columns = detect_mixed_data_columns(df)
    mixed_column_count = len(mixed_columns)
    mixed_column_percent = (mixed_column_count / len(df.columns) * 100) if len(df.columns) > 0 else 0
    
    # 5. New: Check for date format inconsistencies
    date_format_issues = detect_date_format_inconsistencies(df)
    date_format_issues_count = len(date_format_issues)
    
    # 6. New: Check for decimal separator inconsistencies
    decimal_issues = detect_decimal_separator_inconsistencies(df)
    decimal_issues_count = len(decimal_issues)
    
    # 7. New: Check for numeric values with inconsistent units
    unit_issues = detect_numeric_with_units(df)
    unit_issues_count = len(unit_issues)
    
    # 8. New: Check for string case inconsistencies
    case_issues = detect_string_case_inconsistencies(df)
    case_issues_count = len(case_issues)
    
    # 9. New: Check for potential enum inconsistencies
    enum_issues = detect_potential_enum_inconsistencies(df)
    enum_issues_count = len(enum_issues)
    
    # Calculate combined type issue score (more comprehensive than before)
    total_type_issues = type_issues + date_type_issues + bool_type_issues
    total_type_issues_percent = (total_type_issues / len(df.columns) * 100) if len(df.columns) > 0 else 0
    
    # Calculate combined format issue score (more comprehensive than before)
    format_issues_count = mixed_column_count + date_format_issues_count + decimal_issues_count + unit_issues_count + case_issues_count + enum_issues_count
    format_issues_weight = (
        mixed_column_count * 15 +  # Most severe: mixed data types
        date_format_issues_count * 10 +  # Very severe: inconsistent date formats
        decimal_issues_count * 10 +  # Very severe: inconsistent decimal separators
        unit_issues_count * 8 +  # Severe: inconsistent units
        enum_issues_count * 8 +  # Severe: inconsistent enumerations
        case_issues_count * 5  # Less severe: case inconsistencies
    )
    
    # Calculate completeness score (25% weight)
    completeness_pct = 100 - missing_percent
    completeness_score = 25 * (completeness_pct / 100)
    
    # Calculate consistency score (20% weight) - based on duplicates and column issues
    dup_cols = sum(list(df.columns).count(col) > 1 for col in df.columns)
    consistency_pct = 100 - (duplicate_percent + (dup_cols * 5))  # 5% penalty per duplicate column
    consistency_score = 20 * max(0, consistency_pct / 100)
    
    # Calculate validity score (15% weight) - based on outliers and text issues
    validity_pct = 100 - (outlier_percent + (text_issues_total / df.size * 100 if df.size > 0 else 0))
    validity_score = 15 * max(0, validity_pct / 100)
    
    # Calculate data type correctness score (15% weight) - NOW MORE COMPREHENSIVE
    data_type_pct = 100 - total_type_issues_percent
    data_type_score = 15 * (data_type_pct / 100)
    
    # Calculate data format consistency score (25% weight) - NOW MORE COMPREHENSIVE
    # Apply penalties based on weighted format issues
    if format_issues_count > 0:
        data_format_pct = max(0, 100 - format_issues_weight)
    else:
        data_format_pct = 100
        
    data_format_score = 25 * (data_format_pct / 100)
    
    # Calculate total score (rounded to nearest integer)
    total_score = round(completeness_score + consistency_score + validity_score + data_type_score + data_format_score)
    total_score = max(0, min(100, total_score))  # Ensure score is between 0-100
    
    # Additional penalty for high severity issues
    # Count high severity issues
    high_severity_count = mixed_column_count + date_format_issues_count + decimal_issues_count
    missing_high_severity = sum(1 for col in df.columns if df[col].isna().mean() > 0.2)
    high_severity_count += missing_high_severity
    
    # Apply additional penalty: -5 points for each high severity issue
    high_severity_penalty = min(25, high_severity_count * 5)  # Cap at 25 points to avoid negative scores
    total_score = max(0, total_score - high_severity_penalty)
    
    # Ensure excellent score (85%+) is impossible with ANY mixed data columns or severe format issues
    if (mixed_column_count > 0 or date_format_issues_count > 0 or decimal_issues_count > 0) and total_score > 75:
        total_score = 75  # Cap at 75 if severe format issues are present
    
    # Return the score and detailed metrics
    quality_metrics = {
        "completeness": round(completeness_pct, 1),
        "consistency": round(consistency_pct, 1),
        "validity": round(validity_pct, 1),
        "data_type_correctness": round(data_type_pct, 1),
        "data_format_consistency": round(data_format_pct, 1),
        "missing_values": int(missing_total),
        "duplicate_rows": int(duplicate_rows),
        "outliers": int(total_outliers),
        "text_issues": int(text_issues_total),
        "type_issues": int(total_type_issues),
        "type_issues_columns": type_issues_columns,
        "date_columns": date_columns,
        "bool_columns": bool_columns,
        "mixed_columns": mixed_columns,
        "mixed_column_count": mixed_column_count,
        "date_format_issues": date_format_issues,
        "date_format_issues_count": date_format_issues_count,
        "decimal_issues": decimal_issues,
        "decimal_issues_count": decimal_issues_count,
        "unit_issues": unit_issues,
        "unit_issues_count": unit_issues_count,
        "case_issues": case_issues,
        "case_issues_count": case_issues_count,
        "enum_issues": enum_issues,
        "enum_issues_count": enum_issues_count,
        "high_severity_count": high_severity_count,
        "high_severity_penalty": high_severity_penalty
    }
    
    return total_score, quality_metrics

def detect_text_issues(df):
    """Detect text issues in the dataframe - copied from Main.py"""
    text_issues = pd.DataFrame(index=df.columns, columns=[
        'leading_spaces', 'trailing_spaces', 'extra_spaces',
        'punctuation', 'non_capitalized', 'duplicate_column'
    ])
    
    for col in df.columns:
        # Column duplication check
        text_issues.loc[col, 'duplicate_column'] = list(df.columns).count(col) > 1
        
        if pd.api.types.is_string_dtype(df[col]):
            series = df[col].dropna()
            if not series.empty:
                # Leading spaces
                text_issues.loc[col, 'leading_spaces'] = series.str.startswith(' ').sum()
                # Trailing spaces
                text_issues.loc[col, 'trailing_spaces'] = series.str.endswith(' ').sum()
                # Extra spaces
                text_issues.loc[col, 'extra_spaces'] = series.str.contains(r'\s{2,}').sum()
                # Punctuation
                text_issues.loc[col, 'punctuation'] = series.str.contains(r'[^\w\s]').sum()
                # Capitalization check (if all words are properly capitalized)
                text_issues.loc[col, 'non_capitalized'] = (
                    series != series.str.title()
                ).sum()
                
    return text_issues.fillna(0)

def detect_outliers(series):
    """Detect outliers in a series - copied from Main.py"""
    # Check if the series is already numeric
    if pd.api.types.is_numeric_dtype(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()
    else:
        # Attempt to convert to numeric if possible
        converted = pd.to_numeric(series, errors='coerce')
        # Only proceed if we have some numeric values
        if converted.notna().sum() > 0:
            Q1 = converted.quantile(0.25)
            Q3 = converted.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return ((converted < lower_bound) | (converted > upper_bound)).sum()
    return 0

def identify_data_issues(df):
    """Identify specific data issues"""
    issues = []
    
    # Check for missing values by column
    missing_cols = df.columns[df.isna().any()].tolist()
    for col in missing_cols:
        missing_pct = df[col].isna().mean() * 100
        if missing_pct > 0:
            severity = "high" if missing_pct > 20 else "medium" if missing_pct > 5 else "low"
            issues.append({
                "type": "missing_values",
                "column": col, 
                "description": f"Column '{col}' has {missing_pct:.1f}% missing values",
                "severity": severity
            })
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        dup_pct = duplicates / len(df) * 100
        severity = "high" if dup_pct > 10 else "medium" if dup_pct > 2 else "low"
        issues.append({
            "type": "duplicates",
            "description": f"Dataset contains {duplicates} duplicate rows ({dup_pct:.1f}%)",
            "severity": severity
        })
    
    # Check for potential data type issues
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            # Check if column might be numeric
            try:
                # Try to convert to numeric and see if it works for most values
                numeric_conversion = pd.to_numeric(df[col], errors='coerce')
                if numeric_conversion.notna().mean() > 0.8:  # More than 80% convertible
                    issues.append({
                        "type": "data_type",
                        "column": col,
                        "description": f"Column '{col}' contains numeric values stored as text",
                        "severity": "medium"
                    })
            except:
                pass
            
            # Check if column might be dates
            try:
                # Try to convert to datetime
                datetime_conversion = pd.to_datetime(df[col], errors='coerce')
                if datetime_conversion.notna().mean() > 0.8:  # More than 80% convertible
                    issues.append({
                        "type": "data_type",
                        "column": col,
                        "description": f"Column '{col}' contains date values stored as text",
                        "severity": "medium"
                    })
            except:
                pass
            
            # Check if column might be boolean
            series = df[col].dropna().astype(str).str.lower()
            if len(series) > 0:
                bool_values = series.isin(['true', 'false', 'yes', 'no', 'y', 'n', '1', '0', 't', 'f'])
                if bool_values.mean() > 0.8:  # More than 80% boolean-like
                    issues.append({
                        "type": "data_type",
                        "column": col,
                        "description": f"Column '{col}' contains boolean values stored as text",
                        "severity": "low"
                    })
    
    # Check for mixed data columns
    mixed_columns = detect_mixed_data_columns(df)
    for mixed_col in mixed_columns:
        column = mixed_col['column']
        numeric_percent = mixed_col['numeric_percent']
        text_percent = mixed_col['text_percent']
        issues.append({
            "type": "mixed_data",
            "column": column,
            "description": f"Column '{column}' has mixed data types ({numeric_percent:.1f}% numeric, {text_percent:.1f}% text)",
            "severity": "high"  # Mixed data is a serious issue that needs to be addressed
        })
    
    # Check for date format inconsistencies
    date_format_issues = detect_date_format_inconsistencies(df)
    for date_issue in date_format_issues:
        column = date_issue['column']
        formats = date_issue['formats']
        issues.append({
            "type": "date_format",
            "column": column,
            "description": f"Column '{column}' has inconsistent date formats: {', '.join(formats)}",
            "severity": "high"  # Date format inconsistency is a serious issue
        })
    
    # Check for decimal separator inconsistencies
    decimal_issues = detect_decimal_separator_inconsistencies(df)
    for decimal_issue in decimal_issues:
        column = decimal_issue['column']
        dot_count = decimal_issue['dot_count']
        comma_count = decimal_issue['comma_count']
        issues.append({
            "type": "decimal_format",
            "column": column,
            "description": f"Column '{column}' has inconsistent decimal separators ({dot_count} with '.' and {comma_count} with ',')",
            "severity": "high"  # Decimal inconsistency is a serious issue
        })
    
    # Check for numeric values with inconsistent units
    unit_issues = detect_numeric_with_units(df)
    for unit_issue in unit_issues:
        column = unit_issue['column']
        with_units = unit_issue['with_units']
        pure_numeric = unit_issue['pure_numeric']
        issues.append({
            "type": "unit_inconsistency",
            "column": column,
            "description": f"Column '{column}' has inconsistent units ({with_units} values with units, {pure_numeric} without)",
            "severity": "medium"
        })
    
    # Check for string case inconsistencies
    case_issues = detect_string_case_inconsistencies(df)
    for case_issue in case_issues:
        column = case_issue['column']
        issues.append({
            "type": "case_inconsistency",
            "column": column,
            "description": f"Column '{column}' has inconsistent string casing",
            "severity": "low"
        })
    
    # Check for potential enum inconsistencies
    enum_issues = detect_potential_enum_inconsistencies(df)
    for enum_issue in enum_issues:
        column = enum_issue['column']
        similar_values = enum_issue['similar_values']
        # Only show a few examples if there are many
        example_values = similar_values[:3]
        example_str = ", ".join([f"'{v1}' vs '{v2}'" for v1, v2, _ in example_values])
        issues.append({
            "type": "enum_inconsistency",
            "column": column,
            "description": f"Column '{column}' has similar but inconsistent categorical values (e.g., {example_str})",
            "severity": "medium"
        })
    
    # Check for outliers in numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        try:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                outlier_pct = outliers / len(df) * 100
                severity = "high" if outlier_pct > 10 else "medium" if outlier_pct > 2 else "low"
                issues.append({
                    "type": "outliers",
                    "column": col,
                    "description": f"Column '{col}' contains {outliers} outliers ({outlier_pct:.1f}%)",
                    "severity": severity
                })
        except:
            pass
    
    return issues

def get_recommendation(readiness_score, issues, multiple_files=False):
    """Get recommendation based on readiness score and issues"""
    processing_note = " after processing" if multiple_files else ""
    
    # Generate specific action recommendations based on issues
    action_recommendations = []
    
    # Check for mixed data issues and add recommendation (prioritize these as most critical)
    has_mixed_data = any(issue["type"] == "mixed_data" for issue in issues)
    if has_mixed_data:
        # Get the list of columns with mixed data
        mixed_columns = [issue["column"] for issue in issues if issue["type"] == "mixed_data"]
        if len(mixed_columns) == 1:
            action_recommendations.append(f"• Split column '{mixed_columns[0]}' into separate numeric and text columns or standardize format")
        else:
            action_recommendations.append(f"• Fix mixed data columns ({', '.join(mixed_columns[:3])}{', ...' if len(mixed_columns) > 3 else ''}) by splitting or standardizing formats")
    
    # Check for type issues and add recommendation
    has_type_issues = any(issue["type"] == "data_type" for issue in issues)
    if has_type_issues:
        action_recommendations.append("• Convert text columns with numeric content to appropriate numeric types")
    
    # Check for missing values and add recommendation
    has_missing_values = any(issue["type"] == "missing_values" for issue in issues)
    if has_missing_values:
        action_recommendations.append("• Fill or handle missing values in affected columns")
    
    # Check for outliers and add recommendation
    has_outliers = any(issue["type"] == "outliers" for issue in issues)
    if has_outliers:
        action_recommendations.append("• Address outliers in numeric columns (review or filter)")
    
    # Check for duplicates and add recommendation
    has_duplicates = any(issue["type"] == "duplicates" for issue in issues)
    if has_duplicates:
        action_recommendations.append("• Remove duplicate rows to ensure data integrity")
    
    # Add default recommendations if none were generated
    if not action_recommendations and readiness_score < 85:
        action_recommendations = ["• Review data structure and formats", "• Check for inconsistencies"]
    
    # Join all recommendations
    specific_recommendations = "\n\n".join(action_recommendations) if action_recommendations else ""
    
    # New thresholds aligned with stricter scoring
    if readiness_score >= 85:
        return {
            "action": "visualize",
            "description": f"Your data is in excellent condition! You can proceed directly to visualization{processing_note}.",
            "next_step": "pages/4_Dashboard.py",
            "specific_recommendations": specific_recommendations
        }
    elif readiness_score >= 65:  # Changed from 60 to 65
        return {
            "action": "manual_clean",
            "description": f"Your data has some issues that should be addressed. Manual cleaning is recommended{processing_note}.",
            "next_step": "pages/3_Clean.py",
            "specific_recommendations": specific_recommendations
        }
    else:
        return {
            "action": "ai_clean",
            "description": f"Your data has significant quality issues. AI-powered cleaning is recommended for best results{processing_note}.",
            "next_step": "pages/3_Clean.py",
            "clean_mode": "ai",
            "specific_recommendations": specific_recommendations
        }

def analyze_all_dataframes():
    """Analyze all dataframes in the session state"""
    if not st.session_state.dataframes:
        return
    
    # Clear previous analysis
    st.session_state.data_quality_scores = {}
    
    # Check if there are multiple files
    multiple_files = len(st.session_state.dataframes) > 1
    
    # Analyze each dataframe
    for filename, df in st.session_state.dataframes.items():
        # Calculate readiness score
        readiness_score, quality_metrics = calculate_readiness_score(df)
        
        # Identify issues
        issues = identify_data_issues(df)
        
        # Get recommendation
        recommendation = get_recommendation(readiness_score, issues, multiple_files)
        
        # Store results
        st.session_state.data_quality_scores[filename] = {
            "readiness_score": readiness_score,
            "quality_metrics": quality_metrics,
            "issues": issues,
            "recommendation": recommendation
        }
    
    # Mark as analyzed
    st.session_state.data_analyzed = True

def main():
    # Initialize Streamlit page with progress bar
    cols = st.columns(5)
    with cols[0]:
        st.markdown(f"#### 1. {t('upload')} ✅")
    with cols[1]:
        st.markdown(f"#### 2. {t('Analysis')} 🔄")
    with cols[2]:
        # Show process stage based on if we've gone through it
        if st.session_state.get('progress', {}).get('process', False):
            st.markdown(f"#### 3. {t('process')} ✅")
        else:
            st.markdown(f"#### 3. {t('process')}")
    with cols[3]:
        # Show clean stage based on if we've gone through it
        if st.session_state.get('progress', {}).get('clean', False):
            st.markdown(f"#### 4. {t('clean')} ✅") 
        else:
            st.markdown(f"#### 4. {t('clean')}")
    with cols[4]:
        st.markdown(f"#### 5. {t('visualize')}")
    
    # Determine where we are in the workflow
    coming_from_clean = (st.session_state.get('progress', {}).get('clean', False) == True)
    coming_from_process = (st.session_state.get('progress', {}).get('process', False) == True and not coming_from_clean)
    
    # Adjust progress based on where we are in the workflow
    if coming_from_clean:
        # If coming from cleaning, we're at 80% progress (4 of 5 steps)
        st.progress(0.8)
    elif coming_from_process:
        # If coming from processing, we're at 60% progress (3 of 5 steps)
        st.progress(0.6)
    else:
        # Just coming from upload, we're at 25% progress (1.25 of 5 steps)
        st.progress(0.25)
    
    # Check if there are uploaded files
    if "dataframes" not in st.session_state or not st.session_state.dataframes:
        st.info(t('no_files_uploaded'))
        if st.button(t('back_to_upload'), key="no_files_back", use_container_width=True):
            st.switch_page("Home.py")
        return
    
    # Main page title - adjust based on context
    if coming_from_clean:
        st.markdown("# Cleaned Data Analysis")
        st.success("Your data has been cleaned and is now being analyzed again to verify improvements.")
    elif coming_from_process:
        st.markdown("# Processed Data Analysis") 
        st.success("Your data has been processed and is now being analyzed to check quality before cleaning.")
    else:
        st.markdown("# Data Quality Analysis")
    
    st.markdown("---")
    
    # Always run analysis when visiting the page to ensure latest metrics are used
    with st.spinner("Analyzing your data..."):
        # Add a slight delay to show the spinner
        time.sleep(1)
        analyze_all_dataframes()
    
    # Show analysis for each file
    file_tabs = st.tabs([f"{filename}" for filename in st.session_state.data_quality_scores.keys()])
    
    for i, (filename, analysis) in enumerate(st.session_state.data_quality_scores.items()):
        with file_tabs[i]:
            # Get the dataframe for this file
            df = st.session_state.dataframes[filename]
            
            # Display main score and summary
            score = analysis["readiness_score"]
            score_class = "readiness-high" if score >= 85 else "readiness-medium" if score >= 60 else "readiness-low"
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                <div class="readiness-score {score_class}">{score}%</div>
                <h3 style="text-align: center;">Data Readiness Score</h3>
                """, unsafe_allow_html=True)
                
                # Quality metrics
                metrics = analysis["quality_metrics"]
                st.markdown("### Data Quality Metrics")
                st.markdown(f"✓ **Completeness:** {metrics['completeness']}%")
                st.markdown(f"✓ **Consistency:** {metrics['consistency']}%")
                st.markdown(f"✓ **Validity:** {metrics['validity']}%")
                
                # Check if the new metrics exist (for backward compatibility)
                if 'data_type_correctness' in metrics:
                    st.markdown(f"✓ **Data Type Correctness:** {metrics['data_type_correctness']}%")
                
                if 'data_format_consistency' in metrics:
                    st.markdown(f"✓ **Data Format Consistency:** {metrics['data_format_consistency']}%")
                
                st.markdown(f"✓ **Missing Values:** {metrics['missing_values']}")
                st.markdown(f"✓ **Duplicate Rows:** {metrics['duplicate_rows']}")
                st.markdown(f"✓ **Outliers:** {metrics['outliers']}")
                st.markdown(f"✓ **Text Issues:** {metrics['text_issues']}")
                
                # Check if type_issues exists (for backward compatibility)
                if 'type_issues' in metrics:
                    st.markdown(f"✓ **Type Issues:** {metrics['type_issues']}")
                
                # Check if mixed column count exists (for backward compatibility)
                if 'mixed_column_count' in metrics:
                    st.markdown(f"✓ **Mixed Data Columns:** {metrics['mixed_column_count']}")
                    
                # Display high severity penalties if available
                if 'high_severity_penalty' in metrics and metrics['high_severity_penalty'] > 0:
                    st.markdown("### Score Penalties")
                    st.markdown(f"⚠️ **High Severity Issues:** -{metrics['high_severity_penalty']} points")
                    
                    if metrics['mixed_column_count'] > 0 and 'mixed_column_count' in metrics:
                        st.markdown(f"⚠️ **Mixed Data Cap:** Maximum score capped at 75% due to mixed data")
                
                # Recommendation
                recommendation = analysis["recommendation"]
                st.markdown("### Recommendation")
                st.markdown(f"**{recommendation['description']}**")
                
                # Show specific recommendations if available
                if "specific_recommendations" in recommendation and recommendation["specific_recommendations"]:
                    st.markdown("#### Suggested Actions:")
                    st.markdown(recommendation["specific_recommendations"])
            
            with col2:
                # Data preview
                st.markdown("### Data Preview")
                st.dataframe(df.head(5), height=200)
                
                # Column types table
                st.markdown("### Column Data Types")
                
                try:
                    # Prepare column types dataframe
                    column_types = []
                    for col in df.columns:
                        dtype = str(df[col].dtype)
                        inferred_type = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else \
                                       "datetime" if pd.api.types.is_datetime64_dtype(df[col]) else \
                                       "categorical" if pd.api.types.is_categorical_dtype(df[col]) else \
                                       "text"
                        
                        # Check if column is likely numeric but stored as text
                        type_issue = False
                        if pd.api.types.is_object_dtype(df[col]):
                            try:
                                numeric_conversion = pd.to_numeric(df[col], errors='coerce')
                                if numeric_conversion.notna().mean() > 0.8:  # More than 80% convertible
                                    type_issue = True
                                    inferred_type = "likely numeric"
                            except:
                                pass
                        
                        column_types.append({
                            "Column Name": col,
                            "Data Type": dtype,
                            "Inferred Type": inferred_type,
                            "Issue": "✓" if type_issue else ""
                        })
                    
                    # Convert to dataframe and display
                    col_types_df = pd.DataFrame(column_types)
                    st.dataframe(col_types_df, height=200)
                except Exception as e:
                    st.error(f"Could not display column types table: {str(e)}")
                    # Show a simpler version
                    st.dataframe(pd.DataFrame({'Column': df.columns, 'Type': df.dtypes.astype(str)}), height=200)
                
                # Data issues
                st.markdown("### Data Issues Identified")
                if analysis["issues"]:
                    for issue in analysis["issues"]:
                        badge_color = "red" if issue["severity"] == "high" else "yellow" if issue["severity"] == "medium" else "green"
                        st.markdown(f"""
                        <span class="badge badge-{badge_color}">{issue["severity"]}</span> {issue["description"]}
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("✓ No significant issues found")
            
    # Navigation buttons
    st.markdown("---")
    
    # Determine where we are in the workflow
    coming_from_clean = (st.session_state.get('progress', {}).get('clean', False) == True)
    coming_from_process = (st.session_state.get('progress', {}).get('process', False) == True and not coming_from_clean)
    
    # Case 1: Coming from cleaning - show navigation to return to cleaning or proceed to visualization
    if coming_from_clean:
        st.markdown("### Next Steps")
        st.info("Your data has been cleaned and analyzed. You can now proceed to visualization.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Cleaning", key="back_to_clean_btn", use_container_width=True):
                st.switch_page("pages/3_Clean.py")
        with col2:
            if st.button("📊 Generate Visualization Dashboard →", key="viz_btn", use_container_width=True, type="primary"):
                st.session_state.progress['visualize'] = True
                # Ensure we start at step 1 of the dashboard workflow
                st.session_state.dashboard_step = 1
                st.switch_page("pages/4_Dashboard.py")
    
    # Case 2: Coming from processing - show navigation to return to processing or proceed to cleaning
    elif coming_from_process:
        st.markdown("### Next Steps")
        st.info("Your processed data has been analyzed. You can now proceed to cleaning or return to processing.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Processing", key="back_to_process_btn", use_container_width=True):
                st.switch_page("pages/2_Process.py")
        with col2:
            if st.button("🧹 Proceed to Cleaning →", key="proceed_to_clean_btn", use_container_width=True, type="primary"):
                st.session_state.progress['process'] = True
                st.session_state.cleaning_mode = None  # Don't pre-select a cleaning mode
                st.switch_page("pages/3_Clean.py")
    
    # Case 3: Multiple files but not from process/clean - show only process option
    elif len(st.session_state.dataframes) > 1:
        st.markdown("### Next Steps")
        st.info("You have multiple files. Process them (merge/combine) before cleaning.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Upload", key="back_btn", use_container_width=True):
                st.switch_page("Home.py")
        with col2:
            if st.button("🔄 Process Multiple Files", key="process_btn", use_container_width=True, type="primary"):
                st.session_state.progress['upload'] = True
                st.session_state.progress['process'] = True
                # Reset processing_mode to None to show the method selection screen
                st.session_state.processing_mode = None
                st.switch_page("pages/2_Process.py")
    
    # Case 4: Single file not from process/clean - show back button and cleaning option
    else:
        filename = list(st.session_state.dataframes.keys())[0]
        recommendation = st.session_state.data_quality_scores[filename]["recommendation"]
        
        # Add Next Steps title to match multiple files section
        st.markdown("### Next Steps")
        
        # Only show 2 buttons if visualization is not available
        if recommendation["action"] == "visualize":
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("← Back to Upload", key="back_btn", use_container_width=True):
                    st.switch_page("Home.py")
                    
            with col2:
                # Add Clean button in the middle that takes to cleaning method page without pre-selecting a method
                if st.button("🧹 Clean Data", key="clean_btn", use_container_width=True):
                    st.session_state.progress['upload'] = True
                    st.session_state.progress['process'] = True
                    # Don't set a specific cleaning mode, let the user choose on the cleaning page
                    st.session_state.cleaning_mode = None
                    st.switch_page("pages/3_Clean.py")
                    
            with col3:
                # Show visualization button ONLY if score is excellent (85% or higher)
                if st.button("📊 Proceed to Visualization →", key="viz_btn", use_container_width=True, type="primary"):
                    st.session_state.progress['upload'] = True
                    st.session_state.progress['process'] = True 
                    st.session_state.progress['clean'] = True
                    # Ensure we start at step 1 of the dashboard workflow
                    st.session_state.dashboard_step = 1
                    st.switch_page("pages/4_Dashboard.py")
        else:
            # Only show 2 columns if visualization is not recommended
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back to Upload", key="back_btn", use_container_width=True):
                    st.switch_page("Home.py")
                    
            with col2:
                # Add Clean button that takes to cleaning method page without pre-selecting a method
                if st.button("🧹 Clean Data", key="clean_btn", use_container_width=True, type="primary"):
                    st.session_state.progress['upload'] = True
                    st.session_state.progress['process'] = True
                    # Don't set a specific cleaning mode, let the user choose on the cleaning page
                    st.session_state.cleaning_mode = None
                    st.switch_page("pages/3_Clean.py")

if __name__ == "__main__":
    main() 