import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import re
import os
import json
import streamlit as st
from datetime import datetime
import time
import math
import string
import io
import csv

# For OpenAI integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

def text_to_num(text):
    """Convert text number representation to actual number"""
    if not isinstance(text, str):
        return text
    
    # Handle simple cases
    text = text.lower().strip()
    if text == 'none' or text == 'null' or text == 'nan' or text == 'na':
        return None
    
    # Try to convert directly
    try:
        return float(text)
    except ValueError:
        pass
    
    # Try word2number conversion
    try:
        from word2number import w2n
        return w2n.word_to_num(text)
    except (ImportError, ValueError):
        return text

def is_valid_openai_key_format(api_key):
    """Check if API key has valid format"""
    if not api_key or not isinstance(api_key, str):
        return False
    
    # OpenAI API keys typically start with "sk-" and are 51 characters long
    return api_key.startswith("sk-") and len(api_key) > 40

def basic_clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform basic cleaning operations on a dataframe without using AI.
    Returns cleaned dataframe and a report of operations performed.
    """
    report = {
        "operations": [],
        "rows_before": len(df),
        "columns_before": len(df.columns)
    }
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # 1. Handle missing values
    missing_counts = cleaned_df.isna().sum()
    total_missing = missing_counts.sum()
    
    if total_missing > 0:
        # For each column with missing values, use an appropriate strategy
        for col in cleaned_df.columns:
            missing = missing_counts[col]
            if missing == 0:
                continue
                
            # Strategy depends on the data type and missing percentage
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                # For numeric columns, replace with median
                median_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(median_val)
                report["operations"].append({
                    "operation": "fill_missing",
                    "column": col,
                    "method": "median",
                    "rows_affected": int(missing),
                    "explanation": f"Filled {missing} missing values with median ({median_val:.2f})"
                })
            elif pd.api.types.is_datetime64_dtype(cleaned_df[col]):
                # For datetime, forward fill then backward fill
                cleaned_df[col] = cleaned_df[col].fillna(method='ffill').fillna(method='bfill')
                report["operations"].append({
                    "operation": "fill_missing",
                    "column": col,
                    "method": "ffill/bfill",
                    "rows_affected": int(missing),
                    "explanation": f"Filled {missing} missing datetime values using forward/backward fill"
                })
            else:
                # For categorical/text, use most common value if reasonable
                value_counts = cleaned_df[col].value_counts(dropna=True)
                if len(value_counts) > 0:
                    # Only use mode if it's not too diverse (arbitrary threshold)
                    if len(value_counts) < len(cleaned_df) * 0.5:
                        mode_val = value_counts.index[0]
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                        report["operations"].append({
                            "operation": "fill_missing",
                            "column": col,
                            "method": "mode",
                            "rows_affected": int(missing),
                            "explanation": f"Filled {missing} missing values with most common value: '{mode_val}'"
                        })
                    else:
                        # If too diverse, use a generic placeholder
                        cleaned_df[col] = cleaned_df[col].fillna("Unknown")
                        report["operations"].append({
                            "operation": "fill_missing",
                            "column": col,
                            "method": "placeholder",
                            "rows_affected": int(missing),
                            "explanation": f"Filled {missing} missing values with 'Unknown'"
                        })
    
    # 2. Remove duplicates
    dupe_count = cleaned_df.duplicated().sum()
    if dupe_count > 0:
        cleaned_df = cleaned_df.drop_duplicates()
        report["operations"].append({
            "operation": "remove_duplicates",
            "rows_affected": int(dupe_count),
            "explanation": f"Removed {dupe_count} duplicate rows"
        })
    
    # 3. Fix data types - convert text numbers to numeric
    for col in cleaned_df.columns:
        if pd.api.types.is_object_dtype(cleaned_df[col]):
            # Check if the column contains numbers written as text
            sample = cleaned_df[col].dropna().head(100).tolist()
            if len(sample) > 0:
                # Try to convert a sample to see if it's numeric
                converted = [text_to_num(x) for x in sample]
                numeric_count = sum(1 for x in converted if isinstance(x, (int, float)))
                
                # If >80% can be converted to numbers, convert the column
                if numeric_count / len(sample) > 0.8:
                    original_dtype = cleaned_df[col].dtype
                    cleaned_df[col] = cleaned_df[col].apply(text_to_num)
                    
                    # Handle any remaining missing values
                    if cleaned_df[col].isna().any():
                        cleaned_df[col] = cleaned_df[col].fillna(0)
                    
                    # Convert to appropriate numeric type
                    if all(x is None or (isinstance(x, (int, float)) and (isinstance(x, int) or x.is_integer())) for x in cleaned_df[col] if x is not None):
                        cleaned_df[col] = cleaned_df[col].astype('Int64')
                    else:
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                    
                    report["operations"].append({
                        "operation": "convert_type",
                        "column": col,
                        "from_type": str(original_dtype),
                        "to_type": str(cleaned_df[col].dtype),
                        "rows_affected": len(cleaned_df),
                        "explanation": f"Converted column from text to numeric"
                    })
            
            # Clean up string data (trim whitespace, handle case)
            if pd.api.types.is_object_dtype(cleaned_df[col]):
                # Only process string columns
                if cleaned_df[col].apply(lambda x: isinstance(x, str)).mean() > 0.8:
                    # Remove leading/trailing whitespace
                    cleaned_df[col] = cleaned_df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                    
                    # Standardize case for categorical-like columns with few unique values
                    if cleaned_df[col].nunique() < 10:
                        # For columns that look like categories, standardize to title case
                        cleaned_df[col] = cleaned_df[col].apply(lambda x: x.title() if isinstance(x, str) else x)
                        report["operations"].append({
                            "operation": "standardize_text",
                            "column": col,
                            "rows_affected": cleaned_df[col].apply(lambda x: isinstance(x, str)).sum(),
                            "explanation": f"Standardized text formatting (whitespace, case)"
                        })
    
    # Generate final report
    report["rows_after"] = len(cleaned_df)
    report["columns_after"] = len(cleaned_df.columns)
    report["rows_removed"] = report["rows_before"] - report["rows_after"]
    
    return cleaned_df, report

def clean_with_openai(df: pd.DataFrame, api_key: str, model: str = "gpt-3.5-turbo") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean a dataframe using OpenAI API to directly generate a cleaned CSV
    """
    if not OPENAI_AVAILABLE:
        st.error("OpenAI package is not installed. Use 'pip install openai' to install it.")
        return basic_clean_dataframe(df)
    
    if not api_key:
        st.error("No OpenAI API key provided.")
        return basic_clean_dataframe(df)
    
    # Configure OpenAI
    client = openai.OpenAI(api_key=api_key)
    
    # Create a fallback version using basic cleaning
    st.info("Creating fallback cleaning as a backup...")
    fallback_df, fallback_report = basic_clean_dataframe(df)
    
    try:
        # Generate a sample of the data
        sample_rows = min(10, len(df))
        sample = df.head(sample_rows)
        
        # Get comprehensive dataset info
        info = []
        problematic_values = []
        
        st.write("Analyzing dataset...")
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            unique_count = df[col].nunique()
            
            # Basic column info
            col_info = f"- {col} (type: {col_type}): {missing} missing values ({missing_pct:.2f}%), {unique_count} unique values"
            
            # Add numeric statistics if applicable
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    mean = df[col].mean()
                    median = df[col].median()
                    std = df[col].std()
                    min_val = df[col].min()
                    max_val = df[col].max()
                    col_info += f", mean: {mean:.2f}, median: {median:.2f}, std: {std:.2f}, min: {min_val:.2f}, max: {max_val:.2f}"
                except:
                    pass
            
            # Add categorical statistics if applicable
            elif pd.api.types.is_object_dtype(df[col]):
                try:
                    value_counts = df[col].value_counts(normalize=True)
                    top_values = value_counts.head(3).items()
                    col_info += f", most common values: {', '.join([f'{val} ({pct:.1%})' for val, pct in top_values])}"
                except:
                    pass
            
            info.append(col_info)
        
        # Check for common data issues
        total_missing = df.isna().sum().sum()
        if total_missing > 0:
            problematic_values.append(f"Dataset contains {total_missing} missing values across all columns")
        
        # Determine if the dataset is small enough to send directly
        # For larger datasets, we'll send a sample and instructions
        is_small_dataset = len(df) <= 100 and len(df.columns) <= 20
        
        if is_small_dataset:
            # For small datasets, we can send the entire CSV
            csv_data = df.to_csv(index=False)
            data_to_send = f"Complete dataset ({len(df)} rows):\n{csv_data}"
        else:
            # For larger datasets, send a sample and summary
            csv_sample = df.head(20).to_csv(index=False)
            data_to_send = f"Sample of dataset (first 20 of {len(df)} rows):\n{csv_sample}"
        
        # Create prompt for OpenAI with requirements restored
        prompt = f"""You are a data cleaning expert. I need you to clean this dataset and return the CLEANED CSV directly.
        
        Dataset Summary:
        - Total rows: {len(df)}
        - Total columns: {len(df.columns)}
        - Columns: {', '.join(df.columns)}
        
        {data_to_send}
        
        Detailed column statistics:
        {chr(10).join(info)}
        
        Identified data issues that need addressing:
        {chr(10).join(problematic_values) if problematic_values else "No major issues detected automatically, but please analyze the data carefully for hidden problems."}
        
        REQUIREMENTS:
        1. Clean this dataset thoroughly by:
           - Removing duplicate rows
           - Handling ALL missing/null values (nothing should be left empty)
           - Converting columns to the most appropriate data types
           - Converting text number representations (like "forty", "twenty-five") to actual numbers
           - Standardizing date formats to be consistent
           - Standardizing categorical values (e.g., gender: 'male'/'female' instead of 'm'/'f')
           - Fixing any inconsistent text formatting (case, extra spaces, special characters)
           - Removing or handling outliers in numeric columns when appropriate
        
        2. Return ONLY the cleaned CSV data in the following format:
           column1,column2,column3,...
           value1,value2,value3,...
           ...
        
        3. DO NOT include any explanations, code, or other text - ONLY return the cleaned CSV data.
        
        4. If the dataset is too large, clean the entire dataset using the same logic you would apply to the sample.
        """
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data cleaning expert that returns only cleaned CSV data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4000
        )
        
        # Extract the CSV from the response
        cleaned_csv = response.choices[0].message.content.strip()
        
        # Convert the CSV string back to a DataFrame
        try:
            # Use StringIO to create a file-like object from the string
            csv_io = io.StringIO(cleaned_csv)
            df_cleaned = pd.read_csv(csv_io)
            
            # Generate a cleaning report
            rows_removed = len(df) - len(df_cleaned)
            cleaning_report = {
                "operations": ["AI-powered cleaning completed successfully"],
                "rows_before": len(df),
                "rows_after": len(df_cleaned),
                "rows_removed": rows_removed,
                "columns_before": len(df.columns),
                "columns_after": len(df_cleaned.columns)
            }
            
            return df_cleaned, cleaning_report
            
        except Exception as e:
            st.error(f"Error processing the cleaned CSV: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")
            
            return fallback_df, fallback_report
            
    except Exception as e:
        st.error(f"Error with OpenAI processing: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        
        return fallback_df, fallback_report

def process_dataframes_with_ai(dataframes: Dict[str, pd.DataFrame], api_key: str, model: str = "gpt-3.5-turbo") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyze multiple dataframes and use OpenAI to determine the best way to combine or merge them.
    OpenAI will directly return the combined/merged CSV data, similar to the cleaning function.
    
    Args:
        dataframes: Dictionary of dataframes with filenames as keys
        api_key: OpenAI API key
        model: OpenAI model to use
        
    Returns:
        Tuple containing the processed dataframe and a report
    """
    if not OPENAI_AVAILABLE:
        st.error("OpenAI package is not installed. Use 'pip install openai' to install it.")
        return pd.DataFrame(), {"error": "OpenAI not available"}
    
    if not api_key:
        st.error("No OpenAI API key provided.")
        return pd.DataFrame(), {"error": "No API key provided"}
    
    if len(dataframes) < 2:
        st.error("At least two dataframes are required for processing.")
        return pd.DataFrame(), {"error": "Not enough dataframes"}
    
    # Configure OpenAI
    client = openai.OpenAI(api_key=api_key)
    
    try:
        # Generate dataset summaries
        summaries = []
        original_rows_total = 0
        original_columns_total = 0
        
        for file_name, df in dataframes.items():
            # Get basic info about the dataframe
            original_rows_total += len(df)
            original_columns_total += len(df.columns)
            
            # Generate a CSV sample of the data
            csv_sample = df.head(5).to_csv(index=False)
            
            summary = {
                "file_name": file_name,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "sample_data": csv_sample
            }
            summaries.append(summary)
        
        # Create a prompt for OpenAI to analyze and combine the datasets directly
        prompt = f"""You are a data processing expert. I need you to combine multiple datasets into one and return ONLY the combined CSV data.

Dataset Information:
"""
        
        for i, summary in enumerate(summaries):
            prompt += f"""
Dataset {i+1}: {summary['file_name']}
- Rows: {summary['rows']}
- Columns: {summary['columns']}
- Column names: {', '.join(summary['column_names'])}
- Sample data:
{summary['sample_data']}
"""
        
        prompt += """
TASK:
1. Analyze these datasets and determine the best way to combine them:
   - Vertical concatenation (stacking datasets with similar columns)
   - Horizontal concatenation (joining datasets side by side)
   - Merging on common keys

2. Apply the best combination method intelligently:
   - For vertical concatenation: Map similar columns across datasets
   - For horizontal concatenation: Handle column conflicts
   - For merging: Identify appropriate join keys and join type

3. Return ONLY the combined CSV data with no explanations, JSON, or other text.

REQUIREMENTS:
- The output should be valid CSV format only
- Do not include any explanations or analysis
- Include headers in the first row
- Ensure data types are preserved appropriately
- Return the COMPLETE combined dataset, not just a sample
"""
        
        # Prepare small datasets for complete processing
        total_size = sum(len(df) for df in dataframes.values())
        if total_size <= 200:  # For small datasets, we can include full data
            prompt += "\nHere are the complete datasets for more accurate processing:\n"
            for file_name, df in dataframes.items():
                full_csv = df.to_csv(index=False)
                prompt += f"\nComplete data for {file_name}:\n{full_csv}\n"
        
        # Call OpenAI API
        st.write("Processing datasets with AI...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data processing expert that returns only CSV data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4000
        )
        
        # Extract the CSV from the response
        result_csv = response.choices[0].message.content.strip()
        
        # Parse the CSV and convert to DataFrame
        try:
            # Use StringIO to create a file-like object from the string
            csv_io = io.StringIO(result_csv)
            result_df = pd.read_csv(csv_io)
            
            # Generate a basic report for the user interface
            operation_type = "ai_processing"
            if result_df.shape[0] > original_rows_total and result_df.shape[1] < sum(len(df.columns) for df in dataframes.values()):
                operation_type = "vertical_concatenation"
            elif result_df.shape[1] > original_columns_total:
                operation_type = "horizontal_concatenation"
            else:
                operation_type = "merge"
            
            # Create a simple report
            report = {
                "operation": operation_type,
                "files_processed": list(dataframes.keys()),
                "total_rows_before": original_rows_total,
                "total_rows_after": len(result_df),
                "total_columns_before": original_columns_total,
                "total_columns_after": len(result_df.columns),
                "ai_explanation": "AI processed and combined the datasets directly."
            }
            
            return result_df, report
            
        except Exception as e:
            st.error(f"Error processing the combined CSV: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")
            st.code(result_csv)
            
            return pd.DataFrame(), {"error": f"Processing error: {str(e)}"}
            
    except Exception as e:
        st.error(f"Error with OpenAI processing: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        
        return pd.DataFrame(), {"error": f"OpenAI error: {str(e)}"}

def clean_dataframe_with_ai(df: pd.DataFrame, openai_api_key: Optional[str] = None, use_basic_cleaning: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean a dataframe using AI or basic cleaning
    """
    # Validate OpenAI API key if provided
    has_valid_key = False
    if openai_api_key and is_valid_openai_key_format(openai_api_key):
        has_valid_key = True
    
    # Use OpenAI if a valid key is provided and not explicitly using basic cleaning
    if has_valid_key and not use_basic_cleaning:
        try:
            st.info("Using OpenAI for advanced cleaning...")
            return clean_with_openai(df, openai_api_key)
        except Exception as e:
            st.error(f"OpenAI cleaning failed: {str(e)}. Falling back to basic cleaning.")
            return basic_clean_dataframe(df)
    
    # Otherwise, fall back to basic cleaning
    st.warning("No AI method available, using basic cleaning")
    return basic_clean_dataframe(df)

def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze a dataset and return statistics
    """
    # Basic metrics
    rows = len(df)
    columns = len(df.columns)
    missing_values = df.isna().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    # Column-specific statistics
    column_stats = []
    for col in df.columns:
        col_stat = {
            "name": col,
            "type": str(df[col].dtype),
            "missing": int(df[col].isna().sum()),
            "missing_pct": float(df[col].isna().sum() / rows * 100),
            "unique": int(df[col].nunique())
        }
        
        # Add type-specific statistics
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stat["mean"] = float(df[col].mean())
            col_stat["median"] = float(df[col].median())
            col_stat["std"] = float(df[col].std())
            col_stat["min"] = float(df[col].min())
            col_stat["max"] = float(df[col].max())
        elif pd.api.types.is_object_dtype(df[col]):
            # For categorical, get most common values
            most_common = df[col].value_counts().iloc[0] if df[col].value_counts().size > 0 else 0
            most_common_val = df[col].value_counts().index[0] if df[col].value_counts().size > 0 else "N/A"
            
            col_stat["most_common"] = str(most_common_val)
            col_stat["most_common_count"] = int(most_common)
        
        column_stats.append(col_stat)
    
    # Return the complete analysis
    return {
        "rows": rows,
        "columns": columns,
        "missing_values": int(missing_values),
        "duplicate_rows": int(duplicate_rows),
        "column_stats": column_stats
    }

def suggest_visualizations(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Suggest appropriate visualizations based on dataset analysis
    """
    suggestions = []
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
    # Histograms for numeric columns
    for col in numeric_cols[:5]:  # Limit to first 5 to avoid too many suggestions
        suggestions.append({
            "type": "histogram",
            "column": col,
            "title": f"Distribution of {col}"
        })
    
    # Bar plots for categorical columns
    for col in categorical_cols:
        if df[col].nunique() <= 20:  # Only suggest for columns with reasonable number of categories
            suggestions.append({
                "type": "bar",
                "column": col,
                "title": f"Value Counts of {col}"
            })
    
    # Correlation heatmap if multiple numeric columns exist
    if len(numeric_cols) > 1:
        suggestions.append({
            "type": "heatmap",
            "columns": list(numeric_cols),
            "title": "Correlation Heatmap"
        })
    
    return suggestions 