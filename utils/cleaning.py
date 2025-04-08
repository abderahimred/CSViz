import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import streamlit as st
from datetime import datetime
import re

# For OpenAI integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

def basic_clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform basic cleaning operations on a dataframe without using AI
    """
    df_cleaned = df.copy()
    cleaning_report = {
        "operations": [],
        "rows_before": len(df),
        "columns_before": len(df.columns),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 1. Remove duplicate rows
    duplicates = df_cleaned.duplicated()
    if duplicates.sum() > 0:
        df_cleaned = df_cleaned.drop_duplicates()
        cleaning_report["operations"].append({
            "operation": "remove_duplicates",
            "rows_affected": int(duplicates.sum()),
            "explanation": "Removed duplicate rows"
        })
    
    # 2. Handle missing values
    for col in df_cleaned.columns:
        missing = df_cleaned[col].isna().sum()
        if missing > 0:
            # For numeric columns, fill with median
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                median_val = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(median_val)
                cleaning_report["operations"].append({
                    "operation": "fill_missing",
                    "column": col,
                    "method": "median",
                    "rows_affected": int(missing),
                    "explanation": f"Filled missing values with median ({median_val:.2f})"
                })
            # For categorical/text columns, fill with mode
            else:
                mode_val = df_cleaned[col].mode()[0] if not df_cleaned[col].empty else "Unknown"
                df_cleaned[col] = df_cleaned[col].fillna(mode_val)
                cleaning_report["operations"].append({
                    "operation": "fill_missing",
                    "column": col,
                    "method": "mode",
                    "rows_affected": int(missing),
                    "explanation": f"Filled missing values with mode ({mode_val})"
                })
    
    # 3. Standardize column names
    old_columns = df_cleaned.columns.tolist()
    new_columns = [col.lower().replace(' ', '_') for col in old_columns]
    df_cleaned.columns = new_columns
    
    # 4. Convert numeric strings to numbers
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            try:
                numeric_col = pd.to_numeric(df_cleaned[col], errors='coerce')
                # If more than 80% of values can be converted to numeric, do the conversion
                if numeric_col.notna().sum() / len(numeric_col) > 0.8:
                    df_cleaned[col] = numeric_col
                    cleaning_report["operations"].append({
                        "operation": "convert_to_numeric",
                        "column": col,
                        "rows_affected": int(numeric_col.notna().sum()),
                        "explanation": "Converted string values to numeric type"
                    })
            except:
                pass
    
    # 5. Remove outliers from numeric columns
    for col in df_cleaned.select_dtypes(include=[np.number]).columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
        if outliers > 0 and outliers < len(df_cleaned) * 0.1:  # Don't remove more than 10% of data
            df_cleaned = df_cleaned[
                (df_cleaned[col] >= lower_bound) & 
                (df_cleaned[col] <= upper_bound)
            ]
            cleaning_report["operations"].append({
                "operation": "remove_outliers",
                "column": col,
                "rows_affected": int(outliers),
                "explanation": f"Removed {outliers} outliers"
            })
    
    # Update final stats
    cleaning_report["rows_after"] = len(df_cleaned)
    cleaning_report["columns_after"] = len(df_cleaned.columns)
    cleaning_report["rows_removed"] = cleaning_report["rows_before"] - cleaning_report["rows_after"]
    
    return df_cleaned, cleaning_report

def clean_dataframe_with_ai(df: pd.DataFrame, openai_api_key: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean a dataframe using OpenAI's API for advanced cleaning
    """
    if not OPENAI_AVAILABLE:
        st.error("OpenAI package is not installed. Using basic cleaning instead.")
        return basic_clean_dataframe(df)
    
    if not openai_api_key:
        st.error("No OpenAI API key provided. Using basic cleaning instead.")
        return basic_clean_dataframe(df)
    
    try:
        # Configure OpenAI
        client = openai.OpenAI(api_key=openai_api_key)
        
        # Create a sample of the data for analysis
        sample_size = min(10, len(df))
        sample = df.head(sample_size)
        
        # Generate prompt for OpenAI
        prompt = f"""
        You are a data cleaning expert. Clean this dataset and return the cleaning instructions.
        
        Dataset Summary:
        - Total rows: {len(df)}
        - Total columns: {len(df.columns)}
        - Column names: {', '.join(df.columns)}
        
        Sample data:
        {sample.to_string()}
        
        Please provide specific Python code to clean this dataset, addressing:
        1. Missing values
        2. Duplicate rows
        3. Data type conversions
        4. Outlier detection and handling
        5. Text standardization
        6. Any other relevant cleaning steps
        
        Return ONLY the Python code that performs the cleaning.
        """
        
        # Get cleaning instructions from OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data cleaning expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        # Extract cleaning code
        cleaning_code = response.choices[0].message.content.strip()
        
        # Create a cleaning report
        cleaning_report = {
            "operations": [{"operation": "ai_cleaning", "explanation": "Used AI to clean the dataset"}],
            "code": cleaning_code,
            "rows_before": len(df),
            "columns_before": len(df.columns),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Execute the cleaning code
        try:
            # Create a safe local environment
            local_env = {
                'pd': pd,
                'np': np,
                'df': df.copy(),
                'cleaned_df': None
            }
            
            # Execute the cleaning code
            exec(cleaning_code, local_env)
            
            # Get the cleaned dataframe
            if 'cleaned_df' in local_env and isinstance(local_env['cleaned_df'], pd.DataFrame):
                df_cleaned = local_env['cleaned_df']
            else:
                df_cleaned = local_env['df']  # Fallback to the modified input dataframe
            
            # Update the report
            cleaning_report["rows_after"] = len(df_cleaned)
            cleaning_report["columns_after"] = len(df_cleaned.columns)
            cleaning_report["rows_removed"] = cleaning_report["rows_before"] - cleaning_report["rows_after"]
            
            return df_cleaned, cleaning_report
            
        except Exception as e:
            st.error(f"Error executing AI cleaning code: {str(e)}")
            st.warning("Falling back to basic cleaning...")
            return basic_clean_dataframe(df)
            
    except Exception as e:
        st.error(f"Error with OpenAI processing: {str(e)}")
        st.warning("Falling back to basic cleaning...")
        return basic_clean_dataframe(df)

def clean_data_basic(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform basic cleaning operations on a dataframe without using AI.
    
    Args:
        df (pd.DataFrame): Input dataframe to clean
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned dataframe and cleaning report
    """
    df_cleaned = df.copy()
    cleaning_report = {
        "operations": [],
        "column_changes": {},
        "rows_before": len(df),
        "columns_before": len(df.columns),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 1. Remove duplicate rows
    duplicates = df_cleaned.duplicated()
    if duplicates.sum() > 0:
        df_cleaned = df_cleaned.drop_duplicates()
        cleaning_report["operations"].append({
            "operation": "remove_duplicates",
            "rows_affected": int(duplicates.sum()),
            "explanation": "Removed duplicate rows to ensure data integrity"
        })
    
    # 2. Handle missing values
    for col in df_cleaned.columns:
        missing = df_cleaned[col].isna().sum()
        if missing > 0:
            # For numeric columns, fill with median
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                median_val = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(median_val)
                cleaning_report["operations"].append({
                    "operation": "fill_missing",
                    "column": col,
                    "method": "median",
                    "value": float(median_val),
                    "rows_affected": int(missing),
                    "explanation": f"Filled {missing} missing values with the median ({median_val:.2f})"
                })
            # For categorical/text columns, fill with mode
            else:
                mode_val = df_cleaned[col].mode()[0]
                df_cleaned[col] = df_cleaned[col].fillna(mode_val)
                cleaning_report["operations"].append({
                    "operation": "fill_missing",
                    "column": col,
                    "method": "mode",
                    "value": str(mode_val),
                    "rows_affected": int(missing),
                    "explanation": f"Filled {missing} missing values with the most common value '{mode_val}'"
                })
    
    # 3. Standardize column names
    old_columns = df_cleaned.columns.tolist()
    new_columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col).lower() for col in old_columns]
    df_cleaned.columns = new_columns
    
    for old, new in zip(old_columns, new_columns):
        if old != new:
            cleaning_report["column_changes"][old] = new
            cleaning_report["operations"].append({
                "operation": "rename_column",
                "column": old,
                "new_name": new,
                "explanation": f"Standardized column name '{old}' to '{new}'"
            })
    
    # 4. Convert string columns with numeric content to numeric
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            # Check if column contains numeric values
            try:
                numeric_col = pd.to_numeric(df_cleaned[col], errors='coerce')
                # If more than 80% of values can be converted to numeric, do the conversion
                if numeric_col.notna().sum() / len(numeric_col) > 0.8:
                    df_cleaned[col] = numeric_col
                    cleaning_report["operations"].append({
                        "operation": "convert_to_numeric",
                        "column": col,
                        "rows_affected": int(numeric_col.notna().sum()),
                        "explanation": f"Converted string values to numeric type"
                    })
            except:
                pass
    
    # 5. Handle outliers in numeric columns using IQR method
    for col in df_cleaned.select_dtypes(include=[np.number]).columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
        if outliers > 0:
            df_cleaned[col] = df_cleaned[col].clip(lower_bound, upper_bound)
            cleaning_report["operations"].append({
                "operation": "handle_outliers",
                "column": col,
                "method": "IQR",
                "rows_affected": int(outliers),
                "explanation": f"Capped {outliers} outliers in column '{col}' using IQR method"
            })
    
    # Update final stats
    cleaning_report["rows_after"] = len(df_cleaned)
    cleaning_report["columns_after"] = len(df_cleaned.columns)
    cleaning_report["rows_removed"] = cleaning_report["rows_before"] - cleaning_report["rows_after"]
    
    return df_cleaned, cleaning_report

def clean_data_ai(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean data using AI-powered methods. Currently falls back to basic cleaning.
    
    Args:
        df (pd.DataFrame): Input dataframe to clean
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned dataframe and cleaning report
    """
    # For now, we'll use basic cleaning as AI cleaning is not yet implemented
    st.info("AI-powered cleaning is not yet available. Using basic cleaning instead.")
    return clean_data_basic(df) 