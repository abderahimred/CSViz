import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import math
import warnings
import re
import time
warnings.filterwarnings('ignore')
from visualization_decision import (
    get_vis_type_for_single_column, 
    get_vis_type_for_pair, 
    get_vis_type_for_triple,
    get_vis_type_for_groupby,
    get_vis_type_for_groupby_pair
)
import os

def calculate_column_score(df, column_name):
    """
    Calculate a score for an individual column based on multiple criteria.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame with clean data.
    column_name : str
        The name of the column to score
        
    Returns:
    --------
    dict
        A dictionary containing component scores and the total score
    """
    scores = {
        'distribution_score': 0,
        'data_type_score': 0,
        'data_quality_score': 0,
        'predictive_power_score': 0,
        'semantic_content_score': 0,
        'dimensional_analysis_score': 0,
        'variance_info_ratio_score': 0,
        'total_score': 0
    }
    
    # Get column data and determine data type
    col_data = df[column_name]
    
    # Check if column is numeric
    is_numeric = pd.api.types.is_numeric_dtype(col_data)
    # Check if column is datetime
    is_temporal = pd.api.types.is_datetime64_any_dtype(col_data)
    # Check if column is categorical (either explicitly or by having few unique values)
    is_categorical = False
    if not is_numeric and not is_temporal:
        is_categorical = True
    elif is_numeric and col_data.nunique() <= 20:
        is_categorical = True
    
    # 1. Distribution Characteristics Score (0-10)
    if is_numeric and not is_categorical:
        # Calculate coefficient of variation (CV)
        if abs(col_data.mean()) > 0:
            cv = col_data.std() / abs(col_data.mean())
            cv_score = min(10, cv * 5)
        else:
            cv_score = 5  # Default if mean is zero
        
        # Calculate skewness score
        try:
            skewness = col_data.skew()
            skewness_score = min(10, abs(skewness) * 2)
        except:
            skewness_score = 0
        
        # Calculate kurtosis score
        try:
            kurtosis = col_data.kurtosis()
            kurtosis_score = min(10, abs(kurtosis - 3) * 1.5)
        except:
            kurtosis_score = 0
        
        # NEW: Check for multimodality using KDE
        try:
            from scipy.signal import find_peaks
            # Create KDE and find peaks
            kde = stats.gaussian_kde(col_data.dropna())
            x = np.linspace(col_data.min(), col_data.max(), 1000)
            y = kde(x)
            peaks, _ = find_peaks(y, height=0.1*max(y), distance=50)
            
            # Score based on number of modes (peaks)
            multimodality_score = min(10, len(peaks) * 5)
        except:
            multimodality_score = 0
            
        # Average the scores
        scores['distribution_score'] = (cv_score + skewness_score + kurtosis_score + multimodality_score) / 4
        
    elif is_categorical or (is_numeric and is_categorical):
        # Calculate entropy for categorical columns
        value_counts = col_data.value_counts(normalize=True)
        if len(value_counts) > 1:
            entropy = -sum(p * np.log2(p) for p in value_counts if p > 0)
            max_entropy = np.log2(len(value_counts))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            scores['distribution_score'] = normalized_entropy * 10
        else:
            scores['distribution_score'] = 0  # Single value, no entropy
            
    elif is_temporal:
        # For temporal columns, check the range and frequency
        try:
            date_range = (col_data.max() - col_data.min()).total_seconds()
            # Score based on range - higher range gets higher score
            range_score = min(10, np.log10(1 + date_range / 3600) / 2)  # Log scale for date range in hours
            
            # Detect periodicity or patterns
            # This is a simplified approach - just checking value distribution
            time_parts_score = 0
            if hasattr(col_data.dt, 'hour') and col_data.dt.hour.nunique() > 1:
                time_parts_score += 2
            if hasattr(col_data.dt, 'dayofweek') and col_data.dt.dayofweek.nunique() > 1:
                time_parts_score += 2
            if hasattr(col_data.dt, 'month') and col_data.dt.month.nunique() > 1:
                time_parts_score += 2
            if hasattr(col_data.dt, 'year') and col_data.dt.year.nunique() > 1:
                time_parts_score += 4
                
            scores['distribution_score'] = (range_score + time_parts_score) / 2
        except:
            scores['distribution_score'] = 5  # Default score if calculation fails
    
    # 2. Data Type Score (0-10)
    if is_numeric and not is_categorical:
        # Base score for continuous numerical
        base_score = 8
        
        # Check range width
        try:
            col_min = col_data.min()
            col_max = col_data.max()
            if col_min > 0 and col_max / col_min > 1000:  # Spans 3 orders of magnitude
                base_score += 2
            elif col_min < 0 and col_max > 0 and (col_max - col_min) > 1000:
                base_score += 2
        except:
            pass
            
        scores['data_type_score'] = base_score
        
    elif is_categorical:
        # Score based on cardinality
        cardinality = col_data.nunique()
        if cardinality > 0:
            # This function peaks at around 10 categories
            cardinality_score = 10 * (1 - abs(np.log10(cardinality/10)/2))
            scores['data_type_score'] = max(0, min(10, cardinality_score))
        else:
            scores['data_type_score'] = 0
            
    elif is_temporal:
        # Base score for temporal
        base_score = 8
        
        # Check if data spans multiple time cycles
        try:
            if hasattr(col_data.dt, 'year') and col_data.dt.year.nunique() > 1:
                base_score += 1
            if hasattr(col_data.dt, 'month') and col_data.dt.month.nunique() > 1:
                base_score += 0.5
            if hasattr(col_data.dt, 'day') and col_data.dt.day.nunique() > 1:
                base_score += 0.5
        except:
            pass
            
        scores['data_type_score'] = base_score
    
    # 3. Data Quality Score (0-10)
    # Calculate completeness score
    missing_ratio = col_data.isna().mean()
    completeness_score = (1 - missing_ratio) * 10
    
    # Calculate outlier score for numerical columns
    outlier_score = 10
    if is_numeric and not is_categorical:
        try:
            # Use IQR method to identify outliers
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_ratio = ((col_data < lower_bound) | (col_data > upper_bound)).mean()
            outlier_score = (1 - outlier_ratio) * 10
        except:
            outlier_score = 5  # Default if calculation fails
    
    # Calculate unique value ratio
    unique_ratio = col_data.nunique() / len(col_data) if len(col_data) > 0 else 0
    if is_numeric and not is_categorical:
        unique_score = min(10, unique_ratio * 20)
    else:
        unique_score = min(10, unique_ratio * 100)
    
    # Average the scores
    scores['data_quality_score'] = (completeness_score + outlier_score + unique_score) / 3
    
    # 4. Predictive Power Score (0-10)
    # Since we don't have a defined target, we'll score based on correlation with other columns
    max_correlation = 0
    if is_numeric and not is_categorical:
        # Get correlations with other numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            # Exclude self-correlation
            col_corrs = corr_matrix[column_name].drop(column_name, errors='ignore')
            if not col_corrs.empty:
                max_correlation = col_corrs.max()
    
    # For categorical columns, compute Cramer's V with other categorical columns
    elif is_categorical:
        max_assoc = 0
        cat_cols = [c for c in df.columns if c != column_name and 
                    (not pd.api.types.is_numeric_dtype(df[c]) or 
                     df[c].nunique() <= 20)]
        
        for other_col in cat_cols:
            try:
                # Create contingency table
                crosstab = pd.crosstab(df[column_name], df[other_col])
                # Calculate Chi-square and Cramer's V
                chi2, _, _, _ = stats.chi2_contingency(crosstab)
                n = crosstab.sum().sum()
                min_dim = min(crosstab.shape) - 1
                cramers_v = np.sqrt(chi2 / (n * min_dim)) if n * min_dim > 0 else 0
                max_assoc = max(max_assoc, cramers_v)
            except:
                pass
        
        max_correlation = max_assoc
    
    scores['predictive_power_score'] = min(10, max_correlation * 10)
    
    # 5. Semantic Content Analysis (0-10) [NEW]
    # Analyze the column name and content to identify its purpose
    
    # Start with base score of 10 as per documentation
    semantic_score = 10
    
    # Check for ID-like columns
    id_pattern = r'(^|\s|_)(id|key|code|uuid|identifier|seq|sequence|num)($|\s|_)'
    if re.search(id_pattern, column_name.lower()):
        # Penalize ID columns more heavily (-5 points per documentation)
        semantic_score -= 5
    
    # Check if values are sequential integers
    if is_numeric and len(col_data) > 5:
        try:
            # Check for sequential integers or near-sequential
            sorted_vals = sorted(col_data.dropna().unique())
            if len(sorted_vals) >= 3:
                # Check if values are mostly sequential
                diffs = [sorted_vals[i+1] - sorted_vals[i] for i in range(len(sorted_vals)-1)]
                avg_diff = sum(diffs) / len(diffs)
                if 0.9 <= avg_diff <= 1.1 and min(diffs) >= 0:  # Allow small gaps but ensure increasing
                    semantic_score -= 5  # Penalize sequential integers
        except:
            pass
            
    # If cardinality equals or nearly equals row count (unique identifier)
    if col_data.nunique() > 0.8 * len(col_data) and len(col_data) > 10:
        semantic_score -= 5
        
    # Check for internal reference columns (high cardinality with non-meaningful values)
    if is_categorical and col_data.nunique() > 100:
        sample_vals = col_data.dropna().sample(min(10, len(col_data))).astype(str)
        non_word_chars = sum(1 for val in sample_vals for c in val if not c.isalnum())
        avg_non_word = non_word_chars / len(sample_vals) if len(sample_vals) > 0 else 0
        
        if avg_non_word > 3:  # Many non-word characters suggests technical/reference fields
            semantic_score -= 5
    
    # Identify and reward metrics columns
    metrics_pattern = r'(^|\s|_)(total|sum|count|amount|rate|ratio|percent|score|avg|average|mean|max|min|std|var)($|\s|_)'
    if re.search(metrics_pattern, column_name.lower()) and is_numeric and not is_categorical:
        semantic_score += 2
        
        # If column contains mostly non-zero positive numbers
        if is_numeric and (col_data > 0).mean() > 0.8:
            semantic_score += 2
    
    # Identify and reward descriptor columns
    desc_pattern = r'(^|\s|_)(name|description|title|label|category|type|group|reason|status)($|\s|_)'
    if re.search(desc_pattern, column_name.lower()) and is_categorical:
        semantic_score += 2
        
        # Additional reward for meaningful text values
        if not is_numeric:
            sample_vals = col_data.dropna().sample(min(10, len(col_data))).astype(str)
            avg_len = sum(len(str(v)) for v in sample_vals) / len(sample_vals) if len(sample_vals) > 0 else 0
            
            if 3 < avg_len < 50:  # Reasonable length for readable descriptions
                semantic_score += 1
                
    # Cap the score
    scores['semantic_content_score'] = max(0, min(10, semantic_score))
    
    # 6. Dimensional Analysis (0-10) [NEW]
    # Evaluate column's role in dimensional modeling
    
    # Start with base score based on documentation
    if is_numeric and not is_categorical:
        dimensional_score = 8  # Base for fact/measure columns
    elif is_categorical:
        dimensional_score = 7  # Base for dimension columns
    elif is_numeric and col_data.nunique() == len(col_data) and len(col_data) > 10:
        dimensional_score = 2  # Base for surrogate keys
    else:
        dimensional_score = 5  # Default
    
    # Identify fact/measure columns
    if is_numeric and not is_categorical:
        # Check if has metric-related name
        metric_pattern = r'(^|\s|_)(amount|total|sum|count|qty|quantity|volume|sales|revenue|profit|loss|cost|price|value)($|\s|_)'
        if re.search(metric_pattern, column_name.lower()):
            dimensional_score += 2
    
    # Identify dimension columns
    elif is_categorical:
        # Check for reasonable cardinality (5-20)
        if 5 <= col_data.nunique() <= 20:
            dimensional_score += 2
            
        # Check for dimension-like names
        dim_pattern = r'(^|\s|_)(region|country|state|city|location|category|dept|department|segment|customer|product|date|year|month|day)($|\s|_)'
        if re.search(dim_pattern, column_name.lower()):
            dimensional_score += 1  # Reward for natural hierarchy potential
    
    # Penalize surrogate keys and technical fields
    tech_pattern = r'(^|\s|_)(id$|key$|code$|uuid$|hash$|index$|created_at|updated_at|timestamp$)($|\s|_)'
    if re.search(tech_pattern, column_name.lower()):
        if is_numeric and col_data.nunique() > 0.7 * len(col_data):
            dimensional_score -= 2  # Higher penalty for high cardinality technical fields
        else:
            dimensional_score -= 1
    
    # Check for date dimensions
    if is_temporal:
        dimensional_score += 2  # Date dimensions are valuable
        
    # Cap the score
    scores['dimensional_analysis_score'] = max(0, min(10, dimensional_score))
    
    # 7. Variance Information Ratio (0-10) [NEW]
    # Calculate information density relative to variance
    
    if is_numeric and not is_categorical:
        # For numerical columns, calculate coefficient of variation but penalize artificially high variance
        try:
            # Calculate basic variance statistics
            std = col_data.std()
            mean_abs = abs(col_data.mean())
            
            if mean_abs > 0:
                cv = std / mean_abs
                
                # Check if outliers are driving the variance
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                
                # Calculate trimmed variance (removing outliers)
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                trimmed = col_data[(col_data >= lower_bound) & (col_data <= upper_bound)]
                
                if len(trimmed) > 0:
                    trimmed_cv = trimmed.std() / abs(trimmed.mean()) if abs(trimmed.mean()) > 0 else 0
                    
                    # If trimmed CV is much lower, the variance is driven by outliers
                    outlier_variance_ratio = trimmed_cv / cv if cv > 0 else 1
                    
                    # Score based on information density
                    variance_info_score = min(10, outlier_variance_ratio * 10)
                else:
                    variance_info_score = 5
            else:
                variance_info_score = 5
        except:
            variance_info_score = 5
    
    elif is_categorical:
        # For categorical, calculate information-to-category ratio
        try:
            n_categories = col_data.nunique()
            
            if n_categories > 1:
                # Calculate entropy
                value_counts = col_data.value_counts(normalize=True)
                entropy = -sum(p * np.log2(p) for p in value_counts if p > 0)
                
                # Calculate maximum possible entropy for this number of categories
                max_entropy = np.log2(n_categories)
                
                # Information density is ratio of actual entropy to max entropy
                information_density = entropy / max_entropy if max_entropy > 0 else 0
                
                # Score higher for efficient encoding (high entropy relative to # of categories)
                variance_info_score = information_density * 10
            else:
                variance_info_score = 0  # Single category has no information
        except:
            variance_info_score = 5
    
    elif is_temporal:
        # For temporal data, look at the distribution across time units
        try:
            # Check distribution across time components
            if hasattr(col_data.dt, 'dayofweek') and col_data.dt.dayofweek.nunique() > 1:
                dow_counts = col_data.dt.dayofweek.value_counts(normalize=True)
                dow_entropy = -sum(p * np.log2(p) for p in dow_counts if p > 0)
                dow_max_entropy = np.log2(7)  # 7 days in a week
                dow_info_density = dow_entropy / dow_max_entropy if dow_max_entropy > 0 else 0
                
                # Score based on even distribution across days (higher entropy = better)
                variance_info_score = dow_info_density * 10
            else:
                variance_info_score = 5
        except:
            variance_info_score = 5
    else:
        variance_info_score = 5
    
    scores['variance_info_ratio_score'] = max(0, min(10, variance_info_score))
    
    # Calculate total score using weighted formula from documentation
    scores['total_score'] = (
        0.20 * scores['distribution_score'] + 
        0.15 * scores['data_type_score'] + 
        0.10 * scores['data_quality_score'] + 
        0.15 * scores['predictive_power_score'] +
        0.20 * scores['semantic_content_score'] +
        0.10 * scores['dimensional_analysis_score'] +
        0.10 * scores['variance_info_ratio_score']
    )
    
    return scores

def calculate_pair_score(df, col1, col2):
    """
    Calculate a score for a pair of columns based on multiple criteria.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame with clean data.
    col1 : str
        The name of the first column
    col2 : str
        The name of the second column
        
    Returns:
    --------
    dict
        A dictionary containing component scores and the total score
    """
    scores = {
        'statistical_association': 0,
        'visualization_complexity': 0,
        'pattern_detection': 0,
        'anomaly_highlighting': 0,
        'information_complementarity': 0,
        'redundancy_penalization': 0,
        'practical_utility_score': 0,  # NEW: score for practical visualization utility
        'total_score': 0
    }
    
    # Get column data and determine data types
    data1 = df[col1]
    data2 = df[col2]
    
    # Check data types
    is_numeric1 = pd.api.types.is_numeric_dtype(data1)
    is_numeric2 = pd.api.types.is_numeric_dtype(data2)
    is_temporal1 = pd.api.types.is_datetime64_any_dtype(data1)
    is_temporal2 = pd.api.types.is_datetime64_any_dtype(data2)
    
    # Define categorical as either explicitly categorical or numeric with few unique values
    is_categorical1 = not is_numeric1 or (is_numeric1 and data1.nunique() <= 20)
    is_categorical2 = not is_numeric2 or (is_numeric2 and data2.nunique() <= 20)
    
    # NEW: Enhanced validation flags for columns with limited visualization value
    is_low_value_col1 = False
    is_low_value_col2 = False
    
    # Check for ID columns
    id_pattern = r'(^|\s|_)(id|key|code|uuid|identifier|seq|sequence|num)($|\s|_)'
    is_id_col1 = False
    is_id_col2 = False
    
    # Check column names for ID patterns
    if re.search(id_pattern, col1.lower()):
        is_id_col1 = True
        is_low_value_col1 = True
    
    if re.search(id_pattern, col2.lower()):
        is_id_col2 = True
        is_low_value_col2 = True
    
    # NEW: Check for timestamp/login columns with limited visualization value
    timestamp_pattern = r'(^|\s|_)(timestamp|login|last_login|logged_in|created_at|updated_at|date_added|modified)($|\s|_)'
    
    if re.search(timestamp_pattern, col1.lower()) or (is_temporal1 and 'login' in col1.lower()):
        is_low_value_col1 = True
        
    if re.search(timestamp_pattern, col2.lower()) or (is_temporal2 and 'login' in col2.lower()):
        is_low_value_col2 = True
        
    # NEW: Check for timestamp distributions
    # Timestamps that are evenly distributed or clustered at a few points have low visualization value
    if is_temporal1:
        # Check time distribution pattern
        try:
            if hasattr(data1.dt, 'hour'):
                hour_counts = data1.dt.hour.value_counts()
                # If timestamps are all at same hour (like midnight), lower value
                if hour_counts.iloc[0] > 0.8 * len(data1):
                    is_low_value_col1 = True
                    
            # Check if most dates are the same or very recent (last day logins)
            if hasattr(data1.dt, 'date'):
                # If most timestamps are from a single day
                day_counts = data1.dt.date.value_counts()
                if day_counts.iloc[0] > 0.7 * len(data1):
                    is_low_value_col1 = True
        except:
            pass
            
    if is_temporal2:
        # Check time distribution pattern
        try:
            if hasattr(data2.dt, 'hour'):
                hour_counts = data2.dt.hour.value_counts()
                # If timestamps are all at same hour (like midnight), lower value
                if hour_counts.iloc[0] > 0.8 * len(data2):
                    is_low_value_col2 = True
                    
            # Check if most dates are the same or very recent (last day logins)
            if hasattr(data2.dt, 'date'):
                # If most timestamps are from a single day
                day_counts = data2.dt.date.value_counts()
                if day_counts.iloc[0] > 0.7 * len(data2):
                    is_low_value_col2 = True
        except:
            pass
    
    # Check for sequential or high-cardinality numeric columns
    if is_numeric1 and data1.nunique() > 0.8 * len(data1) and len(data1) > 10:
        # Check if it's sequential
        try:
            sorted_vals = sorted(data1.dropna().unique())
            if len(sorted_vals) >= 3:
                diffs = [sorted_vals[i+1] - sorted_vals[i] for i in range(len(sorted_vals)-1)]
                avg_diff = sum(diffs) / len(diffs)
                if 0.9 <= avg_diff <= 1.1 and min(diffs) >= 0:
                    is_id_col1 = True
                    is_low_value_col1 = True
        except:
            pass
    
    if is_numeric2 and data2.nunique() > 0.8 * len(data2) and len(data2) > 10:
        # Check if it's sequential
        try:
            sorted_vals = sorted(data2.dropna().unique())
            if len(sorted_vals) >= 3:
                diffs = [sorted_vals[i+1] - sorted_vals[i] for i in range(len(sorted_vals)-1)]
                avg_diff = sum(diffs) / len(diffs)
                if 0.9 <= avg_diff <= 1.1 and min(diffs) >= 0:
                    is_id_col2 = True
                    is_low_value_col2 = True
        except:
            pass
    
    # For numeric analysis, convert categoricals to numeric if needed
    if is_categorical1 and not is_numeric1:
        # Create a copy to avoid warnings about setting values on a slice
        data1_numeric = pd.Series(LabelEncoder().fit_transform(data1.fillna('missing')))
    else:
        data1_numeric = data1
        
    if is_categorical2 and not is_numeric2:
        # Create a copy to avoid warnings about setting values on a slice
        data2_numeric = pd.Series(LabelEncoder().fit_transform(data2.fillna('missing')))
    else:
        data2_numeric = data2
    
    # 1. Statistical Association Score (0-10)
    # Numeric vs Numeric
    if is_numeric1 and is_numeric2 and not is_categorical1 and not is_categorical2:
        try:
            # Calculate Pearson correlation
            pearson_corr = data1.corr(data2)
            corr_score = abs(pearson_corr) * 10
            
            # Calculate mutual information for non-linearity check
            # Normalize data for MI calculation
            x = (data1 - data1.min()) / (data1.max() - data1.min()) if data1.max() > data1.min() else data1
            y = (data2 - data2.min()) / (data2.max() - data2.min()) if data2.max() > data2.min() else data2
            
            try:
                mi = mutual_info_regression(x.values.reshape(-1, 1), y)[0]
                mi_normalized = mi / np.sqrt(stats.entropy(x.value_counts(normalize=True))) if stats.entropy(x.value_counts(normalize=True)) > 0 else 0
                
                # If MI suggests stronger relationship than correlation, boost the score
                if mi_normalized > abs(pearson_corr):
                    corr_score += min(2, (mi_normalized - abs(pearson_corr)) * 5)
            except:
                pass
                
            scores['statistical_association'] = min(10, corr_score)
        except:
            scores['statistical_association'] = 0
            
    # Categorical vs Categorical
    elif is_categorical1 and is_categorical2:
        try:
            # Create contingency table
            crosstab = pd.crosstab(data1, data2)
            
            # Calculate Chi-square and Cramer's V
            chi2, p, _, _ = stats.chi2_contingency(crosstab)
            n = crosstab.sum().sum()
            min_dim = min(crosstab.shape) - 1
            if n * min_dim > 0:
                cramers_v = np.sqrt(chi2 / (n * min_dim))
                cramers_v_score = cramers_v * 10
            else:
                cramers_v_score = 0
                
            # Calculate normalized mutual information
            try:
                # Use the numeric versions
                mi = mutual_info_classif(data1_numeric.values.reshape(-1, 1), data2_numeric)[0]
                
                # Normalize by entropy
                h1 = stats.entropy(data1.value_counts(normalize=True))
                h2 = stats.entropy(data2.value_counts(normalize=True))
                if min(h1, h2) > 0:
                    nmi = mi / np.sqrt(h1 * h2)
                    nmi_score = nmi * 10
                else:
                    nmi_score = 0
            except:
                nmi_score = 0
                
            # Take the maximum of the two scores
            scores['statistical_association'] = max(cramers_v_score, nmi_score)
        except:
            scores['statistical_association'] = 0
            
    # Numeric vs Categorical
    elif (is_numeric1 and not is_categorical1 and is_categorical2) or \
         (is_numeric2 and not is_categorical2 and is_categorical1):
        
        # Ensure numeric is first, categorical is second
        if is_categorical1:
            num_data = data2
            cat_data = data1
            num_is_id = is_id_col2
            cat_is_id = is_id_col1
        else:
            num_data = data1
            cat_data = data2
            num_is_id = is_id_col1
            cat_is_id = is_id_col2
            
        try:
            # Calculate ANOVA-derived Eta squared
            categories = cat_data.unique()
            if len(categories) > 1:
                # Create groups based on categories
                groups = [num_data[cat_data == cat].values for cat in categories if sum(cat_data == cat) > 0]
                if len(groups) > 1 and all(len(g) > 0 for g in groups):
                    f_val, p_val = stats.f_oneway(*groups)
                    # Calculate Eta squared
                    total_var = num_data.var() * len(num_data)
                    between_var = f_val * (len(categories) - 1)
                    eta_squared = between_var / (between_var + (len(num_data) - len(categories)))
                    eta_squared_score = eta_squared * 10
                    
                    # If binary categorical, also calculate point-biserial correlation
                    if len(categories) == 2:
                        try:
                            # Convert to binary
                            binary_cat = (cat_data == categories[0]).astype(int)
                            r_pb, p_val = stats.pointbiserialr(binary_cat, num_data)
                            rpb_score = abs(r_pb) * 10
                            scores['statistical_association'] = max(eta_squared_score, rpb_score)
                        except:
                            scores['statistical_association'] = eta_squared_score
                    else:
                        scores['statistical_association'] = eta_squared_score
                else:
                    scores['statistical_association'] = 0
            else:
                scores['statistical_association'] = 0
        except:
            scores['statistical_association'] = 0
    
    # Time series analysis
    elif (is_temporal1 and is_numeric2 and not is_categorical2) or \
         (is_temporal2 and is_numeric1 and not is_categorical1):
        
        # Ensure temporal is first, numeric is second
        if is_temporal2:
            temp_data = data2
            num_data = data1
        else:
            temp_data = data1
            num_data = data2
            
        try:
            # Sort by time
            combined = pd.DataFrame({'time': temp_data, 'value': num_data})
            combined = combined.sort_values('time')
            
            # Check for trend using correlation between time index and value
            time_idx = np.arange(len(combined))
            trend_corr = np.corrcoef(time_idx, combined['value'])[0, 1]
            trend_score = abs(trend_corr) * 10
            scores['statistical_association'] = trend_score
        except:
            scores['statistical_association'] = 0
    
    # 2. Visualization Complexity Score (0-10)
    # Numeric vs Numeric
    if is_numeric1 and is_numeric2 and not is_categorical1 and not is_categorical2:
        try:
            # Create a 2D histogram to check for empty quadrants
            hist2d, _, _ = np.histogram2d(data1, data2, bins=10)
            total_quadrants = hist2d.size
            empty_quadrants = np.sum(hist2d == 0)
            
            # Calculate sparsity penalty
            sparsity_penalty = empty_quadrants / total_quadrants
            scores['visualization_complexity'] = 10 - min(9, sparsity_penalty * 10)
        except:
            scores['visualization_complexity'] = 5  # Default if calculation fails
            
    # Categorical vs Categorical
    elif is_categorical1 and is_categorical2:
        try:
            # Create contingency table
            crosstab = pd.crosstab(data1, data2)
            
            # Calculate table sparsity
            total_cells = crosstab.size
            empty_cells = np.sum(crosstab == 0)
            
            if total_cells > 0:
                sparsity = empty_cells / total_cells
                scores['visualization_complexity'] = 10 - min(9, sparsity * 10)
            else:
                scores['visualization_complexity'] = 5
        except:
            scores['visualization_complexity'] = 5
            
    # Numeric vs Categorical
    elif (is_numeric1 and not is_categorical1 and is_categorical2) or \
         (is_numeric2 and not is_categorical2 and is_categorical1):
        
        # Ensure numeric is first, categorical is second
        if is_categorical1:
            num_data = data2
            cat_data = data1
        else:
            num_data = data1
            cat_data = data2
            
        try:
            # Calculate separability score
            categories = cat_data.unique()
            if len(categories) > 1:
                # Calculate between-group and total variance
                group_means = [num_data[cat_data == cat].mean() for cat in categories if sum(cat_data == cat) > 0]
                overall_mean = num_data.mean()
                
                if len(group_means) > 1:
                    # Calculate weighted between-group variance
                    between_var = sum((grp_mean - overall_mean) ** 2 * sum(cat_data == cat) 
                                     for grp_mean, cat in zip(group_means, categories) 
                                     if sum(cat_data == cat) > 0)
                    
                    # Total variance
                    total_var = sum((x - overall_mean) ** 2 for x in num_data)
                    
                    if total_var > 0:
                        separability = between_var / total_var
                        scores['visualization_complexity'] = separability * 10
                    else:
                        scores['visualization_complexity'] = 5
                else:
                    scores['visualization_complexity'] = 5
            else:
                scores['visualization_complexity'] = 5
        except:
            scores['visualization_complexity'] = 5
    
    # 3. Pattern Detection Score (0-10)
    # Numeric vs Numeric - Cluster detection
    if is_numeric1 and is_numeric2 and not is_categorical1 and not is_categorical2:
        try:
            # Prepare data for clustering
            X = np.column_stack([data1, data2])
            
            # Normalize data
            X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
            
            # Use DBSCAN for cluster detection
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(X_norm)
            
            # Count number of clusters (excluding noise points labeled as -1)
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            
            # Calculate ratio of points in clusters vs noise
            if len(clusters) > 0:
                non_noise_ratio = sum(clusters != -1) / len(clusters)
                
                # Score based on clusters found and non-noise ratio
                if n_clusters > 0:
                    cluster_score = min(10, (n_clusters * 2 + non_noise_ratio * 8))
                else:
                    cluster_score = non_noise_ratio * 5  # Some points are clustered but no distinct clusters
            else:
                cluster_score = 0
                
            scores['pattern_detection'] = cluster_score
        except:
            scores['pattern_detection'] = 0
            
    # Time series analysis
    elif (is_temporal1 and is_numeric2 and not is_categorical2) or \
         (is_temporal2 and is_numeric1 and not is_categorical1):
        
        # Ensure temporal is first, numeric is second
        if is_temporal2:
            temp_data = data2
            num_data = data1
        else:
            temp_data = data1
            num_data = data2
            
        try:
            # Sort by time
            combined = pd.DataFrame({'time': temp_data, 'value': num_data})
            combined = combined.sort_values('time')
            
            # Check for trend
            trend_strength = 0
            try:
                # Simple linear regression for trend detection
                x = np.arange(len(combined))
                y = combined['value'].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                trend_strength = abs(r_value) * 5  # Scale to 0-5
            except:
                pass
                
            # Check for seasonality (simplified)
            seasonality_score = 0
            try:
                # Use autocorrelation as a proxy for seasonality
                autocorr = combined['value'].autocorr(lag=1)
                if not np.isnan(autocorr):
                    seasonality_score = abs(autocorr) * 5  # Scale to 0-5
            except:
                pass
                
            scores['pattern_detection'] = trend_strength + seasonality_score
        except:
            scores['pattern_detection'] = 0
    
    # 4. Anomaly Highlighting Score (0-10)
    # For all numeric pairs, use isolation forest to detect outliers
    if (is_numeric1 or is_numeric2) and not (is_categorical1 and is_categorical2):
        try:
            # Prepare data based on column types
            if is_numeric1 and is_numeric2 and not is_categorical1 and not is_categorical2:
                # Two numeric columns
                X = np.column_stack([data1, data2])
            elif is_numeric1 and not is_categorical1:
                # Only first column is usable numeric
                X = data1.values.reshape(-1, 1)
            elif is_numeric2 and not is_categorical2:
                # Only second column is usable numeric
                X = data2.values.reshape(-1, 1)
            else:
                # No usable numeric columns
                scores['anomaly_highlighting'] = 0
                X = None
                
            if X is not None:
                # Use isolation forest for outlier detection
                iso_forest = IsolationForest(random_state=42, contamination=0.1)
                outliers = iso_forest.fit_predict(X)
                
                # Calculate ratio of outliers
                outlier_ratio = sum(outliers == -1) / len(outliers)
                
                # Score based on outlier ratio (penalize if too many or too few)
                ideal_ratio = 0.05  # Ideal outlier ratio around 5%
                outlier_score = 10 * (1 - min(1, abs(outlier_ratio - ideal_ratio) / 0.1))
                
                scores['anomaly_highlighting'] = outlier_score
        except:
            scores['anomaly_highlighting'] = 0
            
    # For categorical pairs, check for unexpected conditional frequencies
    elif is_categorical1 and is_categorical2:
        try:
            # Create contingency table
            crosstab = pd.crosstab(data1, data2)
            
            # Calculate expected frequencies
            row_sums = crosstab.sum(axis=1)
            col_sums = crosstab.sum(axis=0)
            total = crosstab.sum().sum()
            
            expected = np.outer(row_sums, col_sums) / total
            
            # Calculate deviation from expected
            deviation = np.abs(crosstab.values - expected) / expected
            
            # Replace infinities and NaNs
            deviation = np.nan_to_num(deviation, nan=0, posinf=0, neginf=0)
            
            # Score based on average deviation
            if crosstab.size > 0:
                avg_deviation = np.sum(deviation) / crosstab.size
                scores['anomaly_highlighting'] = min(10, avg_deviation * 10)
            else:
                scores['anomaly_highlighting'] = 0
        except:
            scores['anomaly_highlighting'] = 0
    
    # 5. Information Complementarity (0-10) [NEW]
    # Measure how columns mutually enhance understanding
    
    # Initialize the score
    info_comp_score = 0
    
    # Calculate for different column type combinations
    if is_numeric1 and is_numeric2 and not is_categorical1 and not is_categorical2:
        try:
            # For two numeric columns, check if knowing one column improves prediction of the other
            
            # Calculate mutual information
            x = data1.values.reshape(-1, 1)
            y = data2
            
            # Normalize both columns for consistent MI calculation
            x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
            y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y
            
            mi = mutual_info_regression(x_norm, y_norm)[0]
            
            # Calculate entropies for normalization
            h_x = stats.entropy(np.histogram(x_norm.flatten(), bins=20)[0]/len(x_norm)) if len(x_norm) > 0 else 0
            h_y = stats.entropy(np.histogram(y_norm, bins=20)[0]/len(y_norm)) if len(y_norm) > 0 else 0
            
            # Normalized mutual information
            if min(h_x, h_y) > 0:
                nmi = mi / min(h_x, h_y)
                
                # Calculate conditional variance reduction
                # (How much variance in y is explained by x)
                r_squared = data1.corr(data2) ** 2
                
                # Average of NMI and R²
                info_comp_score = ((nmi * 10) + (r_squared * 10)) / 2
            else:
                info_comp_score = 0
                
        except:
            info_comp_score = 0
    
    elif is_categorical1 and is_categorical2:
        try:
            # For two categorical columns, use joint entropy reduction
            
            # Calculate individual entropies
            p1 = data1.value_counts(normalize=True)
            h1 = -sum(p * np.log2(p) for p in p1 if p > 0)
            
            p2 = data2.value_counts(normalize=True)
            h2 = -sum(p * np.log2(p) for p in p2 if p > 0)
            
            # Calculate joint entropy using crosstab
            crosstab = pd.crosstab(data1, data2, normalize=True)
            joint_probs = crosstab.values.flatten()
            h_joint = -sum(p * np.log2(p) for p in joint_probs if p > 0)
            
            # Calculate mutual information
            mi = h1 + h2 - h_joint
            
            # Normalize by the minimum entropy
            min_entropy = min(h1, h2)
            if min_entropy > 0:
                # Asymmetric information ratio
                info_comp_score = (mi / min_entropy) * 10
            else:
                info_comp_score = 0
                
        except:
            info_comp_score = 0
        
    elif (is_numeric1 and not is_categorical1 and is_categorical2) or \
         (is_numeric2 and not is_categorical2 and is_categorical1):
        
        # Ensure numeric is first, categorical is second
        if is_categorical1:
            num_data = data2
            cat_data = data1
        else:
            num_data = data1
            cat_data = data2
            
        try:
            # Calculate conditional variance reduction
            categories = cat_data.unique()
            
            if len(categories) > 1:
                # Overall variance
                total_variance = num_data.var()
                
                # Weighted average of within-group variances
                within_group_vars = []
                group_sizes = []
                
                for cat in categories:
                    group = num_data[cat_data == cat]
                    if len(group) > 1:  # Need at least 2 points to calculate variance
                        within_group_vars.append(group.var())
                        group_sizes.append(len(group))
                
                if len(within_group_vars) > 0 and sum(group_sizes) > 0:
                    avg_within_var = sum(v * s for v, s in zip(within_group_vars, group_sizes)) / sum(group_sizes)
                    
                    # Variance explained by the categorical variable
                    var_explained = 1 - (avg_within_var / total_variance) if total_variance > 0 else 0
                    
                    # Score based on variance explained
                    info_comp_score = var_explained * 10
                else:
                    info_comp_score = 0
            else:
                info_comp_score = 0
                
        except:
            info_comp_score = 0
        
    elif (is_temporal1 and is_numeric2) or (is_temporal2 and is_numeric1):
        # For temporal + numeric, look at how time patterns enhance understanding
        
        # Ensure temporal is first
        if is_temporal2:
            temp_data = data2
            num_data = data1
        else:
            temp_data = data1
            num_data = data2
            
        try:
            # Sort by time
            combined = pd.DataFrame({'time': temp_data, 'value': num_data})
            combined = combined.sort_values('time')
            
            # Calculate trend
            x = np.arange(len(combined))
            y = combined['value'].values
            
            # Calculate R² of linear trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value ** 2
            
            # See if grouping by time periods improves understanding
            if hasattr(temp_data.dt, 'month'):
                # Group by month and calculate variance reduction
                month_groups = combined.groupby(temp_data.dt.month)['value']
                
                if len(month_groups) > 1:
                    # Calculate total variance vs. within-month variance
                    total_var = combined['value'].var()
                    weighted_within_var = sum(g.var() * len(g) for _, g in month_groups) / len(combined)
                    
                    month_var_explained = 1 - (weighted_within_var / total_var) if total_var > 0 else 0
                    
                    # Take the better of trend R² or month grouping
                    info_comp_score = max(r_squared, month_var_explained) * 10
                else:
                    info_comp_score = r_squared * 10
            else:
                info_comp_score = r_squared * 10
                
        except:
            info_comp_score = 0

    # Heavily penalize information complementarity for ID columns
    if is_id_col1 or is_id_col2:
        # ID columns typically provide minimal complementary information
        info_comp_score = max(0, info_comp_score - 7)
    
    # Cap the score
    scores['information_complementarity'] = max(0, min(10, info_comp_score))
    
    # 6. Redundancy Penalization (0-10) [NEW]
    # Start with perfect score and reduce based on redundancy
    redundancy_score = 10
    
    # Special case for ID columns
    if is_id_col1 or is_id_col2:
        # ID columns are often redundant with anything else
        redundancy_score = 0  # Maximum penalty
    else:
        # Regular redundancy checks for non-ID columns
        # Check for different column type combinations
        if is_numeric1 and is_numeric2 and not is_categorical1 and not is_categorical2:
            try:
                # Calculate correlation for numeric columns
                correlation = abs(data1.corr(data2))
                
                # Penalize high correlation (close to 1)
                if correlation > 0.95:  # Almost perfect correlation
                    redundancy_score = 0  # Completely redundant
                elif correlation > 0.8:
                    redundancy_score = 3  # Highly redundant
                elif correlation > 0.6:
                    redundancy_score = 6  # Moderately redundant
                
                # Also check for functional relationships that might not have high correlation
                # (e.g., quadratic, exponential)
                try:
                    # Calculate R² for polynomial fits
                    x = data1.values
                    y = data2.values
                    
                    # Try polynomial fit (degree 2)
                    poly_model = np.polyfit(x, y, 2)
                    p = np.poly1d(poly_model)
                    y_pred = p(x)
                    
                    # Calculate R²
                    ss_total = np.sum((y - np.mean(y)) ** 2)
                    ss_residual = np.sum((y - y_pred) ** 2)
                    r_squared_poly = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
                    
                    # If polynomial fit is very good but correlation is not very high,
                    # there's a non-linear functional dependency
                    if r_squared_poly > 0.95 and correlation < 0.95:
                        redundancy_score = min(redundancy_score, 2)  # Strong non-linear dependency
                    elif r_squared_poly > 0.8 and correlation < 0.8:
                        redundancy_score = min(redundancy_score, 5)  # Moderate non-linear dependency
                except:
                    pass
                    
            except:
                pass
                
        elif is_categorical1 and is_categorical2:
            try:
                # Calculate Cramer's V for categorical columns
                crosstab = pd.crosstab(data1, data2)
                chi2, _, _, _ = stats.chi2_contingency(crosstab)
                n = crosstab.sum().sum()
                min_dim = min(crosstab.shape) - 1
                
                if n * min_dim > 0:
                    cramers_v = np.sqrt(chi2 / (n * min_dim))
                    
                    # Penalize high Cramer's V (close to 1)
                    if cramers_v > 0.9:
                        redundancy_score = 0  # Completely redundant
                    elif cramers_v > 0.7:
                        redundancy_score = 3  # Highly redundant
                    elif cramers_v > 0.5:
                        redundancy_score = 6  # Moderately redundant
                
                # Check for functional dependency
                # (if one column can be perfectly determined from the other)
                try:
                    # For each value in col1, check if it maps to exactly one value in col2
                    value_map = {}
                    functional_dependency = True
                    
                    for val1, val2 in zip(data1, data2):
                        if val1 in value_map and value_map[val1] != val2:
                            functional_dependency = False
                            break
                        value_map[val1] = val2
                    
                    if functional_dependency and len(value_map) > 1:
                        redundancy_score = min(redundancy_score, 1)  # Strong functional dependency
                except:
                    pass
            except:
                pass
                
        elif (is_numeric1 and not is_categorical1 and is_categorical2) or \
             (is_numeric2 and not is_categorical2 and is_categorical1):
            # For mixed types, check if the categorical perfectly separates the numeric
            
            # Ensure numeric is first, categorical is second
            if is_categorical1:
                num_data = data2
                cat_data = data1
            else:
                num_data = data1
                cat_data = data2
                
            try:
                # Calculate eta squared (similar to R² in ANOVA)
                categories = cat_data.unique()
                
                if len(categories) > 1:
                    # Calculate between-group and total variance
                    group_means = {cat: num_data[cat_data == cat].mean() 
                                  for cat in categories if sum(cat_data == cat) > 0}
                    
                    # If each category maps to exactly one numeric value, they're functionally dependent
                    if all(num_data[cat_data == cat].nunique() == 1 for cat in categories if sum(cat_data == cat) > 0):
                        redundancy_score = 2  # Strong functional relationship
                        
                    # Check for high eta squared (categorical explains most of numeric variance)
                    elif len(group_means) > 0:
                        overall_mean = num_data.mean()
                        
                        # Between-group sum of squares
                        between_ss = sum((group_means[cat] - overall_mean) ** 2 * sum(cat_data == cat)
                                        for cat in group_means.keys())
                        
                        # Total sum of squares
                        total_ss = sum((x - overall_mean) ** 2 for x in num_data)
                        
                        # Calculate eta squared
                        eta_squared = between_ss / total_ss if total_ss > 0 else 0
                        
                        # Penalize high eta squared
                        if eta_squared > 0.95:
                            redundancy_score = min(redundancy_score, 3)
                        elif eta_squared > 0.8:
                            redundancy_score = min(redundancy_score, 6)
                
            except:
                pass
        
        # Information overlap calculation for any column type
        try:
            # Calculate mutual information for any column pair
            mi = mutual_info_classif(data1_numeric.values.reshape(-1, 1), data2_numeric)[0]
            
            # Calculate individual entropies
            h1 = stats.entropy(data1.value_counts(normalize=True)) if data1.nunique() > 1 else 0
            h2 = stats.entropy(data2.value_counts(normalize=True)) if data2.nunique() > 1 else 0
            
            # Calculate information overlap ratio
            if max(h1, h2) > 0:
                overlap_ratio = mi / max(h1, h2)
                
                # Penalize high overlap
                if overlap_ratio > 0.9:
                    redundancy_score = min(redundancy_score, 1)
                elif overlap_ratio > 0.7:
                    redundancy_score = min(redundancy_score, 4)
                elif overlap_ratio > 0.5:
                    redundancy_score = min(redundancy_score, 7)
        except:
            pass
    
    # Cap the score
    scores['redundancy_penalization'] = max(0, min(10, redundancy_score))
    
    # Calculate Practical Utility Score (NEW)
    # Start with perfect score and reduce based on practical utility issues
    utility_score = 10
    
    # Check if both columns are low value
    if is_low_value_col1 and is_low_value_col2:
        utility_score = 0  # No utility in visualizing two low-value columns
    elif is_low_value_col1 or is_low_value_col2:
        # If only one column is low value
        utility_score = 3  # Limited utility
        
        # Special case for timestamp + categorical that might be useful
        if (is_temporal1 and is_categorical2 and not is_low_value_col2 and data2.nunique() <= 10) or \
           (is_temporal2 and is_categorical1 and not is_low_value_col1 and data1.nunique() <= 10):
            # Time series by category could be useful if the categorical has few categories
            utility_score = 6
            
        # Time + numerical could still be useful for trend analysis
        if (is_temporal1 and is_numeric2 and not is_categorical2 and not is_low_value_col2) or \
           (is_temporal2 and is_numeric1 and not is_categorical1 and not is_low_value_col1):
            # Time series for a metric could be useful
            utility_score = 7
    else:
        # Neither column is low value
        
        # Reward certain combinations that are typically insightful
        # Categorical + Numerical combinations (like boxplots)
        if (is_categorical1 and is_numeric2 and not is_categorical2) or \
           (is_categorical2 and is_numeric1 and not is_categorical1):
            # Boxplots, bar charts - highly insightful
            utility_score = 10
        
        # Two categorical columns - potentially insightful for heatmaps
        elif is_categorical1 and is_categorical2:
            # If both have reasonable cardinality
            if data1.nunique() <= 15 and data2.nunique() <= 15:
                utility_score = 9
            else:
                # Too many categories makes visualization cluttered
                utility_score = max(3, 10 - (data1.nunique() + data2.nunique()) / 10)
        
        # Two numerical columns - potentially insightful for scatter plots
        elif is_numeric1 and is_numeric2 and not is_categorical1 and not is_categorical2:
            # Scatter plots - potentially insightful
            utility_score = 8
            
            # If they have good statistical association, boost the score
            if scores['statistical_association'] > 6:
                utility_score = 10
    
    # Cap the score
    scores['practical_utility_score'] = max(0, min(10, utility_score))
    
    # Calculate total score using weighted formula that prioritizes practical utility
    scores['total_score'] = (
        0.15 * scores['statistical_association'] + 
        0.10 * scores['visualization_complexity'] + 
        0.15 * scores['pattern_detection'] + 
        0.05 * scores['anomaly_highlighting'] +
        0.15 * scores['information_complementarity'] +
        0.10 * scores['redundancy_penalization'] +
        0.30 * scores['practical_utility_score']  # Heavily weight practical utility
    )
    
    # Final penalty for ID column pairs - prioritize columns that are not IDs
    if is_id_col1 or is_id_col2:
        scores['total_score'] *= 0.4  # Reduce total score by 60% for pairs with ID columns
    
    # Final penalty for pairs with low-value columns like timestamps
    if is_low_value_col1 and is_low_value_col2:
        scores['total_score'] *= 0.3  # Severe penalty for two low-value columns
    elif is_low_value_col1 or is_low_value_col2:
        scores['total_score'] *= 0.6  # Moderate penalty for one low-value column
    
    return scores

def calculate_triple_score(df, col1, col2, col3):
    """
    Calculate a score for a combination of three columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame with clean data.
    col1 : str
        The name of the first column
    col2 : str
        The name of the second column
    col3 : str
        The name of the third column
        
    Returns:
    --------
    dict
        A dictionary containing component scores and the total score
    """
    scores = {
        'dimensional_balance': 0,       # How well balanced the dimensions are
        'information_density': 0,       # How much information is conveyed
        'visualization_feasibility': 0, # How practical it is to visualize
        'insight_potential': 0,         # Potential for generating insights
        'interaction_synergy': 0,       # How well the columns interact
        'complexity_penalty': 0,        # Penalty for excessive complexity
        'total_score': 0
    }
    
    # Get column data and determine data types
    data1 = df[col1]
    data2 = df[col2]
    data3 = df[col3]
    
    # Check data types
    is_numeric1 = pd.api.types.is_numeric_dtype(data1)
    is_numeric2 = pd.api.types.is_numeric_dtype(data2)
    is_numeric3 = pd.api.types.is_numeric_dtype(data3)
    
    is_temporal1 = pd.api.types.is_datetime64_any_dtype(data1)
    is_temporal2 = pd.api.types.is_datetime64_any_dtype(data2)
    is_temporal3 = pd.api.types.is_datetime64_any_dtype(data3)
    
    # Define categorical as either explicitly categorical or numeric with few unique values
    is_categorical1 = not is_numeric1 or (is_numeric1 and data1.nunique() <= 20)
    is_categorical2 = not is_numeric2 or (is_numeric2 and data2.nunique() <= 20)
    is_categorical3 = not is_numeric3 or (is_numeric3 and data3.nunique() <= 20)
    
    # Check for ID and low-value columns
    id_pattern = r'(^|\s|_)(id|key|code|uuid|identifier|seq|sequence|num)($|\s|_)'
    timestamp_pattern = r'(^|\s|_)(timestamp|login|last_login|logged_in|created_at|updated_at|date_added|modified)($|\s|_)'
    
    is_id_col1 = bool(re.search(id_pattern, col1.lower()))
    is_id_col2 = bool(re.search(id_pattern, col2.lower()))
    is_id_col3 = bool(re.search(id_pattern, col3.lower()))
    
    is_low_value_col1 = is_id_col1 or bool(re.search(timestamp_pattern, col1.lower()))
    is_low_value_col2 = is_id_col2 or bool(re.search(timestamp_pattern, col2.lower()))
    is_low_value_col3 = is_id_col3 or bool(re.search(timestamp_pattern, col3.lower()))
    
    # Count low value columns
    low_value_count = sum([is_low_value_col1, is_low_value_col2, is_low_value_col3])
    
    # Count dimensional columns (categorical or temporal)
    dimensional_columns = []
    if is_categorical1 or is_temporal1:
        dimensional_columns.append(col1)
    if is_categorical2 or is_temporal2:
        dimensional_columns.append(col2)
    if is_categorical3 or is_temporal3:
        dimensional_columns.append(col3)
    
    dimensional_count = len(dimensional_columns)
    
    # Get cardinality (number of unique values)
    card1 = data1.nunique()
    card2 = data2.nunique()
    card3 = data3.nunique()
    
    # Check for excessive cardinality that might make visualization difficult
    high_cardinality = card1 > 20 or card2 > 20 or card3 > 20
    
    # 1. Dimensional Balance Score (0-10)
    # Ideal: One dimensional column (categorical/temporal) and two metric columns
    if dimensional_count == 1:
        # Perfect balance - one dimension and two metrics
        scores['dimensional_balance'] = 10
    elif dimensional_count == 2:
        # Two dimensions and one metric - still good but less ideal
        scores['dimensional_balance'] = 8
        
        # If both dimensions have reasonable cardinality, score higher
        if all(df[col].nunique() <= 10 for col in dimensional_columns):
            scores['dimensional_balance'] = 9
    elif dimensional_count == 3:
        # Three dimensions - may be hard to visualize effectively
        scores['dimensional_balance'] = 5
        
        # If all have low cardinality, it might work
        if all(df[col].nunique() <= 7 for col in dimensional_columns):
            scores['dimensional_balance'] = 7
    else:
        # No dimensional columns - not good for 3D visualization
        scores['dimensional_balance'] = 3
    
    # Penalize if we have too many low-value columns
    if low_value_count >= 2:
        scores['dimensional_balance'] *= 0.5
    elif low_value_count == 1:
        scores['dimensional_balance'] *= 0.8
    
    # 2. Information Density Score (0-10)
    # Measure how much information is conveyed by this combination
    
    # Start with base score
    info_density_score = 7
    
    # Check if the triplet forms a "fact table + dimensions" pattern
    has_fact_dimension_pattern = False
    metric_columns = []
    
    # Identify metrics vs dimensions
    if is_numeric1 and not is_categorical1 and not is_low_value_col1:
        metric_columns.append(col1)
    if is_numeric2 and not is_categorical2 and not is_low_value_col2:
        metric_columns.append(col2)
    if is_numeric3 and not is_categorical3 and not is_low_value_col3:
        metric_columns.append(col3)
    
    # Fact table pattern: at least one metric and at least one dimension
    if len(metric_columns) >= 1 and dimensional_count >= 1:
        has_fact_dimension_pattern = True
        info_density_score += 2
    
    # If we have ID columns or low-value columns, reduce score
    info_density_score -= low_value_count * 2
    
    # Check for potential statistical relationships
    if len(metric_columns) >= 2:
        # If we have multiple metrics, check if they're related
        # Calculate average pairwise correlation between metrics
        corrs = []
        for i, col_i in enumerate(metric_columns):
            for col_j in metric_columns[i+1:]:
                try:
                    corr = abs(df[col_i].corr(df[col_j]))
                    corrs.append(corr)
                except:
                    pass
        
        if corrs:
            avg_corr = sum(corrs) / len(corrs)
            if avg_corr > 0.7:  # Strong correlation
                info_density_score += 1
    
    # Penalize if dimensions don't partition the data well
    if dimensional_count > 0:
        # For each dimensional column, check how evenly it splits the data
        for dim_col in dimensional_columns:
            val_counts = df[dim_col].value_counts(normalize=True)
            if len(val_counts) > 0:
                # Calculate entropy of distribution
                entropy = -sum(p * np.log2(p) for p in val_counts if p > 0)
                max_entropy = np.log2(len(val_counts)) if len(val_counts) > 0 else 0
                
                # If entropy is low, dimension doesn't partition data well
                if max_entropy > 0:
                    entropy_ratio = entropy / max_entropy
                    if entropy_ratio < 0.6:  # Uneven distribution
                        info_density_score -= 1
    
    scores['information_density'] = max(0, min(10, info_density_score))
    
    # 3. Visualization Feasibility Score (0-10)
    # How practical it is to visualize this combination
    
    # Start with base feasibility score
    vis_feasibility = 8
    
    # Penalize for high cardinality
    if high_cardinality:
        highest_card = max(card1, card2, card3)
        # Severe penalty for very high cardinality
        if highest_card > 50:
            vis_feasibility -= 5
        elif highest_card > 30:
            vis_feasibility -= 3
        else:
            vis_feasibility -= 1
    
    # Best case: 1 dimension + 2 metrics
    if dimensional_count == 1 and len(metric_columns) == 2:
        vis_feasibility += 2
        
        # If the dimension has reasonable cardinality (5-15 values), it's optimal
        dim_col = dimensional_columns[0]
        dim_card = df[dim_col].nunique()
        if 5 <= dim_card <= 15:
            vis_feasibility += 1
    
    # Second best: 2 dimensions + 1 metric, with reasonable cardinality
    elif dimensional_count == 2 and len(metric_columns) == 1:
        if all(df[col].nunique() <= 10 for col in dimensional_columns):
            vis_feasibility += 1
    
    # Visualizing 3 categorical dimensions is challenging
    elif dimensional_count == 3:
        vis_feasibility -= 2
        
        # Unless all have very few categories
        if all(df[col].nunique() <= 5 for col in dimensional_columns):
            vis_feasibility += 3
    
    # Penalize for missing dimension (all metrics)
    elif dimensional_count == 0:
        vis_feasibility -= 2
    
    # Penalize heavily for low-value columns
    vis_feasibility -= low_value_count * 2
    
    scores['visualization_feasibility'] = max(0, min(10, vis_feasibility))
    
    # 4. Insight Potential Score (0-10)
    # Potential for generating valuable insights
    
    # Base potential score
    insight_score = 7
    
    # Ideal: One clean categorical dimension, two related metrics
    if dimensional_count == 1 and len(metric_columns) == 2:
        # Perfect for comparison analysis
        insight_score += 2
        
        # Check if metrics are related but not perfectly correlated
        if len(metric_columns) >= 2:
            try:
                corr = abs(df[metric_columns[0]].corr(df[metric_columns[1]]))
                if 0.3 < corr < 0.9:  # Related but not redundant
                    insight_score += 1
            except:
                pass
    
    # Time series with metric and category can be insightful
    elif sum([is_temporal1, is_temporal2, is_temporal3]) == 1 and len(metric_columns) >= 1:
        insight_score += 2
        
        # If we also have a categorical dimension, even better
        if sum([is_categorical1, is_categorical2, is_categorical3]) >= 1:
            insight_score += 1
    
    # Two dimensions and one metric can reveal interactions
    elif dimensional_count == 2 and len(metric_columns) == 1:
        insight_score += 1
        
        # If dimensions have reasonable cardinality
        if all(df[col].nunique() <= 12 for col in dimensional_columns):
            insight_score += 1
    
    # Penalize for low-value columns
    insight_score -= low_value_count * 2
    
    # Penalize for ID columns
    if is_id_col1 or is_id_col2 or is_id_col3:
        insight_score -= 3
    
    scores['insight_potential'] = max(0, min(10, insight_score))
    
    # 5. Interaction Synergy Score (0-10)
    # How well the three columns complement each other
    
    # Base synergy score
    synergy_score = 6
    
    # Check for meaningful patterns between all three columns
    # Metrics should be related to dimensions in a meaningful way
    if dimensional_count >= 1 and len(metric_columns) >= 1:
        # For each dimensional column, check if it meaningfully separates metrics
        for dim_col in dimensional_columns:
            for metric_col in metric_columns:
                try:
                    # Group by dimension and calculate variance in metric
                    grouped = df.groupby(dim_col)[metric_col]
                    
                    # Calculate between-group variance
                    group_means = grouped.mean()
                    overall_mean = df[metric_col].mean()
                    
                    # Simple F-statistic approximation
                    between_var = sum((m - overall_mean)**2 for m in group_means) / len(group_means)
                    within_var = grouped.var().mean()
                    
                    if within_var > 0:
                        f_approx = between_var / within_var
                        
                        # If F is high, dimension meaningfully separates the metric
                        if f_approx > 2:
                            synergy_score += 1
                        if f_approx > 5:
                            synergy_score += 1
                except:
                    pass
    
    # For multiple dimensions, check if they interact meaningfully
    if dimensional_count >= 2:
        dim_pairs = []
        for i, col_i in enumerate(dimensional_columns):
            for col_j in dimensional_columns[i+1:]:
                dim_pairs.append((col_i, col_j))
        
        # Check for meaningful interactions between dimensions
        for dim1, dim2 in dim_pairs:
            try:
                # Create contingency table
                crosstab = pd.crosstab(df[dim1], df[dim2])
                
                # Calculate chi-square
                chi2, p, _, _ = stats.chi2_contingency(crosstab)
                
                # If significant relationship exists between dimensions
                if p < 0.05:
                    synergy_score += 1
                    if p < 0.001:
                        synergy_score += 1
            except:
                pass
    
    # Penalize for redundant information
    redundancy_penalty = 0
    
    # Check for high correlations among metrics
    if len(metric_columns) >= 2:
        metric_pairs = []
        for i, col_i in enumerate(metric_columns):
            for col_j in metric_columns[i+1:]:
                metric_pairs.append((col_i, col_j))
        
        # Check for high correlations
        for met1, met2 in metric_pairs:
            try:
                corr = abs(df[met1].corr(df[met2]))
                if corr > 0.9:  # Very high correlation
                    redundancy_penalty += 2
            except:
                pass
    
    # Check for functional dependencies between dimensions
    if dimensional_count >= 2:
        for dim1, dim2 in dim_pairs:
            # Check if dim1 uniquely determines dim2
            grouped = df.groupby(dim1)[dim2].nunique()
            if all(grouped <= 1):
                redundancy_penalty += 2
    
    synergy_score -= redundancy_penalty
    synergy_score -= low_value_count * 1.5
    
    scores['interaction_synergy'] = max(0, min(10, synergy_score))
    
    # 6. Complexity Penalty (0-10, higher is better = lower penalty)
    # Start with perfect score and reduce for complexity issues
    complexity_score = 10
    
    # Penalize for high total cardinality
    total_cardinality = card1 * card2 * card3
    if total_cardinality > 10000:
        complexity_score -= 5
    elif total_cardinality > 1000:
        complexity_score -= 3
    elif total_cardinality > 500:
        complexity_score -= 1
    
    # Penalize for multiple high-cardinality dimensions
    high_card_dims = sum(1 for col in dimensional_columns if df[col].nunique() > 10)
    complexity_score -= high_card_dims * 2
    
    # Heavily penalize for low-value columns
    complexity_score -= low_value_count * 2
    
    # Penalize for ID columns
    if is_id_col1 or is_id_col2 or is_id_col3:
        complexity_score -= 3
    
    scores['complexity_penalty'] = max(0, min(10, complexity_score))
    
    # Calculate total score using weighted components
    scores['total_score'] = (
        0.20 * scores['dimensional_balance'] + 
        0.15 * scores['information_density'] + 
        0.20 * scores['visualization_feasibility'] + 
        0.20 * scores['insight_potential'] +
        0.15 * scores['interaction_synergy'] +
        0.10 * scores['complexity_penalty']
    )
    
    # Final penalty for combinations with excessive low-value columns
    if low_value_count >= 2:
        scores['total_score'] *= 0.4  # Severe penalty
    elif low_value_count == 1:
        scores['total_score'] *= 0.8  # Moderate penalty
    
    # Final penalty if all three are ID columns
    if is_id_col1 and is_id_col2 and is_id_col3:
        scores['total_score'] *= 0.2  # Extreme penalty
    
    return scores

def get_visualization_recommendation(row, df, scores_dict):
    """
    Generate visualization recommendations using the decision tree.
    
    Parameters:
    -----------
    row : pandas.Series
        A row from the scored dataframe
    df : pandas.DataFrame
        The original dataframe with data
    scores_dict : dict
        Dictionary containing scores for all columns and pairs
        
    Returns:
    --------
    str
        Recommended visualization type
    """
    if row['Type'] == 'Column':
        return get_vis_type_for_single_column(df, row['Name'], scores_dict[row['Name']])
    elif row['Type'] == 'Pair':
        # Enhanced pair column extraction with multiple separators
        pair_name = row['Name']
        columns = []
        
        # Try different separator patterns in order of likelihood
        for sep in [' & ', ', ', ' vs ', ' by ', ' and ', ' with ']:
            if sep in pair_name:
                columns = [col.strip() for col in pair_name.split(sep)]
                if len(columns) >= 2:
                    break
        
        # If no separator worked and it looks like a tuple, try to parse it
        if len(columns) < 2 and pair_name.startswith('(') and pair_name.endswith(')'):
            try:
                # Remove parentheses and split by comma
                cols_str = pair_name[1:-1]
                extracted_columns = []
                for col in cols_str.split(','):
                    col = col.strip()
                    # Remove quotes if present
                    if (col.startswith("'") and col.endswith("'")) or (col.startswith('"') and col.endswith('"')):
                        col = col[1:-1]
                    extracted_columns.append(col)
                if len(extracted_columns) >= 2:
                    columns = extracted_columns
            except:
                # Fallback if tuple parsing fails
                pass
        
        # Last resort fallback
        if len(columns) < 2:
            print(f"WARNING: Could not parse pair columns from: {pair_name}. Using naive ' & ' split.")
            columns = pair_name.split(' & ')
        
        # Ensure we have two valid columns
        if len(columns) >= 2:
            col1, col2 = columns[0], columns[1]
            # Verify columns exist in dataframe
            if col1 in df.columns and col2 in df.columns:
                return get_vis_type_for_pair(df, col1, col2, scores_dict[row['Name']])
            else:
                print(f"WARNING: One or more columns not found in dataframe: {col1}, {col2}")
                # Find columns that do exist and use them if possible
                valid_columns = [col for col in columns if col in df.columns]
                if len(valid_columns) >= 2:
                    return get_vis_type_for_pair(df, valid_columns[0], valid_columns[1], scores_dict[row['Name']])
                elif len(valid_columns) == 1 and len(df.columns) > 1:
                    # If only one valid column, find another column to pair with
                    other_col = next((col for col in df.columns if col != valid_columns[0]), None)
                    if other_col:
                        print(f"Using fallback columns for pair: {valid_columns[0]}, {other_col}")
                        return get_vis_type_for_pair(df, valid_columns[0], other_col, scores_dict[row['Name']])
        
        # Ultimate fallback - use first two columns in dataframe
        if len(df.columns) >= 2:
            print(f"WARNING: Using first two columns in dataframe as fallback for pair")
            return get_vis_type_for_pair(df, df.columns[0], df.columns[1], scores_dict[row['Name']])
        else:
            # If only one column, treat as single column
            return get_vis_type_for_single_column(df, df.columns[0], scores_dict[df.columns[0]])
    elif row['Type'] == 'Triple':
        # Enhanced triple extraction with better error handling
        triple_name = row['Name']
        columns = []
        
        # Try different separator patterns
        for sep in [' & ', ', ']:
            if sep in triple_name:
                columns = [col.strip() for col in triple_name.split(sep)]
                if len(columns) >= 3:
                    break
        
        # If no separator worked or we don't have 3 columns, try to parse more complex formats
        if len(columns) < 3:
            # Try to handle "X vs Y by Z" format
            if ' vs ' in triple_name and ' by ' in triple_name:
                parts = triple_name.split(' vs ')
                col1 = parts[0].strip()
                parts2 = parts[1].split(' by ')
                col2 = parts2[0].strip()
                col3 = parts2[1].strip()
                columns = [col1, col2, col3]
        
        # Last resort fallback
        if len(columns) < 3:
            print(f"WARNING: Could not parse triple columns from: {triple_name}. Using naive ' & ' split.")
            columns = triple_name.split(' & ')
        
        # Ensure we have three valid columns
        if len(columns) >= 3:
            col1, col2, col3 = columns[0], columns[1], columns[2]
            # Verify columns exist in dataframe
            if col1 in df.columns and col2 in df.columns and col3 in df.columns:
                return get_vis_type_for_triple(df, col1, col2, col3, scores_dict[row['Name']])
            else:
                # Find columns that do exist
                valid_columns = [col for col in columns if col in df.columns]
                if len(valid_columns) >= 3:
                    return get_vis_type_for_triple(df, valid_columns[0], valid_columns[1], valid_columns[2], scores_dict[row['Name']])
        
        # Ultimate fallback
        if len(df.columns) >= 3:
            print(f"WARNING: Using first three columns in dataframe as fallback for triple")
            return get_vis_type_for_triple(df, df.columns[0], df.columns[1], df.columns[2], scores_dict[row['Name']])
        elif len(df.columns) >= 2:
            # If only two columns, treat as pair
            return get_vis_type_for_pair(df, df.columns[0], df.columns[1], scores_dict[row['Name']])
        else:
            # If only one column, treat as single column
            return get_vis_type_for_single_column(df, df.columns[0], scores_dict[df.columns[0]])

def score_all_columns_and_pairs(df):
    """
    Score all columns and column pairs in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame with clean data.
        
    Returns:
    --------
    tuple
        (column_scores, pair_scores, triple_scores, groupby_scores) - dictionaries of scores
    """
    # For very large datasets, sample rows to improve performance
    max_rows_for_full_analysis = 10000
    if len(df) > max_rows_for_full_analysis:
        sample_size = min(max_rows_for_full_analysis, int(len(df) * 0.3))  # Sample 30% or max 10k rows
        df_sample = df.sample(sample_size, random_state=42)
        # Removed the info message for less clutter
    else:
        df_sample = df
    
    # Remove individual progress bars and loading messages
    # Score individual columns
    column_scores = {}
    columns = list(df_sample.columns)
    for i, column in enumerate(columns):
        column_scores[column] = calculate_column_score(df_sample, column)
    
    # Score column pairs without progress tracking
    # Calculate total number of pairs
    total_pairs = (len(columns) * (len(columns) - 1)) // 2
    
    # Calculate column pairs in sorted order
    pair_scores = {}
    pair_count = 0
    
    # Pre-compute categorical status for all columns to avoid redundant calculations
    column_types = {}
    for col in columns:
        is_numeric = pd.api.types.is_numeric_dtype(df_sample[col])
        is_temporal = pd.api.types.is_datetime64_any_dtype(df_sample[col])
        is_categorical = False
        if not is_numeric and not is_temporal:
            is_categorical = True
        elif is_numeric and df_sample[col].nunique() <= 20:
            is_categorical = True
            
        # Check if column is low-value or ID column
        id_pattern = r'(^|\s|_)(id|key|code|uuid|identifier|seq|sequence|num)($|\s|_)'
        timestamp_pattern = r'(^|\s|_)(timestamp|login|last_login|logged_in|created_at|updated_at|date_added|modified)($|\s|_)'
        
        is_id_col = bool(re.search(id_pattern, col.lower()))
        is_low_value = is_id_col or bool(re.search(timestamp_pattern, col.lower()))
        
        column_types[col] = {
            'is_numeric': is_numeric,
            'is_temporal': is_temporal,
            'is_categorical': is_categorical,
            'is_id_col': is_id_col,
            'is_low_value': is_low_value
        }
    
    # Skip some column pairs for performance if dataset is large
    skip_low_value_pairs = len(columns) > 15
    
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns[i+1:], i+1):
            # Skip low-value column pairs to improve performance with large schemas
            if skip_low_value_pairs:
                # Skip pairs where both columns are low value
                if column_types[col1]['is_low_value'] and column_types[col2]['is_low_value']:
                    # Update counter but skip calculation
                    pair_count += 1
                    continue
            
            pair_name = f"{col1} & {col2}"
            pair_scores[pair_name] = calculate_pair_score(df_sample, col1, col2)
            
            # Update counter
            pair_count += 1
    
    # Score column triples without progress tracking
    
    # Calculate total number of triples (combinations of 3 columns from n columns)
    total_triples = (len(columns) * (len(columns) - 1) * (len(columns) - 2)) // 6
    
    # Limit the number of triples to analyze for large datasets
    max_triples_to_analyze = 100
    analyze_all_triples = total_triples <= max_triples_to_analyze
    
    # Calculate column triples
    triple_scores = {}
    triple_count = 0
    
    # Find dimensional columns (categorical or temporal)
    dimensional_columns = [col for col in columns if column_types[col]['is_categorical'] or column_types[col]['is_temporal']]
    
    # For each potential triple combination
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns[i+1:], i+1):
            for k, col3 in enumerate(columns[j+1:], j+1):
                # Skip if we're not analyzing all triples and we've reached the limit
                if not analyze_all_triples and triple_count >= max_triples_to_analyze:
                    break
                
                # Skip if all three columns are low value
                if (column_types[col1]['is_low_value'] and 
                    column_types[col2]['is_low_value'] and 
                    column_types[col3]['is_low_value']):
                    triple_count += 1
                    continue
                
                # Calculate how many dimensional columns we have in this triple
                dimensional_count = sum([
                    1 if col in dimensional_columns else 0 
                    for col in [col1, col2, col3]
                ])
                
                # Skip if there are too many dimensional columns (more than 1)
                if dimensional_count > 1:
                    triple_count += 1
                    continue
                
                # Score this triple
                triple_name = f"{col1} & {col2} & {col3}"
                triple_scores[triple_name] = calculate_triple_score(df_sample, col1, col2, col3)
                
                # Update counter
                triple_count += 1
    
    # Score columns for groupby and aggregation
    
    # Score individual columns for groupby potential
    groupby_column_scores = {}
    agg_column_scores = {}
    
    for i, column in enumerate(columns):
        # Score as groupby candidate
        groupby_column_scores[column] = score_groupby_column(df_sample, column)
        
        # Score as aggregation candidate
        agg_column_scores[column] = score_aggregation_column(df_sample, column)
    
    # Find top groupby and aggregation candidates
    groupby_candidates = sorted([(col, scores['total_score']) 
                                for col, scores in groupby_column_scores.items()], 
                                key=lambda x: x[1], reverse=True)[:10]
    
    agg_candidates = sorted([(col, scores['total_score']) 
                            for col, scores in agg_column_scores.items()], 
                            key=lambda x: x[1], reverse=True)[:10]
    
    # Score groupby-aggregation pairs
    groupby_pair_scores = {}
    
    # Only evaluate promising pairs for efficiency
    for i, (groupby_col, groupby_score) in enumerate(groupby_candidates):
        for j, (agg_col, agg_score) in enumerate(agg_candidates):
            if groupby_col != agg_col:  # Don't group by and aggregate the same column
                pair_name = (groupby_col, agg_col)
                groupby_pair_scores[pair_name] = calculate_groupby_pair_score(df_sample, groupby_col, agg_col)
    
    return column_scores, pair_scores, triple_scores, groupby_pair_scores

def score_groupby_column(df, column_name):
    """
    Score a column as a candidate for groupby operations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame with clean data.
    column_name : str
        The name of the column to score
        
    Returns:
    --------
    dict
        A dictionary containing component scores and the total score
    """
    scores = {
        'cardinality_score': 0,
        'data_type_score': 0,
        'name_relevance_score': 0,
        'distribution_score': 0,
        'missing_value_score': 0,
        'total_score': 0
    }
    
    # Get column data
    col_data = df[column_name]
    
    # Check data types
    is_numeric = pd.api.types.is_numeric_dtype(col_data)
    is_temporal = pd.api.types.is_datetime64_any_dtype(col_data)
    is_string = pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data)
    
    # Calculate cardinality ratio
    total_rows = len(df)
    unique_values = col_data.nunique()
    cardinality_ratio = unique_values / total_rows if total_rows > 0 else 0
    
    # 1. Cardinality Score (0-10)
    # Ideal cardinality for groupby: not too many, not too few unique values
    if 0.001 <= cardinality_ratio <= 0.2:
        # Ideal range: columns with moderate number of unique values
        if 5 <= unique_values <= 50:
            scores['cardinality_score'] = 10
        elif 3 <= unique_values < 5 or 50 < unique_values <= 100:
            scores['cardinality_score'] = 8
        elif unique_values > 100:
            # Too many unique values makes for cluttered visualization
            penalty = min(7, np.log10(unique_values/100) * 3)
            scores['cardinality_score'] = max(1, 8 - penalty)
        else:  # 1 or 2 unique values
            scores['cardinality_score'] = 3  # Too few categories
    elif 0.2 < cardinality_ratio <= 0.5:
        # Higher cardinality but still potentially useful
        scores['cardinality_score'] = 7 - min(6, (cardinality_ratio - 0.2) * 15)
    elif cardinality_ratio > 0.5:
        # Very high cardinality is not good for groupby
        scores['cardinality_score'] = max(1, 4 - min(3, (cardinality_ratio - 0.5) * 6))
    else:  # cardinality_ratio < 0.001
        # Extremely low cardinality might be a constant
        scores['cardinality_score'] = max(1, unique_values * 2)
    
    # 2. Data Type Score (0-10)
    # Categorical and temporal columns make good groupby candidates
    if is_string:
        # String columns often make good categorical dimensions
        scores['data_type_score'] = 9
    elif is_temporal:
        # Date/time columns are great for time-based grouping
        scores['data_type_score'] = 10
    elif is_numeric and unique_values <= 30:
        # Numeric with few unique values could be categorical
        scores['data_type_score'] = 7
    elif is_numeric:
        # Numeric with many values is not ideal for groupby
        scores['data_type_score'] = 3
    else:
        # Other types
        scores['data_type_score'] = 4
    
    # 3. Name Relevance Score (0-10)
    # Check if column name indicates it's a good groupby candidate
    col_name_lower = column_name.lower()
    
    # Highly relevant dimension names
    dimension_patterns = [
        r'(^|\s|_)(category|type|class|group|segment|region|country|state|city|location)($|\s|_)',
        r'(^|\s|_)(dept|department|division|status|gender|age_group|generation)($|\s|_)',
        r'(^|\s|_)(industry|sector|market|channel|platform|source|media)($|\s|_)',
        r'(^|\s|_)(year|quarter|month|week|day|period|season)($|\s|_)'
    ]
    
    # Check for dimension patterns in name
    name_score = 5  # Base score
    for pattern in dimension_patterns:
        if re.search(pattern, col_name_lower):
            name_score = 10
            break
    
    # Penalize names suggesting this is NOT a groupby column
    negative_patterns = [
        r'(^|\s|_)(id$|uuid|identifier|amount|sum|total|price|cost)($|\s|_)',
        r'(^|\s|_)(value|score|rate|ratio|percent|average|mean|count)($|\s|_)'
    ]
    
    for pattern in negative_patterns:
        if re.search(pattern, col_name_lower):
            name_score = max(0, name_score - 4)
            break
    
    scores['name_relevance_score'] = name_score
    
    # 4. Distribution Score (0-10)
    # Evaluate how well the values are distributed
    if unique_values > 1:
        try:
            # Calculate entropy of the distribution
            value_counts = col_data.value_counts(normalize=True)
            entropy = -sum(p * np.log2(p) for p in value_counts if p > 0)
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(unique_values)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # Ideal: somewhat even distribution (not too uniform, not too skewed)
            if 0.7 <= normalized_entropy <= 0.99:
                scores['distribution_score'] = 10
            elif 0.5 <= normalized_entropy < 0.7:
                scores['distribution_score'] = 8
            elif 0.3 <= normalized_entropy < 0.5:
                scores['distribution_score'] = 6
            elif normalized_entropy > 0.99:
                # Almost perfectly uniform - might be synthetic or not interesting
                scores['distribution_score'] = 7
            else:
                # Very skewed distribution
                scores['distribution_score'] = 4
        except:
            scores['distribution_score'] = 5
    else:
        scores['distribution_score'] = 0  # Only one value
    
    # 5. Missing Value Score (0-10)
    # Few missing values is better for groupby
    missing_ratio = col_data.isna().mean()
    scores['missing_value_score'] = (1 - missing_ratio) * 10
    
    # Calculate total score using weighted formula
    scores['total_score'] = (
        0.35 * scores['cardinality_score'] + 
        0.20 * scores['data_type_score'] + 
        0.15 * scores['name_relevance_score'] +
        0.20 * scores['distribution_score'] +
        0.10 * scores['missing_value_score']
    )
    
    return scores

def score_aggregation_column(df, column_name):
    """
    Score a column as a candidate for aggregation operations (sum, mean, etc.).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame with clean data.
    column_name : str
        The name of the column to score
        
    Returns:
    --------
    dict
        A dictionary containing component scores and the total score
    """
    scores = {
        'numeric_quality_score': 0,
        'name_relevance_score': 0,
        'distribution_score': 0,
        'non_id_score': 0,
        'null_handling_score': 0,
        'total_score': 0
    }
    
    # Get column data
    col_data = df[column_name]
    
    # 1. Numeric Quality Score (0-10)
    # Aggregation columns should be numeric
    if pd.api.types.is_numeric_dtype(col_data):
        numeric_score = 10
        
        # Check if values are mostly integers or mostly decimals
        if col_data.dropna().apply(lambda x: x.is_integer() if hasattr(x, 'is_integer') else True).mean() > 0.9:
            # Mostly integers - good for count/sum
            numeric_score = 9
        else:
            # Has decimals - good for averages, sums, etc.
            numeric_score = 10
            
        # Check for negative values
        if (col_data < 0).any():
            # Has negatives - might be good for net calculations
            numeric_score = 9
            
        scores['numeric_quality_score'] = numeric_score
    else:
        # Non-numeric columns get a low score for aggregation
        scores['numeric_quality_score'] = 2
    
    # 2. Name Relevance Score (0-10)
    # Check if column name indicates it's a good aggregation candidate
    col_name_lower = column_name.lower()
    
    # Highly relevant measure names
    measure_patterns = [
        r'(^|\s|_)(amount|sum|total|value|price|cost|sales|revenue|profit|loss)($|\s|_)',
        r'(^|\s|_)(count|quantity|qty|volume|number|frequency|instances)($|\s|_)',
        r'(^|\s|_)(score|rate|ratio|percent|percentage|average|avg|mean)($|\s|_)'
    ]
    
    # Check for measure patterns in name
    name_score = 5  # Base score
    for pattern in measure_patterns:
        if re.search(pattern, col_name_lower):
            name_score = 10
            break
    
    # Penalize names suggesting this is NOT a measure column
    negative_patterns = [
        r'(^|\s|_)(id$|uuid|identifier|code|key|name|type|category)($|\s|_)',
        r'(^|\s|_)(region|country|state|city|location|status|flag)($|\s|_)'
    ]
    
    for pattern in negative_patterns:
        if re.search(pattern, col_name_lower):
            name_score = max(0, name_score - 5)
            break
    
    scores['name_relevance_score'] = name_score
    
    # 3. Distribution Score (0-10)
    # Evaluate distribution properties for aggregation
    if pd.api.types.is_numeric_dtype(col_data):
        try:
            # Calculate basic statistics
            if len(col_data.dropna()) > 5:
                # Check for approximate normal distribution (common for measures)
                skewness = abs(col_data.skew())
                kurtosis = abs(col_data.kurtosis())
                
                # Look for distributions that are reasonable for aggregation
                if skewness < 3 and kurtosis < 10:
                    # Moderately well-behaved distribution
                    dist_score = 8
                    
                    # Closer to normal is better for certain aggregations like mean
                    if skewness < 1 and kurtosis < 3:
                        dist_score = 10
                else:
                    # Very skewed or heavy-tailed distribution
                    dist_score = 6
                    
                # Check for zero-inflation (common in meaningful measures)
                zero_ratio = (col_data == 0).mean()
                if 0.2 <= zero_ratio <= 0.8:
                    # Zero-inflated distributions are often meaningful measures
                    dist_score = max(dist_score, 7)
                
                scores['distribution_score'] = dist_score
            else:
                scores['distribution_score'] = 5  # Too few values to analyze
        except:
            scores['distribution_score'] = 5
    else:
        scores['distribution_score'] = 3  # Non-numeric, not great for aggregation
    
    # 4. Non-ID Score (0-10)
    # Ensure the numeric column isn't an ID column
    col_name_lower = column_name.lower()
    unique_ratio = col_data.nunique() / len(col_data) if len(col_data) > 0 else 0
    
    # Check if column name contains ID patterns
    id_pattern = r'(^|\s|_)(id|key|code|uuid|identifier|seq|sequence|num)($|\s|_)'
    is_id_col = bool(re.search(id_pattern, col_name_lower))
    
    # Calculate score (higher means less likely to be an ID)
    if is_id_col:
        # Name suggests it's an ID
        id_score = 2
    elif unique_ratio > 0.8:
        # High uniqueness suggests an ID
        id_score = 3
    elif unique_ratio > 0.5:
        # Somewhat high uniqueness, might be an ID
        id_score = 7
    else:
        # Low uniqueness, unlikely to be an ID
        id_score = 10
    
    scores['non_id_score'] = id_score
    
    # 5. Null Handling Score (0-10)
    # Few missing values is better for aggregation
    missing_ratio = col_data.isna().mean()
    if missing_ratio < 0.01:
        # Almost no missing values
        scores['null_handling_score'] = 10
    elif missing_ratio < 0.1:
        # Few missing values
        scores['null_handling_score'] = 8
    elif missing_ratio < 0.3:
        # Some missing values
        scores['null_handling_score'] = 6
    else:
        # Many missing values
        scores['null_handling_score'] = max(0, 10 - (missing_ratio * 20))
    
    # Calculate total score using weighted formula
    scores['total_score'] = (
        0.35 * scores['numeric_quality_score'] + 
        0.20 * scores['name_relevance_score'] +
        0.15 * scores['distribution_score'] +
        0.20 * scores['non_id_score'] +
        0.10 * scores['null_handling_score']
    )
    
    return scores

def calculate_groupby_pair_score(df, groupby_col, agg_col):
    """
    Calculate a score for a groupby-aggregation column pair.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame with clean data.
    groupby_col : str
        The name of the column to group by
    agg_col : str
        The name of the column to aggregate
        
    Returns:
    --------
    dict
        A dictionary containing component scores and the total score
    """
    scores = {
        'group_differentiation': 0,
        'aggregation_meaningfulness': 0,
        'group_size_balance': 0,
        'outlier_robustness': 0,
        'visualization_potential': 0,
        'total_score': 0
    }
    
    # Get column data
    group_data = df[groupby_col]
    agg_data = df[agg_col]
    
    # Skip calculation if agg_data is not numeric
    if not pd.api.types.is_numeric_dtype(agg_data):
        scores['total_score'] = 0
        return scores
    
    # Calculate number of groups and check if we have too many
    unique_groups = group_data.nunique()
    if unique_groups > 100:
        # Too many groups for meaningful visualization or analysis
        scores['total_score'] = max(0, 5 - min(5, np.log10(unique_groups/100) * 2))
        return scores
    elif unique_groups <= 1:
        # No groups to compare
        scores['total_score'] = 0
        return scores
    
    # 1. Group Differentiation Score (0-10)
    # Measure how much the aggregation value differs across groups
    try:
        # Group by and calculate mean for each group
        grouped = df.groupby(groupby_col)[agg_col].mean().reset_index()
        
        if len(grouped) > 1:
            # Calculate coefficient of variation across group means
            group_means = grouped[agg_col]
            cv = group_means.std() / group_means.mean() if group_means.mean() != 0 else 0
            
            # Higher variation between groups is better
            if cv > 1:
                scores['group_differentiation'] = 10
            elif cv > 0.5:
                scores['group_differentiation'] = 8
            elif cv > 0.2:
                scores['group_differentiation'] = 6
            elif cv > 0.1:
                scores['group_differentiation'] = 4
            else:
                scores['group_differentiation'] = 2
            
            # Alternative: use ANOVA to measure differences between groups
            try:
                # Create groups for ANOVA
                groups = []
                for group in grouped[groupby_col].unique():
                    group_values = df[df[groupby_col] == group][agg_col].dropna()
                    if len(group_values) > 0:
                        groups.append(group_values)
                
                if len(groups) > 1 and all(len(g) > 0 for g in groups):
                    f_val, p_val = stats.f_oneway(*groups)
                    
                    # Lower p-value indicates stronger group differences
                    if p_val < 0.001:
                        anova_score = 10
                    elif p_val < 0.01:
                        anova_score = 8
                    elif p_val < 0.05:
                        anova_score = 6
                    elif p_val < 0.1:
                        anova_score = 4
                    else:
                        anova_score = 2
                        
                    # Take the better of CV or ANOVA score
                    scores['group_differentiation'] = max(scores['group_differentiation'], anova_score)
            except:
                pass  # Stick with CV score if ANOVA fails
        else:
            scores['group_differentiation'] = 0
    except:
        scores['group_differentiation'] = 5  # Default if calculation fails
    
    # 2. Aggregation Meaningfulness Score (0-10)
    # Is the aggregation meaningful based on distribution, outliers, etc.
    try:
        # Calculate aggregation statistics
        grouped_sum = df.groupby(groupby_col)[agg_col].sum().reset_index()
        grouped_mean = df.groupby(groupby_col)[agg_col].mean().reset_index()
        
        # Check if sum and mean tell different stories
        sum_cv = grouped_sum[agg_col].std() / grouped_sum[agg_col].mean() if grouped_sum[agg_col].mean() != 0 else 0
        mean_cv = grouped_mean[agg_col].std() / grouped_mean[agg_col].mean() if grouped_mean[agg_col].mean() != 0 else 0
        
        # Meaningful aggregations often show differences in both sum and mean
        if sum_cv > 0.2 and mean_cv > 0.2:
            agg_score = 9
        elif sum_cv > 0.2 or mean_cv > 0.2:
            agg_score = 7
        else:
            agg_score = 5
            
        # Check if data has appropriate skewness for aggregation
        skewness = abs(agg_data.skew())
        if skewness < 5:
            # Not too skewed, good for aggregation
            agg_score += 1
        else:
            # Very skewed, maybe less ideal
            agg_score -= 1
            
        scores['aggregation_meaningfulness'] = max(0, min(10, agg_score))
    except:
        scores['aggregation_meaningfulness'] = 5  # Default if calculation fails
    
    # 3. Group Size Balance Score (0-10)
    # Groups should be reasonably balanced in size
    try:
        group_counts = group_data.value_counts(normalize=True)
        
        # Calculate entropy of group sizes
        entropy = -sum(p * np.log2(p) for p in group_counts if p > 0)
        max_entropy = np.log2(len(group_counts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # More balanced group sizes get higher scores
        if normalized_entropy > 0.9:
            scores['group_size_balance'] = 10  # Very balanced
        elif normalized_entropy > 0.7:
            scores['group_size_balance'] = 8  # Reasonably balanced
        elif normalized_entropy > 0.5:
            scores['group_size_balance'] = 6  # Somewhat balanced
        elif normalized_entropy > 0.3:
            scores['group_size_balance'] = 4  # Not very balanced
        else:
            scores['group_size_balance'] = 2  # Very imbalanced
    except:
        scores['group_size_balance'] = 5  # Default if calculation fails
    
    # 4. Outlier Robustness Score (0-10)
    # Aggregations should be robust to outliers
    try:
        # Check if outliers significantly affect the aggregation
        group_means = df.groupby(groupby_col)[agg_col].mean()
        group_medians = df.groupby(groupby_col)[agg_col].median()
        
        # Calculate mean absolute percentage difference between mean and median
        pct_diff = abs(group_means - group_medians) / (group_medians.replace(0, 1))
        mean_pct_diff = pct_diff.mean()
        
        # Lower difference means aggregation is more robust to outliers
        if mean_pct_diff < 0.1:
            scores['outlier_robustness'] = 10  # Very robust
        elif mean_pct_diff < 0.2:
            scores['outlier_robustness'] = 8  # Robust
        elif mean_pct_diff < 0.5:
            scores['outlier_robustness'] = 6  # Somewhat robust
        elif mean_pct_diff < 1.0:
            scores['outlier_robustness'] = 4  # Not very robust
        else:
            scores['outlier_robustness'] = 2  # Not robust at all
    except:
        scores['outlier_robustness'] = 5  # Default if calculation fails
    
    # 5. Visualization Potential Score (0-10)
    # Assess how well this groupby-aggregate pair would visualize
    try:
        # Calculate stats that suggest good visualization potential
        
        # Number of groups - not too few, not too many
        if 3 <= unique_groups <= 15:
            vis_score = 9  # Ideal number of groups for visualization
        elif 15 < unique_groups <= 30:
            vis_score = 7  # Still visualizable but getting crowded
        elif unique_groups < 3:
            vis_score = 5  # Too few groups
        else:
            vis_score = max(2, 10 - np.log2(unique_groups/15))  # Penalize for many groups
            
        # Check for good distribution of aggregated values
        grouped_agg = df.groupby(groupby_col)[agg_col].sum()
        agg_range = grouped_agg.max() - grouped_agg.min() if len(grouped_agg) > 0 else 0
        
        if agg_range > 0:
            # Calculate how spread out the values are
            normalized_range = grouped_agg / grouped_agg.max()
            pct_below_threshold = (normalized_range < 0.1).mean()
            
            # If too many groups have very small values, visualization is less effective
            if pct_below_threshold > 0.7:
                vis_score -= 3  # Many groups are visually insignificant
            elif pct_below_threshold > 0.5:
                vis_score -= 2  # Several groups are visually insignificant
            elif pct_below_threshold > 0.3:
                vis_score -= 1  # Some groups are visually insignificant
                
        scores['visualization_potential'] = max(0, min(10, vis_score))
    except:
        scores['visualization_potential'] = 5  # Default if calculation fails
    
    # Calculate total score using weighted formula
    scores['total_score'] = (
        0.25 * scores['group_differentiation'] + 
        0.25 * scores['aggregation_meaningfulness'] + 
        0.15 * scores['group_size_balance'] +
        0.15 * scores['outlier_robustness'] +
        0.20 * scores['visualization_potential']
    )
    
    return scores

def get_groupby_visualization_recommendation(df, groupby_col, agg_col, score):
    """
    Generate a visualization recommendation for a groupby-aggregation pair.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The original dataframe with data
    groupby_col : str
        The column to group by
    agg_col : str
        The column to aggregate
    score : dict
        The scores dictionary for this pair
        
    Returns:
    --------
    str
        Recommended visualization type
    """
    # Use the new get_vis_type_for_groupby function
    return get_vis_type_for_groupby(df, groupby_col, agg_col, 'sum', score)

def visualize_triple(df, col1, col2, col3):
    """
    Visualize a combination of three columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame with data.
    col1 : str
        First column name
    col2 : str
        Second column name
    col3 : str
        Third column name
    """
    # Skip if any column doesn't exist in the dataframe
    for col in [col1, col2, col3]:
        if col not in df.columns:
            st.warning(f"Column not found in the dataset: {col}")
            return
    
    # Get column data types
    is_numeric1 = pd.api.types.is_numeric_dtype(df[col1])
    is_numeric2 = pd.api.types.is_numeric_dtype(df[col2])
    is_numeric3 = pd.api.types.is_numeric_dtype(df[col3])
    
    is_temporal1 = pd.api.types.is_datetime64_any_dtype(df[col1])
    is_temporal2 = pd.api.types.is_datetime64_any_dtype(df[col2])
    is_temporal3 = pd.api.types.is_datetime64_any_dtype(df[col3])
    
    # Count numerical and temporal columns
    num_numeric = sum([is_numeric1, is_numeric2, is_numeric3])
    num_temporal = sum([is_temporal1, is_temporal2, is_temporal3])
    
    # Limit to 1000 rows for performance
    if len(df) > 1000:
        df_subset = df.sample(1000, random_state=42)
        st.info(f"Visualizing a sample of 1000 rows from {len(df)} total rows")
    else:
        df_subset = df
    
    # Handle missing values
    df_subset = df_subset.dropna(subset=[col1, col2, col3])
    
    # Get vis type based on column types
    vis_type = get_vis_type_for_triple(df, col1, col2, col3, {})
    
    # Create plot container
    st.subheader(f"Visualization of {col1}, {col2}, and {col3}")
    
    # All three are numerical - 3D Scatter
    if num_numeric == 3 and "3D Scatter" in vis_type:
        fig = px.scatter_3d(
            df_subset, x=col1, y=col2, z=col3,
            title=f"3D Scatter Plot of {col1}, {col2}, and {col3}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Two numerical, one categorical - Scatter with color
    elif num_numeric == 2 and "Scatter Plot with Color" in vis_type:
        # Identify categorical column
        cat_col = col1
        num_cols = [col2, col3]
        if is_numeric1 and is_numeric2:
            cat_col = col3
            num_cols = [col1, col2]
        elif is_numeric1 and is_numeric3:
            cat_col = col2
            num_cols = [col1, col3]
        
        fig = px.scatter(
            df_subset, x=num_cols[0], y=num_cols[1], color=cat_col,
            title=f"Scatter Plot of {num_cols[0]} vs {num_cols[1]} colored by {cat_col}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # One numerical, two categorical - Grouped Bar Chart
    elif num_numeric == 1 and "Bar Chart" in vis_type:
        # Identify numerical column
        num_col = col1 if is_numeric1 else (col2 if is_numeric2 else col3)
        cat_cols = [c for c, is_num in zip([col1, col2, col3], [is_numeric1, is_numeric2, is_numeric3]) if not is_num]
        
        # Aggregate data
        agg_df = df.groupby(cat_cols)[num_col].mean().reset_index()
        
        # Limit to top categories if too many
        if len(agg_df) > 30:
            st.warning(f"Too many category combinations ({len(agg_df)}). Showing top 30.")
            agg_df = agg_df.sort_values(num_col, ascending=False).head(30)
        
        fig = px.bar(
            agg_df, x=cat_cols[0], y=num_col, color=cat_cols[1],
            title=f"Mean {num_col} by {cat_cols[0]} and {cat_cols[1]}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Includes temporal data - Time series with categories
    elif num_temporal >= 1 and num_numeric >= 1 and "Multi-line Chart" in vis_type:
        # Identify temporal column
        time_col = next(col for col, is_temp in zip([col1, col2, col3], [is_temporal1, is_temporal2, is_temporal3]) if is_temp)
        
        # Identify numerical column (if exists)
        num_col = next((col for col, is_num in zip([col1, col2, col3], [is_numeric1, is_numeric2, is_numeric3]) if is_num), None)
        
        # Identify categorical column (not temporal, not the chosen numeric)
        cat_col = next((col for col in [col1, col2, col3] if col != time_col and col != num_col), None)
        
        if num_col and cat_col:
            fig = px.line(
                df_subset.sort_values(time_col), x=time_col, y=num_col, color=cat_col,
                title=f"{num_col} over {time_col} by {cat_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not identify appropriate columns for a time series visualization")
    
    # Fallback
    else:
        st.info(f"Recommended visualization type: {vis_type}")
        st.warning("This visualization type is not yet implemented. Showing a data sample instead.")
        st.dataframe(df_subset[list([col1, col2, col3])].head(10))

def convert_to_datetime(df, column_name):
    """
    Attempt to convert a column to datetime format if it contains date-like strings.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the column
    column_name : str
        The name of the column to convert
        
    Returns:
    --------
    pandas.DataFrame
        The dataframe with the column potentially converted to datetime
    bool
        Whether the conversion was successful
    """
    # Skip if already datetime
    if pd.api.types.is_datetime64_any_dtype(df[column_name]):
        return df, True
    
    # Only attempt to convert string columns
    if not pd.api.types.is_string_dtype(df[column_name]):
        return df, False
    
    # Try to convert to datetime
    try:
        # Make a copy to avoid SettingWithCopyWarning
        df_copy = df.copy()
        df_copy[column_name] = pd.to_datetime(df_copy[column_name])
        return df_copy, True
    except Exception as e:
        st.warning(f"Could not convert {column_name} to datetime: {str(e)}")
        return df, False

def visualize_groupby(df, groupby_col, agg_col):
    """
    Visualize a groupby-aggregation pair.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame with data.
    groupby_col : str
        The column to group by
    agg_col : str
        The column to aggregate (must be numeric)
    """
    # Skip if any column doesn't exist in the dataframe
    if groupby_col not in df.columns or agg_col not in df.columns:
        st.warning(f"One or more columns not found in the dataset: {groupby_col}, {agg_col}")
        return
    
    # Check if the aggregation column is numeric
    if not pd.api.types.is_numeric_dtype(df[agg_col]):
        st.warning(f"Column {agg_col} must be numeric for aggregation")
        return
    
    # Try to convert the groupby column to datetime if it contains date-like strings
    df, datetime_converted = convert_to_datetime(df, groupby_col)
    
    # Get column data types
    is_categorical = False
    is_temporal = pd.api.types.is_datetime64_any_dtype(df[groupby_col])
    is_numeric = pd.api.types.is_numeric_dtype(df[groupby_col])
    
    # Enhanced detection of temporal data - check column name for date/time indicators
    temporal_keywords = ['date', 'time', 'year', 'month', 'day', 'quarter', 'week', 'hour', 'minute', 'second']
    has_temporal_name = any(keyword in groupby_col.lower() for keyword in temporal_keywords)
    
    # If column is string and might contain dates, try additional detection
    if not is_temporal and pd.api.types.is_string_dtype(df[groupby_col]):
        # Sample a few values to check for date patterns
        try:
            first_values = df[groupby_col].dropna().head(5)
            could_be_date = False
            
            for val in first_values:
                # Check for common date separators
                if isinstance(val, str) and (
                    '/' in val or '-' in val or ':' in val or 
                    any(month in val.lower() for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
                ):
                    could_be_date = True
                    break
            
            if could_be_date:
                # Consider it potentially temporal data
                is_temporal = True
                st.info(f"Column '{groupby_col}' appears to contain date-like values. Treating as temporal data.")
        except Exception as e:
            st.warning(f"Error checking date patterns: {str(e)}")
    
    # If column has temporal keywords in name, consider it temporal
    if has_temporal_name and not is_temporal:
        is_temporal = True
        st.info(f"Column '{groupby_col}' has a name suggesting time data. Treating as temporal data.")
    
    # Debug information
    st.write("### Debug Information:")
    st.write(f"Group column: {groupby_col}")
    st.write(f"Aggregation column: {agg_col}")
    st.write(f"is_temporal: {is_temporal}")
    st.write(f"has_temporal_name: {has_temporal_name}")
    st.write(f"datetime_converted: {datetime_converted}")
    st.write(f"is_numeric: {is_numeric}")
    st.write(f"Column dtype: {df[groupby_col].dtype}")
    
    # Show a sample of the groupby column data
    st.write("Sample of groupby column data:")
    st.write(df[groupby_col].head())
    
    # Determine if numeric column should be treated as categorical
    if is_numeric and df[groupby_col].nunique() <= 20:
        is_categorical = True
    elif not is_numeric and not is_temporal:
        is_categorical = True
    
    # Get number of unique groups
    unique_groups = df[groupby_col].nunique()
    
    # For columns with too many unique values, sample the top N groups
    max_groups_to_display = 15
    
    # Create aggregation
    agg_df = df.groupby(groupby_col)[agg_col].agg(['sum', 'mean', 'count']).reset_index()
    
    # Limit number of groups if needed
    if unique_groups > max_groups_to_display:
        # Sort by sum and take top groups
        agg_df = agg_df.sort_values('sum', ascending=False).head(max_groups_to_display)
        st.info(f"Showing top {max_groups_to_display} groups by {agg_col} sum (out of {unique_groups} total)")
    
    # Get the visualization recommendation from our decision tree
    scores = {}  # We'll pass an empty scores dict for now
    vis_type = get_vis_type_for_groupby(df, groupby_col, agg_col, 'sum', scores)
    
    # Debug the visualization type decision
    st.write(f"Visualization type returned by decision tree: {vis_type}")
    
    # Add a placeholder for the chart
    container = st.container()
    st.subheader(f"Visualization of {agg_col} grouped by {groupby_col}")
    st.info(f"Recommended visualization: {vis_type}")
    
    # Sort data appropriately
    if is_temporal:
        try:
            # If it's a proper datetime column, sort normally
            if pd.api.types.is_datetime64_any_dtype(df[groupby_col]):
                agg_df = agg_df.sort_values(groupby_col)
            else:
                # For string dates that might not convert cleanly, try to sort them as dates
                try:
                    # Try to convert temporarily for sorting
                    temp_df = agg_df.copy()
                    temp_df['temp_datetime'] = pd.to_datetime(temp_df[groupby_col], errors='coerce')
                    # Sort by the temporary column and drop it
                    temp_df = temp_df.sort_values('temp_datetime')
                    sorted_order = temp_df.index
                    agg_df = agg_df.loc[sorted_order].reset_index(drop=True)
                except:
                    # If all else fails, try string sorting
                    agg_df = agg_df.sort_values(groupby_col)
        except Exception as e:
            st.warning(f"Error sorting temporal data: {e}")
            # Fallback to regular sorting
            agg_df = agg_df.sort_values(groupby_col)
    else:
        agg_df = agg_df.sort_values('sum', ascending=False)
        
    # Debug sorted results
    st.write("Sorted data for visualization:")
    st.write(agg_df.head())
    
    # Override visualization type if we have strong evidence of temporal data
    if (is_temporal or has_temporal_name) and vis_type != "Line Chart (px.line)" and vis_type != "Area Chart (px.area)":
        st.warning(f"Detected temporal data but got {vis_type} recommendation. Overriding to Line Chart.")
        vis_type = "Line Chart (px.line)"
        
    # Create visualization based on the recommendation
    if "Line Chart" in vis_type or "Area Chart" in vis_type:
        try:
            # For proper visualization of temporal data that isn't in datetime format
            if is_temporal and not pd.api.types.is_datetime64_any_dtype(agg_df[groupby_col]):
                # Try to convert column temporarily for the plot
                try:
                    # First try direct conversion for plotting
                    plot_df = agg_df.copy()
                    # For Plotly line charts, having a proper datetime helps with axis formatting
                    plot_df[groupby_col] = pd.to_datetime(plot_df[groupby_col], errors='coerce')
                    
                    # Check if conversion worked by seeing if we have non-NaT values
                    if plot_df[groupby_col].notna().sum() > 0:
                        st.success("Successfully converted date format for visualization")
                        
                        # Line or Area chart with converted dates
                        if "Area Chart" in vis_type:
                            fig = px.area(plot_df, x=groupby_col, y='sum', 
                                     title=f'Sum of {agg_col} by {groupby_col}',
                                     labels={groupby_col: groupby_col, 'sum': f'Sum of {agg_col}'})
                        else:
                            fig = px.line(plot_df, x=groupby_col, y='sum', 
                                     title=f'Sum of {agg_col} by {groupby_col}',
                                     labels={groupby_col: groupby_col, 'sum': f'Sum of {agg_col}'})
                        container.plotly_chart(fig, use_container_width=True)
                        
                        # Skip the regular plotting
                        return
                    else:
                        # Fall back to regular visualization
                        st.warning("Date conversion had too many errors, falling back to basic plotting")
                except Exception as e:
                    st.warning(f"Error in date conversion for visualization: {e}")
            
            # Regular visualization (for properly formatted dates or fallback)
            if "Area Chart" in vis_type:
                fig = px.area(agg_df, x=groupby_col, y='sum', 
                             title=f'Sum of {agg_col} by {groupby_col}',
                             labels={groupby_col: groupby_col, 'sum': f'Sum of {agg_col}'})
            else:
                fig = px.line(agg_df, x=groupby_col, y='sum', 
                             title=f'Sum of {agg_col} by {groupby_col}',
                             labels={groupby_col: groupby_col, 'sum': f'Sum of {agg_col}'})
            container.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating line/area chart: {e}")
            # Fallback to a basic bar chart
            st.info("Falling back to bar chart due to error")
            fig = px.bar(agg_df, x=groupby_col, y='sum',
                        title=f'Sum of {agg_col} by {groupby_col} (Fallback)',
                        labels={groupby_col: groupby_col, 'sum': f'Sum of {agg_col}'})
            container.plotly_chart(fig, use_container_width=True)
    
    elif "Pie Chart" in vis_type:
        # Pie chart for categorical with few unique values
        fig = px.pie(agg_df, names=groupby_col, values='sum',
                     title=f'Sum of {agg_col} by {groupby_col}')
        container.plotly_chart(fig, use_container_width=True)
        
    elif "Treemap" in vis_type:
        # Treemap for categorical with many unique values
        fig = px.treemap(agg_df, path=[groupby_col], values='sum',
                       title=f'Sum of {agg_col} by {groupby_col}')
        container.plotly_chart(fig, use_container_width=True)
        
    elif "Bar Chart" in vis_type:
        # Bar chart (default for most categorical groupings)
        if unique_groups > 8:
            # Horizontal bar chart for many categories
            fig = px.bar(agg_df, y=groupby_col, x='sum', 
                        title=f'Sum of {agg_col} by {groupby_col}',
                        labels={groupby_col: groupby_col, 'sum': f'Sum of {agg_col}'},
                        height=max(400, 50 * min(unique_groups, max_groups_to_display)))
        else:
            # Vertical bar chart for few categories
            fig = px.bar(agg_df, x=groupby_col, y='sum',
                        title=f'Sum of {agg_col} by {groupby_col}',
                        labels={groupby_col: groupby_col, 'sum': f'Sum of {agg_col}'})
        container.plotly_chart(fig, use_container_width=True)
        
    else:
        # Fallback for any other visualization type
        st.warning(f"Visualization type '{vis_type}' is not fully implemented. Showing a bar chart instead.")
        fig = px.bar(agg_df, x=groupby_col, y='sum',
                    title=f'Sum of {agg_col} by {groupby_col}',
                    labels={groupby_col: groupby_col, 'sum': f'Sum of {agg_col}'})
        container.plotly_chart(fig, use_container_width=True)
    
    # Always show the data table for reference
    with st.expander("View aggregated data"):
        st.dataframe(agg_df, use_container_width=True)

# Check if this file is being imported or run directly
# Only show UI if not being imported (IMPORTING_ONLY env var is not set)
if os.environ.get("IMPORTING_ONLY") != "1":
    # Streamlit app
    st.title("Data Visualization Recommender")
    st.write("Upload a CSV file to get visualization recommendations based on column relationships")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # File upload and main app logic
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Display data sample
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Initialize session state for file identification
            if 'file_id' not in st.session_state:
                st.session_state.file_id = None
            
            # Check if we need to recalculate scores (new file upload)
            current_file_id = hash(uploaded_file.name + str(df.shape) + str(list(df.columns)))
            new_file_upload = st.session_state.file_id != current_file_id
            
            # Calculate scores only if this is a new file or first run
            if new_file_upload or 'all_scores' not in st.session_state:
                # Update file ID
                st.session_state.file_id = current_file_id
                
                # Calculate scores for all columns, pairs, and triples
                with st.spinner("Analyzing data and calculating visualization scores..."):
                    st.info("This process may take a few moments depending on dataset size.")
                    start_time = time.time()
                    column_scores, pair_scores, triple_scores, groupby_scores = score_all_columns_and_pairs(df)
                    calc_time = time.time() - start_time
                    st.success(f"Analysis completed in {calc_time:.2f} seconds!")
                
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
                for groupby_pair, scores in groupby_scores.items():
                    groupby_data.append({
                        'Name': groupby_pair,
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
                    groupby_col, agg_col = row['Name'].split(" [by] ")
                    all_scores.at[idx, 'Recommended Visualization'] = get_groupby_visualization_recommendation(
                        df, groupby_col, agg_col, groupby_scores[row['Name']]
                    )
                
                # Store all results in session state
                st.session_state.column_scores = column_scores
                st.session_state.pair_scores = pair_scores
                st.session_state.triple_scores = triple_scores
                st.session_state.groupby_scores = groupby_scores
                
                # Store the all_scores DataFrame
                st.session_state.all_scores = all_scores
            else:
                # Use cached scores from session state
                column_scores = st.session_state.column_scores
                pair_scores = st.session_state.pair_scores
                triple_scores = st.session_state.triple_scores
                groupby_scores = st.session_state.groupby_scores
                all_scores = st.session_state.all_scores
                
                # Ensure 'Recommended Visualization' column exists (in case of session state issues)
                if 'Recommended Visualization' not in all_scores.columns:
                    # Create a recommended visualization type column
                    all_scores['Recommended Visualization'] = all_scores.apply(
                        lambda row: get_visualization_recommendation(row, df, 
                                                                    {**column_scores, **pair_scores, **triple_scores}),
                        axis=1
                    )
                    
                    # Update visualization recommendations for GroupBy pairs
                    for idx, row in all_scores[all_scores['Type'] == 'GroupBy'].iterrows():
                        groupby_col, agg_col = row['Name'].split(" [by] ")
                        all_scores.at[idx, 'Recommended Visualization'] = get_groupby_visualization_recommendation(
                            df, groupby_col, agg_col, groupby_scores[row['Name']]
                        )
                    
                    # Update session state
                    st.session_state.all_scores = all_scores
            
            # Display results
            with st.container():
                # ... existing code ...
                
                # Keep the Tab organization
                st.subheader("Visualization Recommendations")
                
                # Only create initial recommendations once when a new file is loaded
                if new_file_upload or 'top_recommendations' not in st.session_state:
                    # Force Longitude & Latitude pair into top recommendations if available
                    longitude_latitude_exists = False
                    longitude_latitude_row = None
                    
                    # Check if Longitude & Latitude pair exists in all_scores
                    if "Longitude" in df.columns and "Latitude" in df.columns:
                        for _, row in all_scores.iterrows():
                            if row['Type'] == 'Pair' and (
                                row['Name'] == "Longitude & Latitude" or
                                (row['Name'].split(" & ")[0] == "Longitude" and row['Name'].split(" & ")[1] == "Latitude") or
                                (row['Name'].split(" & ")[0] == "Latitude" and row['Name'].split(" & ")[1] == "Longitude")
                            ):
                                longitude_latitude_exists = True
                                longitude_latitude_row = row
                                break
                    
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
                    
                    # If Longitude & Latitude exists, add it first to ensure it's included
                    if longitude_latitude_exists and longitude_latitude_row is not None:
                        winners.append(longitude_latitude_row)
                    
                    # Check if we have each type available
                    has_columns = not top_columns.empty
                    has_pairs = not top_pairs.empty
                    has_triples = not top_triples.empty
                    has_groupby = not top_groupby.empty
                    
                    # Always include highest scoring column and pair if available
                    if has_columns:
                        best_column = candidates[candidates['Type'] == 'Column'].iloc[0]
                        # Only add if it's not already in winners
                        if not any(w.equals(best_column) for w in winners):
                            winners.append(best_column)
                        
                    if has_pairs:
                        for _, pair_row in candidates[candidates['Type'] == 'Pair'].iterrows():
                            # Skip if this is the Longitude/Latitude pair and it's already added
                            if longitude_latitude_exists and pair_row['Name'] == longitude_latitude_row['Name']:
                                continue
                            # We only want the top pair that's not Longitude/Latitude if it's already added
                            if not any(w.equals(pair_row) for w in winners):
                                winners.append(pair_row)
                                break
                        
                    # Include one triple if available
                    if has_triples:
                        best_triple = candidates[candidates['Type'] == 'Triple'].iloc[0]
                        # Only add if it's not already in winners
                        if not any(w.equals(best_triple) for w in winners):
                            winners.append(best_triple)
                        
                    # Include one groupby if available
                    if has_groupby:
                        best_groupby = candidates[candidates['Type'] == 'GroupBy'].iloc[0]
                        # Only add if it's not already in winners
                        if not any(w.equals(best_groupby) for w in winners):
                            winners.append(best_groupby)
                        
                    # Fill remaining slots with highest scored items not already included
                    for _, row in candidates.iterrows():
                        # Skip if this row is already in winners
                        if any(w.equals(row) for w in winners):
                            continue
                        winners.append(row)
                        if len(winners) >= 5:
                            break
                        
                    # Convert to DataFrame
                    top_recommendations = pd.DataFrame(winners)
                        
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
                
                # Function to handle retry button clicks
                def retry_recommendation(index, row_type):
                    # Get the next candidate of the same type
                    if row_type == 'Column' and not st.session_state.column_candidates.empty:
                        next_candidate = st.session_state.column_candidates.iloc[0]
                        st.session_state.column_candidates = st.session_state.column_candidates.iloc[1:].copy()
                    elif row_type == 'Pair' and not st.session_state.pair_candidates.empty:
                        next_candidate = st.session_state.pair_candidates.iloc[0]
                        st.session_state.pair_candidates = st.session_state.pair_candidates.iloc[1:].copy()
                    elif row_type == 'Triple' and not st.session_state.triple_candidates.empty:
                        next_candidate = st.session_state.triple_candidates.iloc[0]
                        st.session_state.triple_candidates = st.session_state.triple_candidates.iloc[1:].copy()
                    elif row_type == 'GroupBy' and not st.session_state.groupby_candidates.empty:
                        next_candidate = st.session_state.groupby_candidates.iloc[0]
                        st.session_state.groupby_candidates = st.session_state.groupby_candidates.iloc[1:].copy()
                    else:
                        # No more candidates of this type
                        return
                        
                    # Replace the current recommendation with the next candidate
                    st.session_state.current_recommendations.iloc[index] = next_candidate
                        
                    # Increment the retry counter to trigger UI refresh
                    st.session_state.retry_counter += 1
                    
                # Use current recommendations from session state
                displayed_recommendations = st.session_state.current_recommendations
                    
                # Add index for user reference
                display_df = displayed_recommendations.copy()
                display_df.index = range(1, len(display_df) + 1)  # 1-based indexing for user
                    
                # Create a simplified table for display
                table_data = []
                for idx, row in display_df.iterrows():
                    table_data.append({
                        "#": idx,
                        "Column(s)": row['Name'],
                        "Recommended Visualization": row['Recommended Visualization'],
                        "Type": row['Type']
                    })
                    
                # Convert to DataFrame for styling
                table_df = pd.DataFrame(table_data)
                    
                # Style the table with colors by type
                def style_type(val):
                    color_map = {
                        'Column': 'background-color: #e6f3ff; color: #0066cc;',
                        'Pair': 'background-color: #fff2e6; color: #cc6600;',
                        'Triple': 'background-color: #e6ffe6; color: #006600;',
                        'GroupBy': 'background-color: #f9e6ff; color: #6600cc;'
                    }
                    return color_map.get(val, '')
                    
                # Apply styling to the "Type" column and use custom formatting for the table
                styled_table = table_df.style\
                    .applymap(style_type, subset=['Type'])\
                    .set_properties(**{
                        'border': '1px solid #e1e4e8',
                        'text-align': 'left',
                        'font-size': '14px',
                        'padding': '10px',
                    })\
                    .set_table_styles([
                        {'selector': 'th', 
                         'props': [('background-color', '#f6f8fa'), 
                                  ('color', '#24292e'),
                                  ('font-weight', 'bold'),
                                  ('text-align', 'left'),
                                  ('padding', '10px'),
                                  ('border', '1px solid #e1e4e8')]},
                        {'selector': 'tr:nth-of-type(even)', 
                         'props': [('background-color', '#f6f8fa')]},
                        {'selector': 'tr:hover',
                         'props': [('background-color', '#f0f4f8')]},
                    ])
                    
                # Display the styled table
                st.write(styled_table.to_html(escape=False), unsafe_allow_html=True)
                    
                # Add a container for the entire Browse Alternatives section with proper centering
                st.markdown("""
                <style>
                .browse-alternatives-container {
                    max-width: 700px;
                    margin: 20px auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-radius: 10px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    text-align: center;
                }
                </style>
                <div class="browse-alternatives-container">
                    <h3>Browse Alternatives</h3>
                </div>
                """, unsafe_allow_html=True)
                    
                # Create a centered container for all the browse controls
                container = st.container()
                    
                # Use columns to create a centered layout
                _, center_col, _ = st.columns([1, 10, 1])
                    
                with center_col:
                    # Initialize navigation counters if not already in session state
                    if 'nav_position' not in st.session_state:
                        # For each recommendation: create a counter for its current position
                        st.session_state.nav_position = {}
                        
                    # Dropdown to select which recommendation to modify
                    selected_idx = st.selectbox("Select recommendation #:", 
                                              range(1, len(displayed_recommendations)+1),
                                              format_func=lambda x: f"#{x}: {displayed_recommendations.iloc[x-1]['Name'][:30]}...")
                        
                    # Get the type of selected recommendation (using 0-based index)
                    row_idx = selected_idx - 1
                    row_type = displayed_recommendations.iloc[row_idx]['Type']
                        
                    # Create a unique key for this recommendation
                    nav_key = f"{row_idx}_{row_type}"
                        
                    # Initialize counter for this recommendation if needed
                    if nav_key not in st.session_state.nav_position:
                        st.session_state.nav_position[nav_key] = 0
                        
                    # Show the current recommendation
                    current_item = displayed_recommendations.iloc[row_idx]
                        
                    # Add type-specific styling
                    type_color_map = {
                        'Column': '#0066cc',
                        'Pair': '#cc6600',
                        'Triple': '#006600',
                        'GroupBy': '#6600cc'
                    }
                    type_color = type_color_map.get(row_type, 'black')
                        
                    st.markdown(f"""
                    <div style='margin-top: 10px; margin-bottom: 15px; text-align: center;'>
                        <span style='font-weight: bold;'>Current:</span> 
                        <span>{current_item['Name']}</span><br>
                        <span style='font-weight: bold; color: {type_color};'>
                            {current_item['Type']}
                        </span> | 
                        <span style='font-style: italic;'>
                            {current_item['Recommended Visualization']}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                        
                    # Navigation controls
                    nav_cols = st.columns([3, 2, 2, 3])
                        
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
                        
                    with nav_cols[1]:
                        # Previous button - disabled if at position 0
                        prev_disabled = st.session_state.nav_position[nav_key] <= 0
                        if st.button("◀", disabled=prev_disabled, key=f"prev_{nav_key}"):
                            if st.session_state.nav_position[nav_key] > 0:
                                st.session_state.nav_position[nav_key] -= 1
                                st.rerun()
                        
                    with nav_cols[2]:
                        # Next button - disabled if at end
                        next_disabled = st.session_state.nav_position[nav_key] >= total_alternatives - 1
                        if st.button("▶", disabled=next_disabled, key=f"next_{nav_key}"):
                            if st.session_state.nav_position[nav_key] < total_alternatives - 1:
                                st.session_state.nav_position[nav_key] += 1
                                st.rerun()
                        
                        # Show the current position or inform if no alternatives
                        if total_alternatives == 0:
                            st.info("No alternatives available for this recommendation.")
                        else:
                            # Get the alternative based on current position
                            alt_pos = st.session_state.nav_position[nav_key]
                            alternative = candidates.iloc[alt_pos]
                                
                            # Show the alternative being viewed with styling
                            st.markdown(f"""
                            <div style='margin: 15px auto; padding: 10px; 
                                 background-color: #f0f7ff; border-radius: 5px; 
                                 text-align: center; max-width: 600px;'>
                                <span style='font-weight: bold;'>Alternative {alt_pos+1} of {total_alternatives}:</span><br>
                                <span style='font-size: 16px;'>{alternative['Name']}</span><br>
                                <span style='color: {type_color};'>
                                    {alternative['Type']}
                                </span> | 
                                <span style='font-style: italic;'>
                                    {alternative['Recommended Visualization']}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                                
                            # Center the Use button
                            _, button_col, _ = st.columns([3, 4, 3])
                            with button_col:
                                if st.button("Use this alternative", key=f"use_{nav_key}_{alt_pos}"):
                                    # Replace current recommendation 
                                    st.session_state.current_recommendations.iloc[row_idx] = alternative
                                        
                                    # Don't remove from candidates so user can go back to previous choices
                                    # Just add a success message and keep the current navigation position
                                    st.success("Applied successfully! You can continue browsing alternatives.")
                                        
                                    # Force a rerun to update the table
                                    st.rerun()
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("If you're experiencing performance issues, try using a smaller dataset or selecting fewer columns for analysis.")
else:
    # This is being imported, don't show UI
    pass