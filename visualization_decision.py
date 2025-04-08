import pandas as pd
import numpy as np

def is_temporal_column(df, column_name):
    """
    Enhanced function to detect if a column contains temporal data.
    Specifically handles year columns even when they're numeric integers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the column
    column_name : str
        The name of the column to check
        
    Returns:
    --------
    bool
        Whether the column should be considered temporal
    """
    # First check if it's already a datetime type
    if pd.api.types.is_datetime64_any_dtype(df[column_name]):
        return True
    
    # Check if the column name contains temporal keywords
    temporal_keywords = ['date', 'time', 'year', 'annee', 'month', 'day', 'quarter', 'week', 
                          'hour', 'minute', 'second', 'dt', 'timestamp']
    
    has_temporal_name = any(keyword in column_name.lower() for keyword in temporal_keywords)
    
    # Special handling for year columns
    is_year_column = ('year' in column_name.lower() or 'annee' in column_name.lower())
    
    # For columns with year/annee in the name, check if they contain plausible year values
    if is_year_column and pd.api.types.is_numeric_dtype(df[column_name]):
        # Check if values are in a reasonable year range
        try:
            min_val = df[column_name].min()
            max_val = df[column_name].max()
            # Assume years between 1000 and 2100 are legitimate year values
            if (1000 <= min_val <= 2100) and (1000 <= max_val <= 2100):
                return True
        except:
            pass
    
    # For string columns that might contain dates
    if pd.api.types.is_string_dtype(df[column_name]) and has_temporal_name:
        # Sample values to check for date patterns
        try:
            first_values = df[column_name].dropna().head(5)
            for val in first_values:
                # Check for common date separators
                if isinstance(val, str) and (
                    '/' in val or '-' in val or ':' in val or 
                    any(month in val.lower() for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                                         'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
                ):
                    return True
        except:
            pass
    
    return has_temporal_name

def get_vis_type_for_single_column(df, column_name, scores):
    """Determine best visualization type for a single column."""
    col_data = df[column_name]
    is_numeric = pd.api.types.is_numeric_dtype(col_data)
    is_temporal = pd.api.types.is_datetime64_any_dtype(col_data)
    n_unique = col_data.nunique()
    n_rows = len(col_data)
    
    # Enhanced temporal detection
    is_temporal = is_temporal or is_temporal_column(df, column_name)
    
    # Decision tree for single column
    if is_temporal:
        # Temporal Column Analysis
        if n_unique > 365:  # Many unique dates
            return "Timeline (px.scatter)"
        else:
            return "Calendar Heatmap (px.heatmap)"
    
    if is_numeric:
        # Numerical Column Analysis
        if n_unique < 20:  # Discrete numeric
            return "Bar Chart (px.bar)"
        else:  # Continuous numeric
            if scores.get('distribution_score', 0) > 7:
                return "Histogram (px.histogram)"
            elif scores.get('outlier_score', 0) > 6:
                return "Box Plot (px.box)"
            else:
                return "Histogram (px.histogram)"
    
    # Categorical Column Analysis
    if n_unique <= 7:
        if scores.get('proportion_analysis', 0) > 7:
            return "Pie Chart (px.pie)"
        else:
            return "Bar Chart (px.bar)"
    elif n_unique <= 30:
        return "Treemap (px.treemap)"
    else:
        if n_unique / n_rows > 0.5:  # High cardinality
            return "Table (go.Table)"
        else:
            return "Sunburst (px.sunburst)"

def get_vis_type_for_pair(df, col1, col2, scores):
    """Determine best visualization type for a column pair."""
    data1 = df[col1]
    data2 = df[col2]
    
    # Check for exact matches on Longitude and Latitude columns first
    if (col1 == "Longitude" and col2 == "Latitude") or (col1 == "Latitude" and col2 == "Longitude"):
        return "Choropleth Map (px.choropleth)"
    
    # Check for Longitude,Latitude pairs
    if (('latitude' in col1.lower() or 'lat' in col1.lower()) and 
        ('longitude' in col2.lower() or 'lon' in col2.lower())) or \
       (('latitude' in col2.lower() or 'lat' in col2.lower()) and 
        ('longitude' in col1.lower() or 'lon' in col1.lower())):
        return "Choropleth Map (px.choropleth)"
    
    # Check for longitude/latitude and magnitude pairs
    if ('longitude' in col1.lower() or 'lat' in col1.lower() or 'lon' in col1.lower()) and \
       ('magnitude' in col2.lower() or 'value' in col2.lower() or 'intensity' in col2.lower() or 'count' in col2.lower() or 'quantity' in col2.lower()):
        return "Choropleth Map (px.choropleth)"
    
    if ('longitude' in col2.lower() or 'lat' in col2.lower() or 'lon' in col2.lower()) and \
       ('magnitude' in col1.lower() or 'value' in col1.lower() or 'intensity' in col1.lower() or 'count' in col1.lower() or 'quantity' in col1.lower()):
        return "Choropleth Map (px.choropleth)"
    
    is_numeric1 = pd.api.types.is_numeric_dtype(data1)
    is_numeric2 = pd.api.types.is_numeric_dtype(data2)
    
    # Enhanced temporal detection
    is_temporal1 = pd.api.types.is_datetime64_any_dtype(data1) or is_temporal_column(df, col1)
    is_temporal2 = pd.api.types.is_datetime64_any_dtype(data2) or is_temporal_column(df, col2)
    
    n_unique1 = data1.nunique()
    n_unique2 = data2.nunique()
    
    # Decision tree for pairs
    # Temporal vs Numerical
    if (is_temporal1 and is_numeric2) or (is_temporal2 and is_numeric1):
        temp_col = col1 if is_temporal1 else col2
        num_col = col2 if is_temporal1 else col1
        
        if scores.get('volatility_score', 0) > 7:
            return "Candlestick Chart (px.candlestick)"
        elif scores.get('area_relevant', 0) > 6:
            return "Area Chart (px.area)"
        else:
            return "Line Chart (px.line)"
            
    # Temporal vs Categorical
    if (is_temporal1 and not is_numeric2) or (is_temporal2 and not is_numeric1):
        if n_unique2 <= 5 or n_unique1 <= 5:  # Few categories
            return "Grouped Bar Chart (px.bar)"
        else:
            return "Heatmap (px.heatmap)"
    
    # Both numeric
    if is_numeric1 and is_numeric2:
        # We've already handled lat/long pairs at the top
        if n_unique1 < 20 and n_unique2 < 20:  # Both discrete
            return "Heatmap (px.imshow)"
        elif scores.get('pattern_detection', 0) > 7:
            return "Scatter Plot with Colors (px.scatter)"
        elif scores.get('density_relevant', 0) > 6:
            return "Density Heatmap (px.density_heatmap)"
        else:
            return "Scatter Plot (px.scatter)"
    
    # Both categorical
    if (not is_numeric1 or n_unique1 < 20) and (not is_numeric2 or n_unique2 < 20):
        if n_unique1 * n_unique2 <= 100:  # Not too many combinations
            return "Heatmap (px.imshow)"
        elif n_unique1 <= 7 and n_unique2 <= 7:
            return "Grouped Bar Chart (px.bar)"
        else:
            return "Parallel Categories (px.parallel_categories)"
    
    # One numeric, one categorical
    if (is_numeric1 and not is_numeric2) or (is_numeric2 and not is_numeric1):
        num_col = col1 if is_numeric1 else col2
        cat_col = col2 if is_numeric1 else col1
        cat_unique = n_unique2 if is_numeric1 else n_unique1
        
        if cat_unique <= 10:  # Few categories
            if scores.get('statistical_association', 0) > 7:
                return "Histogram (px.histogram)"
            elif scores.get('show_distribution', 0) > 6:
                return "Box Plot (px.box)"
            elif scores.get('show_individual_points', 0) > 6:
                return "Strip Plot (px.box with points)"
            else:
                return "Bar Chart (px.bar)"
        else:  # Many categories
            return "Scatter Plot Matrix (Splom)"
    
    # Fallback
    return "Scatter Plot (px.scatter)"

def get_vis_type_for_triple(df, col1, col2, col3, scores):
    """Determine best visualization type for three columns."""
    data1 = df[col1]
    data2 = df[col2]
    data3 = df[col3]
    
    is_numeric1 = pd.api.types.is_numeric_dtype(data1)
    is_numeric2 = pd.api.types.is_numeric_dtype(data2)
    is_numeric3 = pd.api.types.is_numeric_dtype(data3)
    
    # Enhanced temporal detection
    is_temporal1 = pd.api.types.is_datetime64_any_dtype(data1) or is_temporal_column(df, col1)
    is_temporal2 = pd.api.types.is_datetime64_any_dtype(data2) or is_temporal_column(df, col2)
    is_temporal3 = pd.api.types.is_datetime64_any_dtype(data3) or is_temporal_column(df, col3)
    
    # Define categorical columns (either non-numeric or low cardinality numeric)
    is_cat1 = not is_numeric1 or (is_numeric1 and data1.nunique() <= 20)
    is_cat2 = not is_numeric2 or (is_numeric2 and data2.nunique() <= 20)
    is_cat3 = not is_numeric3 or (is_numeric3 and data3.nunique() <= 20)
    
    # Count categorical and numerical columns
    num_cat = sum([is_cat1, is_cat2, is_cat3])
    num_num = sum([is_numeric1, is_numeric2, is_numeric3])
    num_temporal = sum([is_temporal1, is_temporal2, is_temporal3])
    
    # All three are numerical
    if num_num == 3:
        return "3D Scatter Plot (px.scatter3d)"
    
    # Two numerical, one categorical
    if num_num == 2 and num_cat == 1:
        cat_col = col1 if is_cat1 else (col2 if is_cat2 else col3)
        if df[cat_col].nunique() <= 10:
            return "Scatter Plot with Color (px.scatter)"
        else:
            return "Faceted Scatter Plots (px.subplots)"
    
    # One numerical, two categorical
    if num_num == 1 and num_cat == 2:
        num_col = col1 if is_numeric1 else (col2 if is_numeric2 else col3)
        cat_cols = [c for i, c in enumerate([col1, col2, col3]) 
                   if [is_cat1, is_cat2, is_cat3][i]]
        
        if all(df[c].nunique() <= 7 for c in cat_cols):
            return "Grouped Bar Chart (px.bar)"
        elif all(df[c].nunique() <= 15 for c in cat_cols):
            return "Heatmap (px.heatmap)"
        else:
            return "Sunburst Chart (px.sunburst)"
    
    # Temporal with numerical and categorical
    if num_temporal >= 1 and num_num >= 1 and num_cat >= 1:
        if scores.get('animation_suitable', 0) > 7:
            return "Animated Scatter (px.scatter with frames)"
        else:
            return "Multi-line Chart (px.line)"
    
    # No good visualization for three categorical columns
    if num_cat == 3:
        if all(df[c].nunique() <= 5 for c in [col1, col2, col3]):
            return "Treemap (px.treemap)"
        else:
            return "Table (go.Table)"
    
    # Fallback
    return "3D Scatter Plot (px.scatter3d)"

def get_vis_type_for_groupby(df, groupby_col, agg_col, agg_func='sum', scores=None):
    """Determine best visualization type for a groupby operation."""
    if scores is None:
        scores = {}
    
    groupby_data = df[groupby_col]
    agg_data = df[agg_col]
    
    is_numeric_agg = pd.api.types.is_numeric_dtype(agg_data)
    
    # Enhanced temporal detection
    is_temporal_groupby = pd.api.types.is_datetime64_any_dtype(groupby_data) or is_temporal_column(df, groupby_col)
    
    # Special handling for year columns
    is_year_column = ('year' in groupby_col.lower() or 'annee' in groupby_col.lower())
    if is_year_column and pd.api.types.is_numeric_dtype(groupby_data):
        # Check if values are in a reasonable year range
        try:
            min_val = groupby_data.min()
            max_val = groupby_data.max()
            # Assume years between 1000 and 2100 are legitimate year values
            if (1000 <= min_val <= 2100) and (1000 <= max_val <= 2100):
                is_temporal_groupby = True
        except:
            pass
    
    n_unique_groupby = groupby_data.nunique()
    
    # Cannot aggregate non-numeric data with sum
    if not is_numeric_agg and agg_func == 'sum':
        agg_func = 'count'
    
    # GroupBy Temporal → Sum(Numerical)
    if is_temporal_groupby and is_numeric_agg:
        if agg_func == 'sum':
            if scores.get('area_relevant', 0) > 6:
                return "Area Chart (px.area)"
            else:
                return "Line Chart (px.line)"
        else:
            return "Bar Chart (px.bar)"
    
    # GroupBy Categorical → Sum(Numerical)
    if not is_temporal_groupby and is_numeric_agg:
        if n_unique_groupby <= 7 and agg_func == 'sum':
            return "Pie Chart (px.pie)"
        elif n_unique_groupby <= 30:
            return "Bar Chart (px.bar)"
        else:
            return "Treemap (px.treemap)"
    
    # GroupBy Categorical → Count()
    if not is_temporal_groupby and agg_func == 'count':
        if n_unique_groupby <= 7:
            return "Pie Chart (px.pie)"
        elif n_unique_groupby <= 30:
            return "Bar Chart (px.bar)"
        else:
            return "Treemap (px.treemap)"
    
    # Fallback
    return "Bar Chart (px.bar)"

def get_vis_type_for_groupby_pair(df, groupby_col1, groupby_col2, agg_col, agg_func='sum', scores=None):
    """Determine best visualization type for a groupby with two grouping columns."""
    if scores is None:
        scores = {}
    
    groupby_data1 = df[groupby_col1]
    groupby_data2 = df[groupby_col2]
    agg_data = df[agg_col]
    
    is_numeric_agg = pd.api.types.is_numeric_dtype(agg_data)
    
    # Enhanced temporal detection
    is_temporal1 = pd.api.types.is_datetime64_any_dtype(groupby_data1) or is_temporal_column(df, groupby_col1)
    is_temporal2 = pd.api.types.is_datetime64_any_dtype(groupby_data2) or is_temporal_column(df, groupby_col2)
    
    n_unique1 = groupby_data1.nunique()
    n_unique2 = groupby_data2.nunique()
    
    # Cannot aggregate non-numeric data with sum
    if not is_numeric_agg and agg_func == 'sum':
        agg_func = 'count'
    
    # GroupBy Temporal + Categorical → Sum(Numerical)
    if (is_temporal1 or is_temporal2) and is_numeric_agg:
        temporal_col = groupby_col1 if is_temporal1 else groupby_col2
        cat_col = groupby_col2 if is_temporal1 else groupby_col1
        cat_unique = n_unique2 if is_temporal1 else n_unique1
        
        if cat_unique <= 5:
            return "Multi-line Chart (px.line)"
        elif cat_unique <= 10:
            return "Stacked Area Chart (px.area)"
        else:
            return "Stacked Bar Chart (px.bar)"
    
    # GroupBy Categorical + Categorical → Sum(Numerical)
    if not is_temporal1 and not is_temporal2:
        if n_unique1 <= 10 and n_unique2 <= 10:
            return "Heatmap (px.imshow)"
        elif n_unique1 <= 7 and n_unique2 <= 7:
            return "Grouped Bar Chart (px.bar)"
        elif n_unique1 * n_unique2 <= 100:
            return "Stacked Bar Chart (px.bar)"
        else:
            return "Sunburst Chart (px.sunburst)"
    
    # Fallback
    return "Grouped Bar Chart (px.bar)"

def score_all_columns_and_pairs(df):
    """
    Score all columns and pairs of columns to determine which are most valuable to visualize.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to analyze
        
    Returns:
    --------
    dict
        A dictionary containing scores for single columns, column pairs, and recommendations
    """
    # Prioritize Longitude/Latitude pairs first
    if "Longitude" in df.columns and "Latitude" in df.columns:
        # Give an extremely high score to this specific pair
        geo_pair_score = 9999  # Very high score to ensure this pair is prioritized
        return {
            'pair_scores': {
                ('Longitude', 'Latitude'): geo_pair_score
            },
            'single_scores': {},
            'pair_recommendations': {
                ('Longitude', 'Latitude'): "Choropleth Map (px.choropleth)"
            },
            'single_recommendations': {}
        }
    
    # If no Longitude/Latitude, then continue with regular scoring...
    # ... existing code ...
