import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA
from scipy.stats import entropy

class VisualizationScorer:
    def __init__(self, df):
        """Initialize with a dataframe to analyze."""
        self.df = df
        self.column_types = self._determine_column_types()
        self.column_scores = {}
        self.pair_scores = {}
        self.triple_scores = {}
        self.groupby_scores = {}
    
    def _determine_column_types(self):
        """Determine the type of each column in the dataframe."""
        column_types = {}
        
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Check if it's more like categorical
                if len(self.df[col].unique()) <= 20:
                    column_types[col] = 'numeric_categorical'
                else:
                    column_types[col] = 'numeric_continuous'
            elif pd.api.types.is_datetime64_dtype(self.df[col]) or self._is_date_like(col):
                column_types[col] = 'temporal'
            else:
                # Check cardinality for categoricals
                unique_ratio = len(self.df[col].unique()) / len(self.df)
                if unique_ratio > 0.9:  # Likely an ID or a free text field
                    column_types[col] = 'id'
                else:
                    column_types[col] = 'categorical'
        
        return column_types
    
    def _is_date_like(self, column_name):
        """Check if a column has date-like strings."""
        # Simplified implementation
        date_terms = ['date', 'time', 'year', 'month', 'day']
        return any(term in column_name.lower() for term in date_terms)
    
    def _is_id_column(self, column_name):
        """Check if a column is likely an ID column."""
        id_terms = ['id', 'key', 'uuid', 'code']
        if any(term in column_name.lower() for term in id_terms):
            return True
        
        # Check if column has unique values equal to row count
        if self.column_types.get(column_name) == 'id':
            return True
        
        return False
    
    def _calculate_distribution_score(self, column):
        """Calculate distribution characteristics score (0-10)."""
        col_type = self.column_types.get(column)
        
        if col_type in ['numeric_continuous', 'numeric_categorical']:
            data = self.df[column].dropna()
            
            # Calculate basic stats
            if len(data) < 3:  # Not enough data
                return 5
            
            cv = data.std() / data.mean() if data.mean() != 0 else 0
            skewness = stats.skew(data)
            try:
                kurtosis = stats.kurtosis(data)
            except:
                kurtosis = 0
                
            # Calculate scores
            cv_score = min(10, cv * 5)
            skew_score = min(10, abs(skewness) * 2)
            kurtosis_score = min(10, abs(kurtosis - 3) * 1.5)
            
            # Simplified multimodality detection
            multimodality_score = 5  # Default medium score
            
            # Average the scores
            return (cv_score + skew_score + kurtosis_score + multimodality_score) / 4
        
        elif col_type == 'categorical':
            # Calculate entropy
            value_counts = self.df[column].value_counts(normalize=True)
            max_entropy = np.log(len(value_counts))
            if max_entropy == 0:
                return 5
            actual_entropy = stats.entropy(value_counts)
            return (actual_entropy / max_entropy) * 10
        
        elif col_type == 'temporal':
            # Simplified time range scoring
            return 7  # Default good score for temporal
        
        return 5  # Default medium score
    
    def _calculate_data_type_score(self, column):
        """Calculate data type score (0-10)."""
        col_type = self.column_types.get(column)
        
        if col_type == 'numeric_continuous':
            data = self.df[column].dropna()
            if len(data) < 2:
                return 5
            
            # Check range span
            try:
                min_val, max_val = data.min(), data.max()
                if max_val / min_val > 1000:  # Spans 3 orders of magnitude
                    return 10
                return 8
            except:
                return 8
        
        elif col_type in ['categorical', 'numeric_categorical']:
            cardinality = len(self.df[column].unique())
            # Score peaks at around 10 categories
            return 10 * (1 - abs(np.log10(cardinality/10)/2))
        
        elif col_type == 'temporal':
            return 8  # Default good score for temporal
        
        elif col_type == 'id':
            return 2  # Low score for IDs
        
        return 5  # Default medium score
    
    def _calculate_data_quality_score(self, column):
        """Calculate data quality score (0-10)."""
        # Calculate completeness
        missing_ratio = self.df[column].isna().mean()
        completeness_score = (1 - missing_ratio) * 10
        
        # Simplified outlier detection for numeric columns
        outlier_score = 10
        if self.column_types.get(column) in ['numeric_continuous', 'numeric_categorical']:
            data = self.df[column].dropna()
            if len(data) >= 5:
                q1, q3 = data.quantile(0.25), data.quantile(0.75)
                iqr = q3 - q1
                outlier_bounds = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
                outlier_ratio = ((data < outlier_bounds[0]) | (data > outlier_bounds[1])).mean()
                outlier_score = (1 - outlier_ratio) * 10
        
        # Unique value ratio
        unique_ratio = len(self.df[column].unique()) / len(self.df)
        if self.column_types.get(column) in ['numeric_continuous', 'numeric_categorical']:
            unique_score = min(10, (unique_ratio) * 20)
        else:
            unique_score = min(10, (unique_ratio) * 100)
        
        # Average the scores
        return (completeness_score + outlier_score + unique_score) / 3
    
    def _calculate_semantic_content_score(self, column):
        """Calculate semantic content analysis score (0-10)."""
        col_name = column.lower()
        
        # Start with a default score
        score = 5
        
        # ID detection
        if self._is_id_column(column):
            return 0  # Low score for IDs
        
        # Metric detection
        metric_terms = ['total', 'count', 'sum', 'amount', 'price', 'cost', 'revenue']
        if any(term in col_name for term in metric_terms):
            score += 2
        
        # Descriptor detection
        descriptor_terms = ['name', 'description', 'title', 'category', 'type']
        if any(term in col_name for term in descriptor_terms):
            score += 2
        
        return min(10, score)
    
    def _calculate_column_scores(self):
        """Calculate individual column scores."""
        for column in self.df.columns:
            # Calculate individual scores
            distribution_score = self._calculate_distribution_score(column)
            data_type_score = self._calculate_data_type_score(column)
            data_quality_score = self._calculate_data_quality_score(column)
            semantic_score = self._calculate_semantic_content_score(column)
            
            # Simplified predictive power (correlation with other columns)
            predictive_power = 5  # Default medium score
            
            # Simplified dimensional analysis
            dimensional_score = 7 if not self._is_id_column(column) else 2
            
            # Simplified variance information ratio
            variance_info_ratio = 6  # Default medium-high score
            
            # Calculate weighted score
            weighted_score = (
                0.20 * distribution_score + 
                0.15 * data_type_score + 
                0.10 * data_quality_score + 
                0.15 * predictive_power +
                0.20 * semantic_score +
                0.10 * dimensional_score +
                0.10 * variance_info_ratio
            )
            
            self.column_scores[column] = {
                'total_score': weighted_score,
                'distribution_score': distribution_score,
                'data_type_score': data_type_score,
                'data_quality_score': data_quality_score,
                'predictive_power_score': predictive_power,
                'semantic_content_score': semantic_score,
                'dimensional_analysis_score': dimensional_score,
                'variance_info_ratio_score': variance_info_ratio
            }
    
    def _calculate_pair_scores(self):
        """Calculate scores for pairs of columns."""
        columns = self.df.columns
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                # Skip if both are IDs
                if self._is_id_column(col1) and self._is_id_column(col2):
                    continue
                
                pair_key = (col1, col2)
                
                # Calculate association score based on column types
                association_score = self._calculate_association_score(col1, col2)
                
                # Simplified visualization complexity
                viz_complexity = 7  # Default good score
                
                # Simplified pattern detection
                pattern_score = 6  # Default medium-high score
                
                # Simplified anomaly highlighting
                anomaly_score = 5  # Default medium score
                
                # Simplified information complementarity
                info_complementarity = 6
                
                # Simplified redundancy penalization (start with perfect score)
                redundancy_score = 10
                if association_score > 9:  # Highly correlated
                    redundancy_score = 3
                elif association_score > 7:  # Moderately correlated
                    redundancy_score = 6
                
                # Practical utility score
                utility_score = self._calculate_practical_utility(col1, col2)
                
                # Calculate weighted score
                weighted_score = (
                    0.15 * association_score + 
                    0.10 * viz_complexity + 
                    0.15 * pattern_score + 
                    0.05 * anomaly_score +
                    0.15 * info_complementarity +
                    0.10 * redundancy_score +
                    0.30 * utility_score
                )
                
                # Apply penalties
                if self._is_id_column(col1) or self._is_id_column(col2):
                    weighted_score *= 0.6  # Reduce by 40%
                
                self.pair_scores[pair_key] = {
                    'total_score': weighted_score,
                    'statistical_association': association_score,
                    'visualization_complexity': viz_complexity,
                    'pattern_detection': pattern_score,
                    'anomaly_highlighting': anomaly_score,
                    'information_complementarity': info_complementarity,
                    'redundancy_penalization': redundancy_score,
                    'practical_utility_score': utility_score
                }
    
    def _calculate_association_score(self, col1, col2):
        """Calculate statistical association between two columns."""
        type1, type2 = self.column_types.get(col1), self.column_types.get(col2)
        
        # Numeric vs Numeric
        if type1 in ['numeric_continuous', 'numeric_categorical'] and type2 in ['numeric_continuous', 'numeric_categorical']:
            try:
                correlation = self.df[col1].corr(self.df[col2])
                return abs(correlation) * 10
            except:
                return 5
        
        # Categorical vs Categorical
        elif type1 in ['categorical', 'numeric_categorical'] and type2 in ['categorical', 'numeric_categorical']:
            # Simplified Cramer's V calculation
            try:
                contingency = pd.crosstab(self.df[col1], self.df[col2])
                chi2 = stats.chi2_contingency(contingency)[0]
                n = contingency.sum().sum()
                phi2 = chi2 / n
                r, k = contingency.shape
                phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
                rcorr = r - ((r-1)**2)/(n-1)
                kcorr = k - ((k-1)**2)/(n-1)
                cramers_v = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
                return cramers_v * 10
            except:
                return 5
        
        # Numeric vs Categorical
        elif (type1 in ['numeric_continuous', 'numeric_categorical'] and type2 in ['categorical']) or \
             (type2 in ['numeric_continuous', 'numeric_categorical'] and type1 in ['categorical']):
            # Simplified Eta squared approximation
            num_col = col1 if type1 in ['numeric_continuous', 'numeric_categorical'] else col2
            cat_col = col2 if type1 in ['numeric_continuous', 'numeric_categorical'] else col1
            
            try:
                categories = self.df[cat_col].unique()
                total_variance = self.df[num_col].var()
                if total_variance == 0:
                    return 5
                    
                between_variance = 0
                for category in categories:
                    subset = self.df[self.df[cat_col] == category][num_col]
                    between_variance += len(subset) * (subset.mean() - self.df[num_col].mean())**2
                
                between_variance /= len(self.df)
                eta_squared = between_variance / total_variance
                return eta_squared * 10
            except:
                return 5
        
        # Temporal vs Numeric
        elif (type1 == 'temporal' and type2 in ['numeric_continuous', 'numeric_categorical']) or \
             (type2 == 'temporal' and type1 in ['numeric_continuous', 'numeric_categorical']):
            # Simplified trend strength
            return 7  # Default good score for time series
        
        # Default for other combinations
        return 5
    
    def _calculate_practical_utility(self, col1, col2):
        """Calculate practical utility score for visualization."""
        type1, type2 = self.column_types.get(col1), self.column_types.get(col2)
        
        # Start with a perfect score
        score = 10
        
        # Low-value column penalties
        if self._is_id_column(col1) and self._is_id_column(col2):
            return 0
        elif self._is_id_column(col1) or self._is_id_column(col2):
            score = 3
        
        # Special cases
        if (type1 == 'temporal' and type2 in ['categorical']) or (type2 == 'temporal' and type1 in ['categorical']):
            cat_col = col1 if type1 in ['categorical'] else col2
            if len(self.df[cat_col].unique()) <= 10:
                score = 6
        
        if (type1 == 'temporal' and type2 in ['numeric_continuous', 'numeric_categorical']) or \
           (type2 == 'temporal' and type1 in ['numeric_continuous', 'numeric_categorical']):
            score = 7
        
        # High-value combinations
        if (type1 in ['categorical'] and type2 in ['numeric_continuous', 'numeric_categorical']) or \
           (type2 in ['categorical'] and type1 in ['numeric_continuous', 'numeric_categorical']):
            score = 10  # Perfect for boxplots, bar charts
        
        if type1 in ['categorical'] and type2 in ['categorical']:
            cat1 = col1
            cat2 = col2
            if len(self.df[cat1].unique()) <= 15 and len(self.df[cat2].unique()) <= 15:
                score = 9  # Good for heatmaps
        
        if type1 in ['numeric_continuous', 'numeric_categorical'] and type2 in ['numeric_continuous', 'numeric_categorical']:
            score = 8  # Good for scatter plots
            try:
                correlation = abs(self.df[col1].corr(self.df[col2]))
                if correlation > 0.6:
                    score = 10
            except:
                pass
        
        return score
    
    def _calculate_triple_scores(self):
        """Calculate scores for triples of columns."""
        # This is simplified as triple analysis is complex
        columns = self.df.columns
        if len(columns) < 3:
            return
        
        # Only get a few combinations to avoid explosion
        top_pairs = sorted(self.pair_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)[:5]
        top_pair_cols = set()
        for pair, _ in top_pairs:
            top_pair_cols.update(pair)
        
        candidate_cols = list(top_pair_cols)
        
        for i, col1 in enumerate(candidate_cols):
            for j, col2 in enumerate(candidate_cols[i+1:], i+1):
                for k, col3 in enumerate(candidate_cols[j+1:], j+1):
                    triple_key = (col1, col2, col3)
                    
                    # Simplified triple scoring
                    # Dimensionality appropriateness
                    all_numeric = all(self.column_types.get(col) in ['numeric_continuous', 'numeric_categorical'] 
                                     for col in triple_key)
                    dim_score = a_score = 7 if all_numeric else 4
                    
                    # Pattern richness
                    pattern_score = 6  # Default medium-high score
                    
                    # Complexity balance
                    complexity_score = 5  # Default medium score
                    
                    # Visual differentiability
                    visual_score = 6  # Default medium-high score
                    
                    # Cognitive load (start with perfect score)
                    cognitive_score = 8
                    
                    # Average the scores
                    total_score = (dim_score + pattern_score + complexity_score + 
                                  visual_score + cognitive_score) / 5
                    
                    # Apply penalties
                    id_count = sum(1 for col in triple_key if self._is_id_column(col))
                    if id_count > 0:
                        total_score *= (1 - id_count * 0.3)
                    
                    self.triple_scores[triple_key] = {
                        'total_score': total_score,
                        'dimensionality_appropriateness': dim_score,
                        'pattern_richness': pattern_score,
                        'complexity_balance': complexity_score,
                        'visual_differentiability': visual_score,
                        'cognitive_load': cognitive_score
                    }
    
    def _calculate_groupby_scores(self):
        """Calculate scores for groupby operations."""
        for potential_group_col in self.df.columns:
            # Skip if column is not suitable for grouping
            if self._is_id_column(potential_group_col) or self.df[potential_group_col].nunique() > 100:
                continue
                
            group_col_score = self._calculate_groupby_column_score(potential_group_col)
            
            for potential_agg_col in self.df.columns:
                # Skip if same column or aggregation col is not numeric
                if potential_group_col == potential_agg_col or not pd.api.types.is_numeric_dtype(self.df[potential_agg_col]):
                    continue
                
                agg_col_score = self._calculate_aggregation_column_score(potential_agg_col)
                
                # If both columns have reasonable scores, calculate the pair score
                if group_col_score > 3 and agg_col_score > 3:
                    pair_key = (potential_group_col, potential_agg_col)
                    
                    # Calculate group differentiation
                    differentiation_score = self._calculate_group_differentiation(potential_group_col, potential_agg_col)
                    
                    # Group balance
                    balance_score = self._calculate_group_balance(potential_group_col)
                    
                    # Aggregation meaningfulness
                    agg_meaning_score = 7  # Default good score
                    
                    # Visualization potential
                    viz_potential = self._calculate_groupby_viz_potential(potential_group_col, potential_agg_col)
                    
                    # Calculate weighted score
                    weighted_score = (
                        0.40 * differentiation_score +
                        0.15 * balance_score +
                        0.25 * agg_meaning_score +
                        0.20 * viz_potential
                    )
                    
                    # Apply penalties
                    if self._is_id_column(potential_group_col):
                        weighted_score *= 0.5  # Reduce by 50%
                    
                    self.groupby_scores[pair_key] = {
                        'total_score': weighted_score,
                        'group_differentiation': differentiation_score,
                        'group_balance': balance_score,
                        'aggregation_meaningfulness': agg_meaning_score,
                        'visualization_potential': viz_potential
                    }
    
    def _calculate_groupby_column_score(self, column):
        """Calculate how suitable a column is for grouping."""
        # Cardinality appropriateness
        cardinality = len(self.df[column].unique())
        cardinality_score = 10 * (1 - abs(np.log10(cardinality/15)/3))
        
        # Group distribution
        value_counts = self.df[column].value_counts(normalize=True)
        max_entropy = np.log(len(value_counts))
        if max_entropy == 0:
            normalized_entropy = 0.5
        else:
            actual_entropy = stats.entropy(value_counts)
            normalized_entropy = actual_entropy / max_entropy
        
        balance_score = min(10, normalized_entropy * 10)
        
        # Data type score
        col_type = self.column_types.get(column)
        if col_type == 'categorical':
            type_score = 8
        elif col_type == 'temporal':
            type_score = 9
        elif col_type in ['numeric_continuous', 'numeric_categorical']:
            type_score = 7 if cardinality <= 20 else 2
        else:
            type_score = 3
        
        # Column name bonus
        grouping_terms = ['category', 'region', 'status', 'type', 'group']
        name_bonus = 1 if any(term in column.lower() for term in grouping_terms) else 0
        
        # ID column penalty
        if self._is_id_column(column):
            return 0
        
        # Average the scores
        return (cardinality_score + balance_score + type_score + name_bonus) / 4
    
    def _calculate_aggregation_column_score(self, column):
        """Calculate how suitable a column is for aggregation."""
        # Base score for numeric
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return 0
        
        score = 7
        
        # Bonus for non-binary numeric
        unique_vals = self.df[column].unique()
        if len(unique_vals) > 2:
            score += 2
        
        # Bonus for metric-like column names
        metric_terms = ['amount', 'sales', 'count', 'price', 'cost', 'revenue', 'profit']
        if any(term in column.lower() for term in metric_terms):
            score += 1
        
        # Penalty for ID-like columns
        if self._is_id_column(column):
            return 0
        
        return min(10, score)
    
    def _calculate_group_differentiation(self, group_col, agg_col):
        """Calculate how well groups differentiate the aggregation column."""
        try:
            # Simplified eta squared calculation
            categories = self.df[group_col].unique()
            total_variance = self.df[agg_col].var()
            if total_variance == 0:
                return 5
                
            between_variance = 0
            for category in categories:
                subset = self.df[self.df[group_col] == category][agg_col]
                if len(subset) > 0:
                    between_variance += len(subset) * (subset.mean() - self.df[agg_col].mean())**2
            
            between_variance /= len(self.df)
            eta_squared = between_variance / total_variance
            return min(10, eta_squared * 10)
        except:
            return 5
    
    def _calculate_group_balance(self, group_col):
        """Calculate how balanced the groups are."""
        value_counts = self.df[group_col].value_counts(normalize=True)
        max_entropy = np.log(len(value_counts))
        if max_entropy == 0:
            return 5
        actual_entropy = stats.entropy(value_counts)
        normalized_entropy = actual_entropy / max_entropy
        return normalized_entropy * 10
    
    def _calculate_groupby_viz_potential(self, group_col, agg_col):
        """Calculate visualization potential for a groupby pair."""
        score = 5  # Default medium score
        
        col_type = self.column_types.get(group_col)
        
        # Temporal groupby with numeric aggregation
        if col_type == 'temporal' and pd.api.types.is_numeric_dtype(self.df[agg_col]):
            score += 2
        
        # Categorical groupby with < 20 categories and numeric aggregation
        if col_type in ['categorical', 'numeric_categorical'] and \
           len(self.df[group_col].unique()) < 20 and \
           pd.api.types.is_numeric_dtype(self.df[agg_col]):
            score += 2
        
        return min(10, score)
    
    def score_all(self):
        """Calculate scores for all visualization possibilities."""
        self._calculate_column_scores()
        self._calculate_pair_scores()
        self._calculate_triple_scores()
        self._calculate_groupby_scores()
        
        return self.get_top_recommendations()
    
    def get_top_recommendations(self, n=5):
        """Get top N visualization recommendations."""
        all_candidates = []
        
        # Get top individual columns
        for col, scores in sorted(self.column_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)[:2]:
            all_candidates.append({
                'type': 'individual',
                'columns': [col],
                'score': scores['total_score'],
                'visualization': self._suggest_viz_for_individual(col)
            })
        
        # Get top column pairs
        for pair, scores in sorted(self.pair_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)[:2]:
            all_candidates.append({
                'type': 'pair',
                'columns': list(pair),
                'score': scores['total_score'],
                'visualization': self._suggest_viz_for_pair(*pair)
            })
        
        # Get top triple
        if self.triple_scores:
            top_triple = sorted(self.triple_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)[0]
            triple, scores = top_triple
            all_candidates.append({
                'type': 'triple',
                'columns': list(triple),
                'score': scores['total_score'],
                'visualization': self._suggest_viz_for_triple(*triple)
            })
        
        # Get top groupby
        if self.groupby_scores:
            top_groupby = sorted(self.groupby_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)[0]
            group_pair, scores = top_groupby
            all_candidates.append({
                'type': 'groupby',
                'columns': list(group_pair),
                'score': scores['total_score'],
                'visualization': self._suggest_viz_for_groupby(*group_pair)
            })
        
        # Sort by score and take top N
        top_recommendations = sorted(all_candidates, key=lambda x: x['score'], reverse=True)[:n]
        
        # Ensure at least one of each type if available and score is high enough
        recommendation_types = {rec['type'] for rec in top_recommendations}
        
        # If no individual column in top recommendations, add the best one
        if 'individual' not in recommendation_types and self.column_scores:
            best_individual = sorted(self.column_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)[0]
            col, scores = best_individual
            top_recommendations.append({
                'type': 'individual',
                'columns': [col],
                'score': scores['total_score'],
                'visualization': self._suggest_viz_for_individual(col)
            })
        
        # If no pair in top recommendations, add the best one
        if 'pair' not in recommendation_types and self.pair_scores:
            best_pair = sorted(self.pair_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)[0]
            pair, scores = best_pair
            top_recommendations.append({
                'type': 'pair',
                'columns': list(pair),
                'score': scores['total_score'],
                'visualization': self._suggest_viz_for_pair(*pair)
            })
        
        # Sort again and limit to N
        return sorted(top_recommendations, key=lambda x: x['score'], reverse=True)[:n]
    
    def _suggest_viz_for_individual(self, column):
        """Suggest visualization type for an individual column."""
        col_type = self.column_types.get(column)
        
        if col_type in ['numeric_continuous', 'numeric_categorical']:
            return 'histogram'
        elif col_type == 'categorical':
            return 'bar_chart'
        elif col_type == 'temporal':
            return 'time_series'
        else:
            return 'bar_chart'  # Default
    
    def _suggest_viz_for_pair(self, col1, col2):
        """Suggest visualization type for a pair of columns."""
        type1, type2 = self.column_types.get(col1), self.column_types.get(col2)
        
        # Numeric vs Numeric
        if type1 in ['numeric_continuous', 'numeric_categorical'] and type2 in ['numeric_continuous', 'numeric_categorical']:
            return 'scatter_plot'
        
        # Categorical vs Categorical
        elif type1 in ['categorical', 'numeric_categorical'] and type2 in ['categorical', 'numeric_categorical']:
            return 'heatmap'
        
        # Numeric vs Categorical
        elif (type1 in ['numeric_continuous', 'numeric_categorical'] and type2 in ['categorical']) or \
             (type2 in ['numeric_continuous', 'numeric_categorical'] and type1 in ['categorical']):
            return 'box_plot'
        
        # Temporal vs Numeric
        elif (type1 == 'temporal' and type2 in ['numeric_continuous', 'numeric_categorical']) or \
             (type2 == 'temporal' and type1 in ['numeric_continuous', 'numeric_categorical']):
            return 'time_series'
        
        # Temporal vs Categorical
        elif (type1 == 'temporal' and type2 in ['categorical']) or \
             (type2 == 'temporal' and type1 in ['categorical']):
            return 'line_chart'
        
        # Default
        return 'scatter_plot'
    
    def _suggest_viz_for_triple(self, col1, col2, col3):
        """Suggest visualization type for a triple of columns."""
        types = [self.column_types.get(col) for col in [col1, col2, col3]]
        
        # All numeric
        if all(t in ['numeric_continuous', 'numeric_categorical'] for t in types):
            return '3d_scatter'
        
        # Two numeric + one categorical
        if sum(1 for t in types if t in ['numeric_continuous', 'numeric_categorical']) == 2 and \
           sum(1 for t in types if t in ['categorical']) == 1:
            return 'bubble_chart'
        
        # One numeric + two categorical
        if sum(1 for t in types if t in ['numeric_continuous', 'numeric_categorical']) == 1 and \
           sum(1 for t in types if t in ['categorical']) == 2:
            return 'grouped_bar_chart'
        
        # Default
        return 'complex_chart'
    
    def _suggest_viz_for_groupby(self, group_col, agg_col):
        """Suggest visualization type for a groupby operation."""
        group_type = self.column_types.get(group_col)
        
        if group_type == 'temporal':
            return 'time_series'
        elif group_type in ['categorical', 'numeric_categorical']:
            cardinality = len(self.df[group_col].unique())
            if cardinality <= 10:
                return 'bar_chart'
            else:
                return 'treemap'
        else:
            return 'bar_chart'  # Default 