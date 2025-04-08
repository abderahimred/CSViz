import streamlit as st

"""
Translations module for the data cleaning app.
"""

# Dictionary of translations
# Structure: {'text_id': {'en': 'English text', 'fr': 'French text'}}
translations = {
    # Common UI elements
    'language_selector': {
        'en': 'Language',
        'fr': 'Langue'
    },
    
    # AI Processing bullet points
    'intelligently_analyze': {
        'en': 'Intelligently analyze and combine multiple datasets',
        'fr': 'Analyser et combiner intelligemment plusieurs ensembles de données'
    },
    'identify_common_fields': {
        'en': 'Identify common fields and optimal joining strategy',
        'fr': 'Identifier les champs communs et la stratégie de jointure optimale'
    },
    'handle_column_mapping': {
        'en': 'Handle column mapping and type conversion automatically',
        'fr': 'Gérer automatiquement le mappage des colonnes et la conversion des types'
    },
    'clean_standardize': {
        'en': 'Clean and standardize data during processing',
        'fr': 'Nettoyer et standardiser les données pendant le traitement'
    },
    
    'optimize_data_structure': {
        'en': 'Optimize data structure for better performance',
        'fr': 'Optimiser la structure des données pour de meilleures performances'
    },
    
    # AI Cleaning bullet points
    'detect_fix_issues': {
        'en': 'Automatically detect and fix data quality issues',
        'fr': 'Détecter et corriger automatiquement les problèmes de qualité des données'
    },
    'convert_data_types': {
        'en': 'Convert incorrect data types to appropriate formats',
        'fr': 'Convertir les types de données incorrects aux formats appropriés'
    },
    'fill_missing_values': {
        'en': 'Fill missing values with intelligent predictions',
        'fr': 'Remplir les valeurs manquantes avec des prédictions intelligentes'
    },
    'remove_outliers': {
        'en': 'Remove outliers and standardize formatting',
        'fr': 'Supprimer les valeurs aberrantes et standardiser le formatage'
    },
    
    # Navigation and workflow
    'upload': {
        'en': 'Upload',
        'fr': 'Télécharger'
    },
    'process': {
        'en': 'Process',
        'fr': 'Traiter'
    },
    'clean': {
        'en': 'Clean',
        'fr': 'Nettoyer'
    },
    'visualize': {
        'en': 'Visualize',
        'fr': 'Visualiser'
    },
    'back_to_upload': {
        'en': '← Back to Upload',
        'fr': '← Retour au téléchargement'
    },
    'back_to_home': {
        'en': '← Back to Home',
        'fr': '← Retour à l\'accueil'
    },
    
    # Home page
    'app_title': {
        'en': 'CSV Processing Tool',
        'fr': 'Outil de traitement CSV'
    },
    'upload_csv_files': {
        'en': 'Upload CSV Files',
        'fr': 'Télécharger des fichiers CSV'
    },
    'drop_files_here': {
        'en': 'Drop your CSV files here or click to browse',
        'fr': 'Déposez vos fichiers CSV ici ou cliquez pour parcourir'
    },
    'uploaded_files': {
        'en': 'Uploaded Files',
        'fr': 'Fichiers téléchargés'
    },
    'filename': {
        'en': 'Filename',
        'fr': 'Nom du fichier'
    },
    'size': {
        'en': 'Size',
        'fr': 'Taille'
    },
    'rows_columns': {
        'en': 'Rows × Columns',
        'fr': 'Lignes × Colonnes'
    },
    'action': {
        'en': 'Action',
        'fr': 'Action'
    },
    'next_steps': {
        'en': 'Next Steps',
        'fr': 'Prochaines étapes'
    },
    'process_multiple_files': {
        'en': '🔗 Process Multiple Files',
        'fr': '🔗 Traiter plusieurs fichiers'
    },
    'clean_data': {
        'en': '🧹 Clean Data',
        'fr': '🧹 Nettoyer les données'
    },
    'upload_instructions': {
        'en': '👆 Please upload one or more CSV files to get started',
        'fr': '👆 Veuillez télécharger un ou plusieurs fichiers CSV pour commencer'
    },
    
    # Process page
    'choose_processing_method': {
        'en': 'Choose Processing Method',
        'fr': 'Choisir la méthode de traitement'
    },
    'ai_processing': {
        'en': '🤖 AI Processing',
        'fr': '🤖 Traitement par IA'
    },
    'manual_processing': {
        'en': '🛠️ Manual Processing',
        'fr': '🛠️ Traitement manuel'
    },
    'let_ai_analyze': {
        'en': 'Let our AI analyze and combine your datasets',
        'fr': 'Laissez notre IA analyser et combiner vos jeux de données'
    },
    'process_data_using_tools': {
        'en': 'Process your data using interactive tools',
        'fr': 'Traitez vos données à l\'aide d\'outils interactifs'
    },
    'openai_api_key': {
        'en': 'OpenAI API Key (required)',
        'fr': 'Clé API OpenAI (obligatoire)'
    },
    'process_with_ai': {
        'en': '🤖 Process with AI',
        'fr': '🤖 Traiter avec l\'IA'
    },
    'enter_api_key': {
        'en': 'Enter OpenAI API key above to enable AI processing',
        'fr': 'Entrez la clé API OpenAI ci-dessus pour activer le traitement par IA'
    },
    'vertical_concatenation': {
        'en': '• Vertical concatenation (stack files)',
        'fr': '• Concaténation verticale (empiler les fichiers)'
    },
    'horizontal_concatenation': {
        'en': '• Horizontal concatenation (add columns)',
        'fr': '• Concaténation horizontale (ajouter des colonnes)'
    },
    'join_merge': {
        'en': '• Join/merge on key columns',
        'fr': '• Joindre/fusionner sur des colonnes clés'
    },
    'column_mapping': {
        'en': '• Column mapping and unification',
        'fr': '• Mappage et unification des colonnes'
    },
    'process_manually': {
        'en': '🛠️ Process Manually',
        'fr': '🛠️ Traiter manuellement'
    },
    
    # Clean page
    'choose_cleaning_method': {
        'en': 'Choose Cleaning Method',
        'fr': 'Choisir la méthode de nettoyage'
    },
    'ai_cleaning': {
        'en': '🤖 AI Cleaning',
        'fr': '🤖 Nettoyage par IA'
    },
    'manual_cleaning': {
        'en': '🛠️ Manual Cleaning',
        'fr': '🛠️ Nettoyage manuel'
    },
    'let_ai_clean': {
        'en': 'Let AI analyze and clean your data automatically',
        'fr': 'Laissez l\'IA analyser et nettoyer vos données automatiquement'
    },
    'clean_with_ai': {
        'en': '🤖 Clean with AI',
        'fr': '🤖 Nettoyer avec l\'IA'
    },
    'clean_data_using_tools': {
        'en': 'Clean your data using interactive tools',
        'fr': 'Nettoyez vos données à l\'aide d\'outils interactifs'
    },
    'text_cleaning': {
        'en': '• Text cleaning & formatting',
        'fr': '• Nettoyage et formatage de texte'
    },
    'handle_missing_values': {
        'en': '• Handle missing values & outliers',
        'fr': '• Gérer les valeurs manquantes et aberrantes'
    },
    'remove_duplicates': {
        'en': '• Remove duplicates',
        'fr': '• Supprimer les doublons'
    },
    'column_operations': {
        'en': '• Column operations (rename, drop, split)',
        'fr': '• Opérations sur les colonnes (renommer, supprimer, diviser)'
    },
    'clean_manually': {
        'en': '🛠️ Clean Manually',
        'fr': '🛠️ Nettoyer manuellement'
    },
    'select_different_cleaning_method': {
        'en': '← Select Different Cleaning Method',
        'fr': '← Sélectionner une méthode de nettoyage différente'
    },
    'use_sidebar_options': {
        'en': '👈 Use these options to clean your data',
        'fr': '👈 Utilisez ces options pour nettoyer vos données'
    },
    'all_cleaning_tools': {
        'en': '👈 All data cleaning tools are available in the sidebar. Use them to transform your data.',
        'fr': '👈 Tous les outils de nettoyage de données sont disponibles dans la barre latérale. Utilisez-les pour transformer vos données.'
    },
    'standard_data_cleaning': {
        'en': 'Standard Data Cleaning ⚡',
        'fr': 'Nettoyage standard des données ⚡'
    },
    'ai_powered_data_cleaning': {
        'en': 'AI-Powered Data Cleaning 🤖',
        'fr': 'Nettoyage des données par IA 🤖'
    },
    'select_dataset_to_clean': {
        'en': 'Select Dataset to Clean',
        'fr': 'Sélectionner le jeu de données à nettoyer'
    },
    'original_dataset': {
        'en': '📁 Original Dataset',
        'fr': '📁 Jeu de données original'
    },
    'cleaned_dataset': {
        'en': 'Cleaned Dataset',
        'fr': 'Jeu de données nettoyé'
    },
    'choose_file_to_clean': {
        'en': 'Choose a file to clean:',
        'fr': 'Choisir un fichier à nettoyer :'
    },
    'rows': {
        'en': 'Rows',
        'fr': 'Lignes'
    },
    'columns': {
        'en': 'Columns',
        'fr': 'Colonnes'
    },
    'missing_values': {
        'en': 'Missing Values',
        'fr': 'Valeurs manquantes'
    },
    'duplicates': {
        'en': 'Duplicates',
        'fr': 'Doublons'
    },
    'working_dataset': {
        'en': '🛠️ Working Dataset',
        'fr': '🛠️ Jeu de données de travail'
    },
    'current_data_preview': {
        'en': 'Current Data Preview:',
        'fr': 'Aperçu des données actuelles :'
    },
    'original_data_preview': {
        'en': 'Original Data Preview:',
        'fr': 'Aperçu des données originales :'
    },
    'current_data_types': {
        'en': 'Current Data Types:',
        'fr': 'Types de données actuels :'
    },
    'original_data_types': {
        'en': 'Original Data Types:',
        'fr': 'Types de données originaux :'
    },
    'generate_dashboard': {
        'en': 'Generate Dashboard',
        'fr': 'Générer le tableau de bord'
    },
    'analyze_cleaned_data': {
        'en': 'Analyze Cleaned Data',
        'fr': 'Analyser les données nettoyées'
    },
    'analyze_processed_data': {
        'en': '📊 Analyze Processed Data',
        'fr': '📊 Analyser les données traitées'
    },
    'select_different_dataset': {
        'en': '← Select Different Dataset', 
        'fr': '← Sélectionner un autre jeu de données'
    },
    'no_changes_made': {
        'en': '⚠️ No changes have been made to the dataset.',
        'fr': '⚠️ Aucune modification n\'a été apportée au jeu de données.'
    },
    'saved_cleaned_version': {
        'en': '✅ Saved cleaned version of',
        'fr': '✅ Version nettoyée sauvegardée de'
    },
    
    # Data cleaning options
    'data_cleaning_options': {
        'en': 'Data Cleaning Options',
        'fr': 'Options de nettoyage des données'
    },
    'text_cleaning_formatting': {
        'en': 'Text Cleaning & Formatting',
        'fr': 'Nettoyage et formatage du texte'
    },
    'delete_leading_spaces': {
        'en': 'Delete leading spaces',
        'fr': 'Supprimer les espaces au début'
    },
    'delete_trailing_spaces': {
        'en': 'Delete trailing spaces',
        'fr': 'Supprimer les espaces à la fin'
    },
    'delete_extra_spaces': {
        'en': 'Delete extra white spaces',
        'fr': 'Supprimer les espaces blancs supplémentaires'
    },
    'remove_punctuation': {
        'en': 'Remove punctuation/special chars',
        'fr': 'Supprimer la ponctuation/caractères spéciaux'
    },
    'capitalize_text': {
        'en': 'Capitalize text (Title Case)',
        'fr': 'Mettre en majuscules (Casse de titre)'
    },
    'apply_text_cleaning': {
        'en': 'Apply Text Cleaning to All Columns',
        'fr': 'Appliquer le nettoyage de texte à toutes les colonnes'
    },
    
    # Additional text cleaning translations
    'text_cleaning_applied': {
        'en': 'Text cleaning applied!',
        'fr': 'Nettoyage de texte appliqué !'
    },
    
    # Split column translations
    'split_column_delimiter': {
        'en': 'Split Column by Delimiter',
        'fr': 'Diviser la colonne par délimiteur'
    },
    'select_column': {
        'en': 'Select column',
        'fr': 'Sélectionner une colonne'
    },
    'delimiter': {
        'en': 'Delimiter',
        'fr': 'Délimiteur'
    },
    'fill_missing_values': {
        'en': 'Fill missing values',
        'fr': 'Remplir les valeurs manquantes'
    },
    'drop_original': {
        'en': 'Drop original',
        'fr': 'Supprimer l\'original'
    },
    'apply_split': {
        'en': 'Apply Split',
        'fr': 'Appliquer la division'
    },
    'split_successful': {
        'en': 'Split successful!',
        'fr': 'Division réussie !'
    },
    
    # Rename columns translations
    'rename_columns': {
        'en': 'Rename Columns',
        'fr': 'Renommer les colonnes'
    },
    'new_name': {
        'en': 'New name',
        'fr': 'Nouveau nom'
    },
    'queue_rename': {
        'en': 'Queue Rename',
        'fr': 'Mettre en file d\'attente'
    },
    'rename_queued': {
        'en': 'Rename queued!',
        'fr': 'Renommage mis en file d\'attente !'
    },
    'pending_renames': {
        'en': 'Pending renames:',
        'fr': 'Renommages en attente :'
    },
    'apply_renames': {
        'en': 'Apply Renames',
        'fr': 'Appliquer les renommages'
    },
    'renames_applied': {
        'en': 'Renames applied!',
        'fr': 'Renommages appliqués !'
    },
    
    # Drop columns translations
    'drop_columns': {
        'en': 'Drop Columns',
        'fr': 'Supprimer des colonnes'
    },
    'select_columns_to_drop': {
        'en': 'Select columns to drop:',
        'fr': 'Sélectionner les colonnes à supprimer :'
    },
    'drop_columns_btn': {
        'en': 'Drop Columns',
        'fr': 'Supprimer les colonnes'
    },
    'dropped': {
        'en': 'Dropped',
        'fr': 'Supprimé'
    },
    'no_columns_selected': {
        'en': 'No columns selected',
        'fr': 'Aucune colonne sélectionnée'
    },
    
    # Change data types translations
    'change_data_types': {
        'en': 'Change Data Types',
        'fr': 'Changer les types de données'
    },
    'select_columns': {
        'en': 'Select columns',
        'fr': 'Sélectionner les colonnes'
    },
    'new_type': {
        'en': 'New type',
        'fr': 'Nouveau type'
    },
    'convert': {
        'en': 'Convert',
        'fr': 'Convertir'
    },
    'converted': {
        'en': 'Converted',
        'fr': 'Converti'
    },
    'to': {
        'en': 'to',
        'fr': 'en'
    },
    'error_converting': {
        'en': 'Error converting',
        'fr': 'Erreur lors de la conversion'
    },
    
    # Handle outliers translations
    'handle_outliers': {
        'en': 'Handle Outliers',
        'fr': 'Gérer les valeurs aberrantes'
    },
    'method': {
        'en': 'Method',
        'fr': 'Méthode'
    },
    'winsorization': {
        'en': 'Winsorization',
        'fr': 'Winsorisation'
    },
    'lower_percentile': {
        'en': 'Lower percentile:',
        'fr': 'Percentile inférieur :'
    },
    'upper_percentile': {
        'en': 'Upper percentile:',
        'fr': 'Percentile supérieur :'
    },
    'fix_outliers': {
        'en': 'Fix Outliers',
        'fr': 'Corriger les valeurs aberrantes'
    },
    'outliers_handled': {
        'en': 'Outliers handled for',
        'fr': 'Valeurs aberrantes traitées pour'
    },
    'error_with': {
        'en': 'Error with',
        'fr': 'Erreur avec'
    },
    
    # Handle missing values translations
    'handle_missing_values': {
        'en': 'Handle Missing Values',
        'fr': 'Gérer les valeurs manquantes'
    },
    'drop': {
        'en': 'Drop',
        'fr': 'Supprimer'
    },
    'fill_mean': {
        'en': 'Fill with Mean',
        'fr': 'Remplir avec la moyenne'
    },
    'fill_median': {
        'en': 'Fill with Median',
        'fr': 'Remplir avec la médiane'
    },
    'fill_mode': {
        'en': 'Fill with Mode',
        'fr': 'Remplir avec le mode'
    },
    'fill_sequential': {
        'en': 'Fill Sequential Gaps',
        'fr': 'Remplir les écarts séquentiels'
    },
    'fill_constant': {
        'en': 'Fill with constant',
        'fr': 'Remplir avec une constante'
    },
    'fill_value': {
        'en': 'Fill value:',
        'fr': 'Valeur de remplissage :'
    },
    'handle_missing': {
        'en': 'Handle Missing',
        'fr': 'Traiter les valeurs manquantes'
    },
    'rows_with_missing': {
        'en': 'rows with missing values',
        'fr': 'lignes avec des valeurs manquantes'
    },
    'column': {
        'en': 'Column',
        'fr': 'Colonne'
    },
    'must_be_integer': {
        'en': 'must be integer type for sequential gap filling.',
        'fr': 'doit être de type entier pour le remplissage séquentiel.'
    },
    'filled_sequential': {
        'en': 'Filled sequential gaps in',
        'fr': 'Écarts séquentiels remplis dans'
    },
    'enter_fill_value': {
        'en': 'Please enter a fill value',
        'fr': 'Veuillez entrer une valeur de remplissage'
    },
    'filled_missing': {
        'en': 'Filled missing values in',
        'fr': 'Valeurs manquantes remplies dans'
    },
    'with': {
        'en': 'with',
        'fr': 'avec'
    },
    'fill_value_match': {
        'en': 'Fill value must match column type of',
        'fr': 'La valeur de remplissage doit correspondre au type de colonne de'
    },
    'cannot_calc_mean': {
        'en': 'Cannot calculate mean/median for non-numeric column',
        'fr': 'Impossible de calculer la moyenne/médiane pour une colonne non numérique'
    },
    'no_missing_columns': {
        'en': 'No columns with missing values found',
        'fr': 'Aucune colonne avec des valeurs manquantes trouvée'
    },
    
    # Drop duplicates translations
    'drop_duplicates': {
        'en': 'Drop Duplicates',
        'fr': 'Supprimer les doublons'
    },
    'select_columns_duplicates': {
        'en': 'Select columns to check for duplicates (empty = all columns):',
        'fr': 'Sélectionner les colonnes à vérifier pour les doublons (vide = toutes les colonnes) :'
    },
    'keep_option': {
        'en': 'Keep option:',
        'fr': 'Option de conservation :'
    },
    'drop_duplicates_btn': {
        'en': 'Drop duplicates',
        'fr': 'Supprimer les doublons'
    },
    'removed': {
        'en': 'Removed',
        'fr': 'Supprimé'
    },
    'duplicates': {
        'en': 'duplicates!',
        'fr': 'doublons !'
    },
    'no_duplicates': {
        'en': 'No duplicates found',
        'fr': 'Aucun doublon trouvé'
    },
    
    # Reorder columns translations
    'reorder_columns': {
        'en': 'Reorder Columns',
        'fr': 'Réorganiser les colonnes'
    },
    'edit_column_order': {
        'en': 'Edit column order (comma-separated)',
        'fr': 'Modifier l\'ordre des colonnes (séparées par des virgules)'
    },
    'apply_column_order': {
        'en': 'Apply Column Order',
        'fr': 'Appliquer l\'ordre des colonnes'
    },
    'missing_columns': {
        'en': 'Missing columns',
        'fr': 'Colonnes manquantes'
    },
    'unknown_columns': {
        'en': 'Unknown columns',
        'fr': 'Colonnes inconnues'
    },
    'column_order_updated': {
        'en': 'Column order updated!',
        'fr': 'Ordre des colonnes mis à jour !'
    },
    
    # Dashboard page
    'dashboard_coming_soon': {
        'en': '🚧 Dashboard Coming Soon',
        'fr': '🚧 Tableau de bord à venir'
    },
    
    # Add Dashboard page translations
    'Dashboard Configuration': {
        'en': 'Dashboard Configuration',
        'fr': 'Configuration du tableau de bord'
    },
    'Step 1: Choose Data Context': {
        'en': 'Step 1: Choose Data Context',
        'fr': 'Étape 1 : Choisir le contexte des données'
    },
    'Select the domain that best matches your data to get more relevant visualization recommendations.': {
        'en': 'Select the domain that best matches your data to get more relevant visualization recommendations.',
        'fr': 'Sélectionnez le domaine qui correspond le mieux à vos données pour obtenir des recommandations de visualisation plus pertinentes.'
    },
    'Select a domain': {
        'en': 'Select a domain',
        'fr': 'Sélectionnez un domaine'
    },
    'Select dataset domain': {
        'en': 'Select dataset domain',
        'fr': 'Sélectionnez le domaine du jeu de données'
    },
    'Next': {
        'en': 'Next',
        'fr': 'Suivant'
    },
    'Using cleaned data from previous steps': {
        'en': 'Using cleaned data from previous steps',
        'fr': 'Utilisation des données nettoyées des étapes précédentes'
    },
    'Using raw data from previous steps': {
        'en': 'Using raw data from previous steps',
        'fr': 'Utilisation des données brutes des étapes précédentes'
    },
    'Using previously uploaded file': {
        'en': 'Using previously uploaded file',
        'fr': 'Utilisation du fichier précédemment téléchargé'
    },
    'No data available. Please go back to process data first.': {
        'en': 'No data available. Please go back to process data first.',
        'fr': 'Aucune donnée disponible. Veuillez d\'abord retourner traiter les données.'
    },
    'Back to Home': {
        'en': 'Back to Home',
        'fr': 'Retour à l\'accueil'
    },
    '### 🏆 Recommended Metrics': {
        'en': '### 🏆 Recommended Metrics',
        'fr': '### 🏆 Métriques recommandées'
    },
    '🔄 Browse Alternatives': {
        'en': '🔄 Browse Alternatives',
        'fr': '🔄 Parcourir les alternatives'
    },
    'Select a metric to replace and browse alternatives.': {
        'en': 'Select a metric to replace and browse alternatives.',
        'fr': 'Sélectionnez une métrique à remplacer et parcourez les alternatives.'
    },
    'Select metric to replace:': {
        'en': 'Select metric to replace:',
        'fr': 'Sélectionnez la métrique à remplacer :'
    },
    'Current Metric': {
        'en': 'Current Metric',
        'fr': 'Métrique actuelle'
    },
    'Alternative': {
        'en': 'Alternative',
        'fr': 'Alternative'
    },
    'of': {
        'en': 'of',
        'fr': 'sur'
    },
    'Score': {
        'en': 'Score',
        'fr': 'Score'
    },
    'Use This Alternative': {
        'en': 'Use This Alternative',
        'fr': 'Utiliser cette alternative'
    },
    'Metric replaced successfully!': {
        'en': 'Metric replaced successfully!',
        'fr': 'Métrique remplacée avec succès !'
    },
    'No alternatives available for this metric.': {
        'en': 'No alternatives available for this metric.',
        'fr': 'Aucune alternative disponible pour cette métrique.'
    },
    'No data found. Please return to previous steps.': {
        'en': 'No data found. Please return to previous steps.',
        'fr': 'Aucune donnée trouvée. Veuillez retourner aux étapes précédentes.'
    },
    'Select recommendation to replace:': {
        'en': 'Select recommendation to replace:',
        'fr': 'Sélectionnez la recommandation à remplacer :'
    },
    'Select a recommendation to replace and browse alternatives.': {
        'en': 'Select a recommendation to replace and browse alternatives.',
        'fr': 'Sélectionnez une recommandation à remplacer et parcourez les alternatives.'
    },
    'Preview': {
        'en': 'Preview',
        'fr': 'Aperçu'
    },
    'Not enough visualizations. Please complete step 3 with at least 5 visualizations.': {
        'en': 'Not enough visualizations. Please complete step 3 with at least 5 visualizations.',
        'fr': 'Pas assez de visualisations. Veuillez compléter l\'étape 3 avec au moins 5 visualisations.'
    },
    'No visualization recommendations available. Please complete step 3 first.': {
        'en': 'No visualization recommendations available. Please complete step 3 first.',
        'fr': 'Aucune recommandation de visualisation disponible. Veuillez d\'abord compléter l\'étape 3.'
    },
    '← Previous': {
        'en': '← Previous',
        'fr': '← Précédent'
    },
    'Approve & View Dashboard': {
        'en': 'Approve & View Dashboard',
        'fr': 'Approuver et afficher le tableau de bord'
    },
    'No.': {
        'en': 'No.',
        'fr': 'N°'
    },
    'Name': {
        'en': 'Name',
        'fr': 'Nom'
    },
    'Type': {
        'en': 'Type',
        'fr': 'Type'
    },
    'Columns': {
        'en': 'Columns',
        'fr': 'Colonnes'
    },
    'Not enough numeric columns for parallel coordinates. Using scatter plot instead.': {
        'en': 'Not enough numeric columns for parallel coordinates. Using scatter plot instead.',
        'fr': 'Pas assez de colonnes numériques pour les coordonnées parallèles. Utilisation d\'un nuage de points à la place.'
    },
    'No geographic column identified for choropleth map.': {
        'en': 'No geographic column identified for choropleth map.',
        'fr': 'Aucune colonne géographique identifiée pour la carte choroplèthe.'
    },
    'Step 1: Domain': {
        'en': 'Step 1: Domain',
        'fr': 'Étape 1 : Domaine'
    },
    'Step 2: Metrics': {
        'en': 'Step 2: Metrics',
        'fr': 'Étape 2 : Métriques'
    },
    'Step 3: Visualization': {
        'en': 'Step 3: Visualization',
        'fr': 'Étape 3 : Visualisation'
    },
    'Step 4: Dashboard': {
        'en': 'Step 4: Dashboard',
        'fr': 'Étape 4 : Tableau de bord'
    },
    '🔵': {
        'en': '🔵',
        'fr': '🔵'
    },
    '✅': {
        'en': '✅',
        'fr': '✅'
    },
    '⚪': {
        'en': '⚪',
        'fr': '⚪'
    },
    
    # Visualization labels
    'by': {
        'en': 'by',
        'fr': 'par'
    },
    'over time': {
        'en': 'over time',
        'fr': 'au fil du temps'
    },
    'over time by': {
        'en': 'over time by',
        'fr': 'au fil du temps par'
    },
    'vs': {
        'en': 'vs',
        'fr': 'vs'
    },
    'Distribution of': {
        'en': 'Distribution of',
        'fr': 'Distribution de'
    },
    'Count by': {
        'en': 'Count by',
        'fr': 'Comptage par'
    },
    'and': {
        'en': 'and',
        'fr': 'et'
    },
    'Heatmap of': {
        'en': 'Heatmap of',
        'fr': 'Carte thermique de'
    },
    'Correlation heatmap of numeric columns': {
        'en': 'Correlation heatmap of numeric columns',
        'fr': 'Carte thermique de corrélation des colonnes numériques'
    },
    'Parallel coordinates plot': {
        'en': 'Parallel coordinates plot',
        'fr': 'Graphique à coordonnées parallèles'
    },
    'plot of': {
        'en': 'plot of',
        'fr': 'graphique de'
    },
    'Map of locations': {
        'en': 'Map of locations',
        'fr': 'Carte des emplacements'
    },
    'Choropleth map of': {
        'en': 'Choropleth map of',
        'fr': 'Carte choroplèthe de'
    },
    'Relationship between': {
        'en': 'Relationship between',
        'fr': 'Relation entre'
    },
    'colored by': {
        'en': 'colored by',
        'fr': 'coloré par'
    },
    'Relationship between columns': {
        'en': 'Relationship between columns',
        'fr': 'Relation entre les colonnes'
    },
    
    # Error messages
    'Could not create': {
        'en': 'Could not create',
        'fr': 'Impossible de créer'
    },
    'Error displaying visualization': {
        'en': 'Error displaying visualization',
        'fr': 'Erreur d\'affichage de la visualisation'
    },
    
    # Process page translations
    'ai_powered_data_processing': {
        'en': 'AI-Powered Data Processing 🤖',
        'fr': 'Traitement de données par IA 🤖'
    },
    'standard_data_processing': {
        'en': 'Standard Data Processing ⚡',
        'fr': 'Traitement standard des données ⚡'
    },
    'manually_combine_datasets': {
        'en': 'Manually combine and merge your datasets',
        'fr': 'Combiner et fusionner manuellement vos jeux de données'
    },
    'original_datasets': {
        'en': 'Original Datasets',
        'fr': 'Jeux de données originaux'
    },
    'available_datasets': {
        'en': 'Available Datasets',
        'fr': 'Jeux de données disponibles'
    },
    'columns_list': {
        'en': 'Columns',
        'fr': 'Colonnes'
    },
    'process_with_ai_button': {
        'en': '🔄 Process with AI',
        'fr': '🔄 Traiter avec l\'IA'
    },
    'processing_with_ai': {
        'en': 'Processing datasets with AI... This may take a minute.',
        'fr': 'Traitement des données avec l\'IA... Cela peut prendre une minute.'
    },
    'ai_processing_failed': {
        'en': 'AI processing failed',
        'fr': 'Le traitement par IA a échoué'
    },
    'processing_completed': {
        'en': '✅ Processing completed successfully!',
        'fr': '✅ Traitement terminé avec succès !'
    },
    'processing_result': {
        'en': 'Processing Result',
        'fr': 'Résultat du traitement'
    },
    'files_processed': {
        'en': 'Files Processed',
        'fr': 'Fichiers traités'
    },
    'original_rows': {
        'en': 'Original Rows',
        'fr': 'Lignes originales'
    },
    'result_rows': {
        'en': 'Result Rows',
        'fr': 'Lignes résultantes'
    },
    'result_columns': {
        'en': 'Result Columns',
        'fr': 'Colonnes résultantes'
    },
    'process_another_dataset': {
        'en': '← Process Another Dataset',
        'fr': '← Traiter un autre jeu de données'
    },
    'proceed_to_cleaning': {
        'en': 'Proceed to Cleaning →',
        'fr': 'Passer au nettoyage →'
    },
    'error_during_ai_processing': {
        'en': 'Error during AI processing',
        'fr': 'Erreur pendant le traitement par IA'
    },
    'select_different_processing_method': {
        'en': '← Select Different Processing Method',
        'fr': '← Sélectionner une méthode de traitement différente'
    },
    'concatenation_tab': {
        'en': '🔗 Concatenation',
        'fr': '🔗 Concaténation'
    },
    'merging_tab': {
        'en': '🔀 Merging',
        'fr': '🔀 Fusion'
    },
    'concatenation_options': {
        'en': 'Concatenation Options',
        'fr': 'Options de concaténation'
    },
    'select_files_to_concat': {
        'en': 'Select files to concatenate:',
        'fr': 'Sélectionner les fichiers à concaténer :'
    },
    'configuration': {
        'en': 'Configuration',
        'fr': 'Configuration'
    },
    'reset_index': {
        'en': 'Reset Index',
        'fr': 'Réinitialiser l\'index'
    },
    'reset_index_help': {
        'en': 'If checked, the resulting dataframe will have a fresh index starting from 0',
        'fr': 'Si coché, le dataframe résultant aura un nouvel index commençant à 0'
    },
    'all_columns_match': {
        'en': '✅ All columns match! You can safely concatenate vertically.',
        'fr': '✅ Toutes les colonnes correspondent ! Vous pouvez concaténer verticalement en toute sécurité.'
    },
    'vertical_concatenation_btn': {
        'en': 'Vertical Concatenation 🠗',
        'fr': 'Concaténation verticale 🠗'
    },
    'processing': {
        'en': 'Processing...',
        'fr': 'Traitement en cours...'
    },
    'dataset_auto_saved': {
        'en': '✅ Dataset automatically saved as',
        'fr': '✅ Jeu de données automatiquement enregistré sous'
    },
    'use_proceed_button': {
        'en': 'Use the \'Proceed to Cleaning\' button at the bottom of the page to continue.',
        'fr': 'Utilisez le bouton \'Passer au nettoyage\' au bas de la page pour continuer.'
    },
    'no_common_columns': {
        'en': '⚠️ No common columns detected! Consider horizontal concatenation.',
        'fr': '⚠️ Aucune colonne commune détectée ! Envisagez une concaténation horizontale.'
    },
    'horizontal_concatenation_btn': {
        'en': 'Horizontal Concatenation ⇨',
        'fr': 'Concaténation horizontale ⇨'
    },
    'concatenation_error': {
        'en': 'Concatenation error',
        'fr': 'Erreur de concaténation'
    },
    'column_mismatch_detected': {
        'en': 'ℹ️ Column mismatch detected. Choose concatenation method and options below.',
        'fr': 'ℹ️ Différence de colonnes détectée. Choisissez la méthode de concaténation et les options ci-dessous.'
    },
    'concatenation_method': {
        'en': 'Concatenation method:',
        'fr': 'Méthode de concaténation :'
    },
    'vertical': {
        'en': 'Vertical',
        'fr': 'Verticale'
    },
    'horizontal': {
        'en': 'Horizontal',
        'fr': 'Horizontale'
    },
    'map_matching_columns': {
        'en': 'Map matching columns for vertical concatenation',
        'fr': 'Mapper les colonnes correspondantes pour la concaténation verticale'
    },
    'select_columns_to_unify': {
        'en': 'Select columns to unify:',
        'fr': 'Sélectionner les colonnes à unifier :'
    },
    'new_unified_column_name': {
        'en': 'New unified column name:',
        'fr': 'Nouveau nom de colonne unifié :'
    },
    'apply_column_unification': {
        'en': 'Apply Column Unification',
        'fr': 'Appliquer l\'unification des colonnes'
    },
    'unified_columns': {
        'en': '✅ Unified',
        'fr': '✅ Unifié'
    },
    'into': {
        'en': 'columns into',
        'fr': 'colonnes en'
    },
    'active_mappings': {
        'en': 'Active Mappings',
        'fr': 'Mappages actifs'
    },
    'clear_all_mappings': {
        'en': 'Clear All Mappings',
        'fr': 'Effacer tous les mappages'
    },
    'column_mapping_complete': {
        'en': '✅ Column mapping is complete! Columns now match.',
        'fr': '✅ Le mappage des colonnes est terminé ! Les colonnes correspondent maintenant.'
    },
    'perform_vertical_concat': {
        'en': 'Perform Vertical Concatenation 🠗',
        'fr': 'Effectuer la concaténation verticale 🠗'
    },
    'vertical_concat_error': {
        'en': 'Error during vertical concatenation',
        'fr': 'Erreur lors de la concaténation verticale'
    },
    'columns_still_dont_match': {
        'en': '⚠️ Columns still don\'t match. Add more mappings to continue.',
        'fr': '⚠️ Les colonnes ne correspondent toujours pas. Ajoutez plus de mappages pour continuer.'
    },
    'rename_common_columns': {
        'en': 'Rename common columns for horizontal concatenation',
        'fr': 'Renommer les colonnes communes pour la concaténation horizontale'
    },
    'common_columns_to_rename': {
        'en': 'Common Columns to Rename',
        'fr': 'Colonnes communes à renommer'
    },
    'rename': {
        'en': 'Rename',
        'fr': 'Renommer'
    },
    'new_name_for': {
        'en': 'New name for',
        'fr': 'Nouveau nom pour'
    },
    'in': {
        'en': 'in',
        'fr': 'dans'
    },
    'apply_renames_for': {
        'en': 'Apply Renames for',
        'fr': 'Appliquer les renommages pour'
    },
    'renamed': {
        'en': 'Renamed',
        'fr': 'Renommé'
    },
    'perform_horizontal_concat': {
        'en': 'Perform Horizontal Concatenation ⇨',
        'fr': 'Effectuer la concaténation horizontale ⇨'
    },
    'horizontal_concat_failed': {
        'en': 'Horizontal concatenation failed',
        'fr': 'La concaténation horizontale a échoué'
    },
    'concatenation_result': {
        'en': 'Concatenation Result',
        'fr': 'Résultat de la concaténation'
    },
    'select_at_least_two_files': {
        'en': 'Select at least 2 files for concatenation',
        'fr': 'Sélectionnez au moins 2 fichiers pour la concaténation'
    },
    'upload_at_least_two_files': {
        'en': 'Upload at least 2 files to use concatenation',
        'fr': 'Téléchargez au moins 2 fichiers pour utiliser la concaténation'
    },
    'merge_configuration': {
        'en': 'Merge Configuration',
        'fr': 'Configuration de fusion'
    },
    'left_dataset': {
        'en': 'Left Dataset',
        'fr': 'Jeu de données de gauche'
    },
    'select_left_dataset': {
        'en': 'Select left dataset',
        'fr': 'Sélectionner le jeu de données de gauche'
    },
    'right_dataset': {
        'en': 'Right Dataset',
        'fr': 'Jeu de données de droite'
    },
    'select_right_dataset': {
        'en': 'Select right dataset',
        'fr': 'Sélectionner le jeu de données de droite'
    },
    'join_type': {
        'en': 'Join Type',
        'fr': 'Type de jointure'
    },
    'join_type_help': {
        'en': '- inner: keep only matching rows\n- left: keep all rows from left dataset\n- right: keep all rows from right dataset\n- outer: keep all rows from both datasets',
        'fr': '- inner: ne conserver que les lignes correspondantes\n- left: conserver toutes les lignes du jeu de données de gauche\n- right: conserver toutes les lignes du jeu de données de droite\n- outer: conserver toutes les lignes des deux jeux de données'
    },
    'join': {
        'en': 'Join',
        'fr': 'Jointure'
    },
    'key_pairing': {
        'en': 'Key Pairing',
        'fr': 'Appariement des clés'
    },
    'select_columns_to_join': {
        'en': 'Select the columns to join on from each dataset',
        'fr': 'Sélectionner les colonnes à joindre de chaque jeu de données'
    },
    'found_common_columns': {
        'en': '✅ Found common columns',
        'fr': '✅ Colonnes communes trouvées'
    },
    'use_common_column_names': {
        'en': 'Use common column names for join',
        'fr': 'Utiliser les noms de colonnes communs pour la jointure'
    },
    'left_key': {
        'en': 'Left key',
        'fr': 'Clé gauche'
    },
    'right_key': {
        'en': 'Right key',
        'fr': 'Clé droite'
    },
    'number_of_key_pairs': {
        'en': 'Number of key pairs',
        'fr': 'Nombre de paires de clés'
    },
    'no_common_column_names': {
        'en': '⚠️ No common column names found. Select keys manually.',
        'fr': '⚠️ Aucun nom de colonne commun trouvé. Sélectionnez les clés manuellement.'
    },
    'advanced_options': {
        'en': 'Advanced Options',
        'fr': 'Options avancées'
    },
    'suffixes_for_duplicates': {
        'en': 'Suffixes for duplicate columns',
        'fr': 'Suffixes pour les colonnes en double'
    },
    'suffixes_help': {
        'en': 'Suffixes to add to duplicate column names',
        'fr': 'Suffixes à ajouter aux noms de colonnes en double'
    },
    'validate_join_keys': {
        'en': 'Validate join keys',
        'fr': 'Valider les clés de jointure'
    },
    'validate_help': {
        'en': 'Validate join keys to ensure no duplicate values',
        'fr': 'Valider les clés de jointure pour s\'assurer qu\'il n\'y a pas de valeurs en double'
    },
    'execute_merge': {
        'en': 'Execute Merge',
        'fr': 'Exécuter la fusion'
    },
    'merging_datasets': {
        'en': 'Merging datasets...',
        'fr': 'Fusion des jeux de données...'
    },
    'merge_completed': {
        'en': '✅ Merge completed and dataset automatically saved as',
        'fr': '✅ Fusion terminée et jeu de données automatiquement enregistré sous'
    },
    'merge_failed': {
        'en': 'Merge failed',
        'fr': 'La fusion a échoué'
    },
    'merge_result': {
        'en': 'Merge Result',
        'fr': 'Résultat de la fusion'
    },
    'left_rows': {
        'en': 'Left Rows',
        'fr': 'Lignes gauches'
    },
    'right_rows': {
        'en': 'Right Rows',
        'fr': 'Lignes droites'
    },
    'processing_completed_proceed': {
        'en': '✅ Processing completed! You can now proceed to cleaning.',
        'fr': '✅ Traitement terminé ! Vous pouvez maintenant passer au nettoyage.'
    },
    
    # Home page additional translations
    'files_uploaded_successfully': {
        'en': 'file(s) uploaded successfully',
        'fr': 'fichier(s) téléchargé(s) avec succès'
    },
    'some_files_had_errors': {
        'en': '⚠️ Some files were uploaded but had errors. Check the details below.',
        'fr': '⚠️ Certains fichiers ont été téléchargés mais présentent des erreurs. Vérifiez les détails ci-dessous.'
    },
    'let_ai_analyze': {
        'en': 'Let our AI analyze and combine your datasets',
        'fr': 'Laissez notre IA analyser et combiner vos jeux de données'
    },
    'error_processing_file': {
        'en': 'Error processing file',
        'fr': 'Erreur lors du traitement du fichier'
    },
    'empty_csv_file': {
        'en': 'The CSV file is empty.',
        'fr': 'Le fichier CSV est vide.'
    },
    'bytes': {
        'en': 'bytes',
        'fr': 'octets'
    },
    'kb': {
        'en': 'KB',
        'fr': 'Ko'
    },
    'mb': {
        'en': 'MB',
        'fr': 'Mo'
    },
    'need_at_least_two_files': {
        'en': '⚠️ You need at least 2 files to process. Please upload more files.',
        'fr': '⚠️ Vous avez besoin d\'au moins 2 fichiers à traiter. Veuillez télécharger plus de fichiers.'
    },
    'no_files_uploaded': {
        'en': 'No files uploaded. Please upload files on the home page first.',
        'fr': 'Aucun fichier téléchargé. Veuillez d\'abord télécharger des fichiers sur la page d\'accueil.'
    },
    
    # Analysis page
    'Analysis': {
        'en': 'Analysis',
        'fr': 'Analyse'
    }
}

def get_translation(text_id, lang='en'):
    """
    Get the translation for a given text_id in the specified language.
    
    Args:
        text_id (str): The ID of the text to translate
        lang (str): The language code (default: 'en')
        
    Returns:
        str: The translated text, or the text_id if no translation is found
    """
    if text_id in translations:
        if lang in translations[text_id]:
            return translations[text_id][lang]
        elif 'en' in translations[text_id]:  # Fallback to English
            return translations[text_id]['en']
    return text_id  # Return the text_id if no translation is found 

def get_translation_function():
    """
    Return a function that can be used for translating text.
    This is a wrapper around get_translation that uses session state language.
    
    Returns:
        function: A function that takes a text_id and returns the translated text
    """
    def t(text_id):
        return get_translation(text_id, st.session_state.get("language", "en"))
    return t 