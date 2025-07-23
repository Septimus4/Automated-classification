"""
HR Analytics Utilities Module
============================

This module contains data preparation, transformation logic (including logarithmic transformations),
and graph generation functions for the TechNova Partners HR Analytics project.

Functions are designed to be imported and used in Jupyter notebooks for visualization and analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import sqlite3
import json
import os
from datetime import datetime
from scipy import stats
import warnings
from pathlib import Path


# Path Configuration - Robust path handling
def get_project_root():
    """
    Find and return the project root directory (contains pyproject.toml).
    Works regardless of where the code is executed from.
    """
    current_path = Path.cwd()

    # Look for pyproject.toml starting from current directory up to root
    for path in [current_path] + list(current_path.parents):
        if (path / 'pyproject.toml').exists():
            return path

    # Fallback: look for common project indicators
    for path in [current_path] + list(current_path.parents):
        indicators = ['README.md', 'hr_analytics_utils.py', 'extrait_sirh.csv']
        if any((path / indicator).exists() for indicator in indicators):
            return path

    # Last resort: return current working directory
    return current_path


def get_data_file_path(filename):
    """
    Get the full path to a data file, searching from project root.
    
    Parameters:
    -----------
    filename : str
        Name of the data file (e.g., 'extrait_sirh.csv')
        
    Returns:
    --------
    Path
        Full path to the data file
    """
    project_root = get_project_root()

    # Check if file exists in project root
    if (project_root / filename).exists():
        return project_root / filename

    # Check common data directories
    data_dirs = ['data', 'datasets', '.']
    for data_dir in data_dirs:
        data_path = project_root / data_dir / filename
        if data_path.exists():
            return data_path

    # Return expected path even if file doesn't exist (for creation)
    return project_root / filename


def get_results_dir():
    """
    Get the results directory path, creating it if it doesn't exist.
    
    Returns:
    --------
    Path
        Path to the results directory
    """
    project_root = get_project_root()
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    return results_dir


def get_database_path():
    """
    Get the database file path.
    
    Returns:
    --------
    Path
        Path to the SQLite database file
    """
    return get_results_dir() / 'technova_hr.db'


# Configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
TECHNOVA_COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D',
    'light': '#F4F4F9'
}

# Database Configuration
DB_PATH = str(get_database_path())


def load_raw_data_files():
    """
    Load raw data files from their default locations.
    
    Returns:
    --------
    tuple
        (df_sirh, df_eval, df_sondage) - The three main data files
    """
    print("Loading raw data files...")

    try:
        # Load each data file
        df_sirh = pd.read_csv(get_data_file_path('extrait_sirh.csv'))
        df_eval = pd.read_csv(get_data_file_path('extrait_eval.csv'))
        df_sondage = pd.read_csv(get_data_file_path('extrait_sondage.csv'))

        print(f"HRIS data loaded: {df_sirh.shape[0]} rows, {df_sirh.shape[1]} columns")
        print(f"Performance data loaded: {df_eval.shape[0]} rows, {df_eval.shape[1]} columns")
        print(f"Survey data loaded: {df_sondage.shape[0]} rows, {df_sondage.shape[1]} columns")

        return df_sirh, df_eval, df_sondage

    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print(f"   Project root: {get_project_root()}")
        print(f"   Looking for files in: {get_project_root()}")
        raise


def add_project_to_path():
    """
    Add the project root to Python path for imports.
    This allows importing hr_analytics_utils from anywhere.
    """
    import sys
    project_root = str(get_project_root())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added project root to Python path: {project_root}")
    else:
        print(f"Project root already in Python path: {project_root}")


def setup_notebook_environment():
    """
    One-stop setup function for notebooks to handle all path and import issues.
    Creates SQLite database file if it does not exist.

    Returns:
    --------
    dict
        Dictionary with useful paths and status information
    """
    import sys
    import sqlite3

    # Add project to path for imports
    add_project_to_path()

    # Get all important paths
    project_root = get_project_root()
    results_dir = get_results_dir()
    db_path = get_database_path()

    # Create the SQLite database if it doesn't exist
    if not db_path.exists():
        print(f"Creating new SQLite database at: {db_path}")
        db_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent dirs exist
        conn = sqlite3.connect(db_path)
        conn.close()

    # Print status
    print("NOTEBOOK ENVIRONMENT SETUP")
    print("=" * 35)
    print(f"Project root: {project_root}")
    print(f"Results directory: {results_dir}")
    print(f"Database path: {db_path}")
    print(f"Python path includes project: {str(project_root) in sys.path}")

    # Check for data files
    data_files = ['extrait_sirh.csv', 'extrait_eval.csv', 'extrait_sondage.csv']
    print(f"\nData files status:")
    for file in data_files:
        file_path = get_data_file_path(file)
        status = "OK" if file_path.exists() else "Error"
        print(f"   {status} {file}: {file_path}")

    # Final database status
    db_status = "OK" if db_path.exists() else "Error"
    print(f"\nDatabase status: {db_status} {db_path}")

    print("\nEnvironment setup complete!\n")

    return {
        'project_root': project_root,
        'results_dir': results_dir,
        'database_path': db_path,
        'data_files': {file: get_data_file_path(file) for file in data_files}
    }


def load_cleaned_data_from_db():
    """
    Load cleaned data from database (from Phase 1).
    
    Returns:
    --------
    pandas.DataFrame
        Cleaned and merged dataset
    """
    db_path = get_database_path()

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}. Please run Phase 1 (data wrangling) first.")

    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM employee_data", conn)
            print(f"Data loaded from database: {df.shape}")
            return df
    except sqlite3.OperationalError as e:
        print(f"Error loading from database: {e}")
        print("   Make sure Phase 1 (data wrangling) has been completed.")
        raise


class HRDataProcessor:
    """
    Class for processing HR data including transformations and feature engineering.
    """

    def __init__(self, X, y):
        """
        Initialize with features and target data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Features dataset
        y : pandas.Series
            Target variable (0=Stay, 1=Leave)
        """
        self.X = X.copy()
        self.y = y.copy()
        self.X_transformed = None
        self.log_transformed_features = []

    def apply_log_transformations(self, features_to_transform=None):
        """
        Apply logarithmic transformations to specified features or automatically detect
        skewed numerical features.
        
        Parameters:
        -----------
        features_to_transform : list, optional
            List of feature names to transform. If None, auto-detect skewed features.
            
        Returns:
        --------
        pandas.DataFrame
            Dataset with log-transformed features
        """
        self.X_transformed = self.X.copy()

        if features_to_transform is None:
            # Auto-detect numerical features that might benefit from log transformation
            numerical_features = self.X.select_dtypes(include=[np.number]).columns
            features_to_transform = []

            for feature in numerical_features:
                if (self.X[feature] > 0).all():  # Only positive values can be log-transformed
                    skewness = self.X[feature].skew()
                    if abs(skewness) > 1.5:  # Moderately to highly skewed features
                        features_to_transform.append(feature)

        # Apply log transformation
        for feature in features_to_transform:
            if feature in self.X.columns and (self.X[feature] > 0).all():
                self.X_transformed[f'{feature}_log'] = np.log1p(self.X[feature])
                self.log_transformed_features.append(f'{feature}_log')

        print(f"Applied log transformation to {len(features_to_transform)} features:")
        for feature in features_to_transform:
            if feature in self.X.columns:
                print(f"  - {feature} -> {feature}_log")

        return self.X_transformed

    def get_class_imbalance_stats(self):
        """
        Get comprehensive class imbalance statistics.
        
        Returns:
        --------
        dict
            Dictionary containing imbalance statistics
        """
        class_counts = self.y.value_counts()
        total_samples = len(self.y)

        return {
            'class_0_count': class_counts[0],
            'class_1_count': class_counts[1],
            'class_0_percentage': (class_counts[0] / total_samples) * 100,
            'class_1_percentage': (class_counts[1] / total_samples) * 100,
            'imbalance_ratio': class_counts[0] / class_counts[1],
            'minority_class_percentage': (class_counts[1] / total_samples) * 100,
            'total_samples': total_samples
        }

    def identify_most_imbalanced_features(self, top_n=10):
        """
        Identify features with the most significant class imbalance.
        
        Parameters:
        -----------
        top_n : int
            Number of top imbalanced features to return

        Returns:
        --------
        pandas.DataFrame
            Features ranked by imbalance severity
        """
        imbalance_stats = []

        for feature in self.X.columns:
            if self.X[feature].dtype in ['int64', 'float64']:
                # For numerical features, calculate mean difference between classes
                class_0_mean = self.X[self.y == 0][feature].mean()
                class_1_mean = self.X[self.y == 1][feature].mean()

                if class_0_mean != 0:
                    relative_diff = abs((class_1_mean - class_0_mean) / class_0_mean) * 100
                else:
                    relative_diff = abs(class_1_mean - class_0_mean) * 100

                imbalance_stats.append({
                    'feature': feature,
                    'class_0_mean': class_0_mean,
                    'class_1_mean': class_1_mean,
                    'absolute_difference': abs(class_1_mean - class_0_mean),
                    'relative_difference_pct': relative_diff,
                    'feature_type': 'numerical'
                })
            else:
                # For categorical features, calculate distribution differences
                class_0_dist = self.X[self.y == 0][feature].value_counts(normalize=True)
                class_1_dist = self.X[self.y == 1][feature].value_counts(normalize=True)

                # Calculate Hellinger distance as imbalance measure
                all_categories = set(class_0_dist.index) | set(class_1_dist.index)
                hellinger_dist = 0

                for category in all_categories:
                    p0 = class_0_dist.get(category, 0)
                    p1 = class_1_dist.get(category, 0)
                    hellinger_dist += (np.sqrt(p0) - np.sqrt(p1)) ** 2

                hellinger_dist = np.sqrt(hellinger_dist) / np.sqrt(2)

                imbalance_stats.append({
                    'feature': feature,
                    'hellinger_distance': hellinger_dist,
                    'relative_difference_pct': hellinger_dist * 100,
                    'feature_type': 'categorical'
                })

        imbalance_df = pd.DataFrame(imbalance_stats)
        return imbalance_df.nlargest(top_n, 'relative_difference_pct')


class DatabaseManager:
    """
    Manages SQLite database operations for HR analytics data.
    """

    def __init__(self, db_path=DB_PATH):
        """Initialize database manager with connection path."""
        self.db_path = db_path

    def get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)

    def save_model_results(self, results_data, table_name):
        """
        Save model results to database.
        
        Parameters:
        -----------
        results_data : pandas.DataFrame
            Results data to save
        table_name : str
            Name of the table to create/update
        """
        with self.get_connection() as conn:
            # Drop table if exists to replace with new data
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            results_data.to_sql(table_name, conn, index=False)
            print(f"Saved {len(results_data)} records to table '{table_name}'")

    def load_model_results(self, table_name):
        """
        Load model results from database.
        
        Parameters:
        -----------
        table_name : str
            Name of the table to load
            
        Returns:
        --------
        pandas.DataFrame
            Loaded data
        """
        with self.get_connection() as conn:
            try:
                return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            except sqlite3.OperationalError:
                print(f" Table '{table_name}' not found in database")
                return None

    def save_feature_analysis(self, feature_importance_df, feature_info_df=None):
        """
        Save feature analysis results to database.
        
        Parameters:
        -----------
        feature_importance_df : pandas.DataFrame
            Feature importance analysis
        feature_info_df : pandas.DataFrame, optional
            Additional feature information
        """
        with self.get_connection() as conn:
            # Save feature importance
            conn.execute("DROP TABLE IF EXISTS feature_importance")
            feature_importance_df.to_sql('feature_importance', conn, index=False)
            print(f"Saved feature importance analysis ({len(feature_importance_df)} features)")

            # Save feature info if provided
            if feature_info_df is not None:
                conn.execute("DROP TABLE IF EXISTS feature_metadata")
                feature_info_df.to_sql('feature_metadata', conn, index=False)
                print(f"Saved feature metadata ({len(feature_info_df)} features)")

    def save_model_predictions(self, predictions_df, model_name):
        """
        Save model predictions to database.
        
        Parameters:
        -----------
        predictions_df : pandas.DataFrame
            Predictions with actual values
        model_name : str
            Name of the model
        """
        table_name = f"predictions_{model_name.lower().replace(' ', '_')}"

        with self.get_connection() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            predictions_df.to_sql(table_name, conn, index=False)
            print(f"Saved {len(predictions_df)} predictions for '{model_name}' to table '{table_name}'")

    def save_analysis_metadata(self, analysis_type, metadata):
        """
        Save analysis metadata (parameters, settings, etc.) to database.
        
        Parameters:
        -----------
        analysis_type : str
            Type of analysis (e.g., 'baseline_modeling', 'class_imbalance')
        metadata : dict
            Metadata information
        """
        # Add timestamp
        metadata['timestamp'] = datetime.now().isoformat()
        metadata['analysis_type'] = analysis_type

        # Convert to DataFrame
        metadata_df = pd.DataFrame([metadata])

        with self.get_connection() as conn:
            # Append to analysis_metadata table
            metadata_df.to_sql('analysis_metadata', conn, if_exists='append', index=False)
            print(f"Saved metadata for '{analysis_type}' analysis")

    def get_table_info(self):
        """
        Get information about all tables in the database.
        
        Returns:
        --------
        dict
            Dictionary with table names and row counts
        """
        table_info = {}

        with self.get_connection() as conn:
            # Get all table names
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", conn
            )

            for table_name in tables['name']:
                count_result = pd.read_sql_query(
                    f"SELECT COUNT(*) as count FROM {table_name}", conn
                )
                table_info[table_name] = count_result['count'].iloc[0]

        return table_info

    def cleanup_old_tables(self, tables_to_remove):
        """
        Remove specified tables from database.
        
        Parameters:
        -----------
        tables_to_remove : list
            List of table names to remove
        """
        with self.get_connection() as conn:
            for table_name in tables_to_remove:
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                print(f"🗑️  Removed table '{table_name}'")


def migrate_csv_to_database():
    """
    Migrate existing CSV files to SQLite database and remove redundant files.
    """
    db_manager = DatabaseManager()
    results_path = get_results_dir()

    print("MIGRATING CSV DATA TO DATABASE")
    print("=" * 50)

    # Define CSV files and their corresponding database tables
    csv_migrations = {
        'baseline_model_comparison.csv': 'baseline_model_results',
        'class_imbalance_comparison.csv': 'class_imbalance_results',
        'feature_importance_baseline.csv': 'feature_importance',
        'top_models_for_tuning.csv': 'top_models_tuning',
        'baseline_predictions.csv': 'predictions_baseline'
    }

    migrated_files = []

    # Migrate each CSV file
    for csv_file, table_name in csv_migrations.items():
        csv_path = results_path / csv_file
        try:
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                db_manager.save_model_results(df, table_name)
                migrated_files.append(csv_file)
            else:
                print(f" File not found: {csv_file}")
        except Exception as e:
            print(f"Error migrating {csv_file}: {str(e)}")

    # Handle feature info separately if it exists
    feature_info_path = results_path / 'feature_info.csv'
    if feature_info_path.exists():
        try:
            feature_info_df = pd.read_csv(feature_info_path)
            with db_manager.get_connection() as conn:
                conn.execute("DROP TABLE IF EXISTS feature_metadata")
                feature_info_df.to_sql('feature_metadata', conn, index=False)
                print(f"Migrated feature_info.csv to 'feature_metadata' table")
                migrated_files.append('feature_info.csv')
        except Exception as e:
            print(f"Error migrating feature_info.csv: {str(e)}")

    # Show database status
    print(f"\n DATABASE STATUS:")
    table_info = db_manager.get_table_info()
    for table_name, count in table_info.items():
        print(f"   • {table_name}: {count:,} records")

    return migrated_files


def get_relevant_exports():
    """
    Identify which CSV files should be kept vs moved to database.
    
    Returns:
    --------
    dict
        Dictionary categorizing files
    """
    results_path = get_results_dir()

    # Files to keep as CSV (raw data, final datasets)
    keep_as_csv = [
        'merged_cleaned_data.csv',  # Core cleaned dataset
        'modeling_dataset.csv'  # Final modeling dataset
    ]

    # Files to migrate to database (results, analysis outputs)
    migrate_to_db = [
        'baseline_model_comparison.csv',
        'class_imbalance_comparison.csv',
        'feature_importance_baseline.csv',
        'top_models_for_tuning.csv',
        'baseline_predictions.csv',
        'feature_info.csv'
    ]

    # Files to remove after migration (large intermediate files)
    remove_after_migration = [
        'features_X.csv',  # Large feature matrix (in database as modeling_data)
        'target_y.csv'  # Target values (in database as target table)
    ]

    return {
        'keep_as_csv': keep_as_csv,
        'migrate_to_db': migrate_to_db,
        'remove_after_migration': remove_after_migration
    }


def create_database_summary_report():
    """
    Create a summary report of database contents and file organization.
    
    Returns:
    --------
    dict
        Database summary information
    """
    db_manager = DatabaseManager()

    # Get database info
    table_info = db_manager.get_table_info()

    # Get file categorization
    file_categories = get_relevant_exports()

    # Calculate storage savings
    results_path = get_results_dir()
    total_csv_size = 0
    migrated_size = 0

    for file_type, files in file_categories.items():
        for file_name in files:
            file_path = results_path / file_name
            if file_path.exists():
                try:
                    file_size = file_path.stat().st_size
                    total_csv_size += file_size
                    if file_type == 'migrate_to_db':
                        migrated_size += file_size
                except:
                    pass

    # Database file size
    db_path = get_database_path()
    db_size = db_path.stat().st_size if db_path.exists() else 0

    summary = {
        'database_tables': table_info,
        'total_records': sum(table_info.values()),
        'file_organization': file_categories,
        'storage_info': {
            'database_size_mb': db_size / (1024 * 1024),
            'total_csv_size_mb': total_csv_size / (1024 * 1024),
            'migrated_csv_size_mb': migrated_size / (1024 * 1024),
            'storage_efficiency': (migrated_size - db_size) / migrated_size * 100 if migrated_size > 0 else 0
        }
    }

    return summary


def print_database_summary():
    """Print formatted database summary report."""
    summary = create_database_summary_report()

    print("=" * 60)
    print(" TECHNOVA HR DATABASE SUMMARY")
    print("=" * 60)

    print(f"\n Database Tables:")
    for table_name, count in summary['database_tables'].items():
        print(f"   • {table_name}: {count:,} records")

    print(f"\nStorage Efficiency:")
    storage = summary['storage_info']
    print(f"   • Database size: {storage['database_size_mb']:.1f} MB")
    print(f"   • Original CSV files: {storage['total_csv_size_mb']:.1f} MB")
    print(f"   • Migrated data: {storage['migrated_csv_size_mb']:.1f} MB")
    if storage['storage_efficiency'] > 0:
        print(f"   • Space saved: {storage['storage_efficiency']:.1f}%")

    print(f"\nFile Organization:")
    for category, files in summary['file_organization'].items():
        category_name = category.replace('_', ' ').title()
        print(f"   {category_name}:")
        for file_name in files:
            print(f"     - {file_name}")

    print(f"\nTotal records in database: {summary['total_records']:,}")
    print("=" * 60)


def save_model_comparison_to_db(model_results, analysis_type="baseline"):
    """
    Save model comparison results to database instead of CSV.
    
    Parameters:
    -----------
    model_results : pandas.DataFrame
        Model comparison results
    analysis_type : str
        Type of analysis ('baseline', 'class_imbalance', 'hypertuning')
    """
    db_manager = DatabaseManager()
    table_name = f"{analysis_type}_model_results"

    # Add timestamp and analysis metadata
    model_results = model_results.copy()
    model_results['analysis_timestamp'] = datetime.now().isoformat()
    model_results['analysis_type'] = analysis_type

    db_manager.save_model_results(model_results, table_name)
    return table_name


def save_predictions_to_db(y_true, y_pred, y_pred_proba, model_name):
    """
    Save model predictions to database instead of CSV.
    
    Parameters:
    -----------
    y_true : array-like
        Actual target values
    y_pred : array-like
        Predicted values
    y_pred_proba : array-like
        Prediction probabilities
    model_name : str
        Name of the model
    """
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'predicted_proba_0': y_pred_proba[:, 0] if y_pred_proba.ndim > 1 else 1 - y_pred_proba,
        'predicted_proba_1': y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba,
        'correct_prediction': y_true == y_pred,
        'prediction_timestamp': datetime.now().isoformat()
    })

    db_manager = DatabaseManager()
    db_manager.save_model_predictions(predictions_df, model_name)
    return predictions_df


def save_feature_analysis_to_db(feature_importance_dict, feature_metadata=None):
    """
    Save feature importance analysis to database.
    
    Parameters:
    -----------
    feature_importance_dict : dict
        Dictionary with feature names as keys and importance scores as values
    feature_metadata : dict, optional
        Additional metadata about features
    """
    # Convert to DataFrame
    feature_df = pd.DataFrame([
        {'feature': feature, 'importance': importance}
        for feature, importance in feature_importance_dict.items()
    ]).sort_values('importance', ascending=False)

    # Add ranking
    feature_df['rank'] = range(1, len(feature_df) + 1)
    feature_df['analysis_timestamp'] = datetime.now().isoformat()

    # Create metadata DataFrame if provided
    metadata_df = None
    if feature_metadata:
        metadata_df = pd.DataFrame([feature_metadata])

    db_manager = DatabaseManager()
    db_manager.save_feature_analysis(feature_df, metadata_df)
    return feature_df


def load_analysis_from_db(table_name):
    """
    Load analysis results from database.
    
    Parameters:
    -----------
    table_name : str
        Name of the table to load
        
    Returns:
    --------
    pandas.DataFrame or None
        Loaded data or None if table doesn't exist
    """
    db_manager = DatabaseManager()
    return db_manager.load_model_results(table_name)


def create_analysis_summary_from_db():
    """
    Create comprehensive analysis summary from database tables.
    
    Returns:
    --------
    dict
        Summary of all analyses in database
    """
    db_manager = DatabaseManager()
    table_info = db_manager.get_table_info()

    summary = {
        'database_overview': table_info,
        'model_results': {},
        'feature_analysis': {},
        'predictions': {}
    }

    # Load different types of results
    result_tables = [name for name in table_info.keys() if 'model_results' in name]
    for table in result_tables:
        data = db_manager.load_model_results(table)
        if data is not None:
            summary['model_results'][table] = {
                'records': len(data),
                'best_model': data.loc[data['F1-Score'].idxmax()]['Model'] if 'F1-Score' in data.columns else 'N/A',
                'latest_analysis': data[
                    'analysis_timestamp'].max() if 'analysis_timestamp' in data.columns else 'Unknown'
            }

    # Load feature analysis
    feature_data = db_manager.load_model_results('feature_importance')
    if feature_data is not None:
        summary['feature_analysis'] = {
            'total_features': len(feature_data),
            'top_feature': feature_data.iloc[0]['feature'] if len(feature_data) > 0 else 'N/A',
            'analysis_date': feature_data[
                'analysis_timestamp'].max() if 'analysis_timestamp' in feature_data.columns else 'Unknown'
        }

    # Load predictions
    prediction_tables = [name for name in table_info.keys() if name.startswith('predictions_')]
    for table in prediction_tables:
        data = db_manager.load_model_results(table)
        if data is not None:
            model_name = table.replace('predictions_', '').replace('_', ' ').title()
            accuracy = (data['correct_prediction'].sum() / len(
                data)) * 100 if 'correct_prediction' in data.columns else 0
            summary['predictions'][model_name] = {
                'total_predictions': len(data),
                'accuracy': accuracy,
                'prediction_date': data[
                    'prediction_timestamp'].max() if 'prediction_timestamp' in data.columns else 'Unknown'
            }

    return summary


def print_analysis_summary_from_db():
    """Print formatted analysis summary from database."""
    summary = create_analysis_summary_from_db()

    print("=" * 65)
    print(" COMPREHENSIVE ANALYSIS SUMMARY FROM DATABASE")
    print("=" * 65)

    # Database overview
    print(f"\n Database Tables:")
    for table_name, count in summary['database_overview'].items():
        print(f"   • {table_name}: {count:,} records")

    # Model results
    if summary['model_results']:
        print(f"\nModel Analysis Results:")
        for table, info in summary['model_results'].items():
            analysis_type = table.replace('_model_results', '').replace('_', ' ').title()
            print(f"   {analysis_type}:")
            print(f"     - Records: {info['records']}")
            print(f"     - Best Model: {info['best_model']}")
            print(
                f"     - Latest Analysis: {info['latest_analysis'][:19] if info['latest_analysis'] != 'Unknown' else 'Unknown'}")

    # Feature analysis
    if summary['feature_analysis']:
        print(f"\nFeature Analysis:")
        fa = summary['feature_analysis']
        print(f"   • Total Features Analyzed: {fa['total_features']}")
        print(f"   • Top Important Feature: {fa['top_feature']}")
        print(f"   • Analysis Date: {fa['analysis_date'][:19] if fa['analysis_date'] != 'Unknown' else 'Unknown'}")

    # Predictions
    if summary['predictions']:
        print(f"\nModel Predictions:")
        for model_name, info in summary['predictions'].items():
            print(f"   {model_name}:")
            print(f"     - Predictions: {info['total_predictions']:,}")
            print(f"     - Accuracy: {info['accuracy']:.2f}%")
            print(f"     - Date: {info['prediction_date'][:19] if info['prediction_date'] != 'Unknown' else 'Unknown'}")

    print(f"\nAll analysis data successfully stored in centralized database")
    print("=" * 65)


def update_existing_results_workflow():
    """
    Update existing results to use database operations instead of CSV exports.
    This function demonstrates the new workflow pattern.
    """
    print("UPDATED WORKFLOW DEMONSTRATION")
    print("=" * 45)

    print("\n📝 New Workflow Pattern:")
    print("   1. Generate analysis results (DataFrame)")
    print("   2. Save to database using save_*_to_db() functions")
    print("   3. Load from database using load_analysis_from_db()")
    print("   4. Keep only essential CSV files for raw data")

    print("\nBenefits of Database Approach:")
    print("   • Centralized storage")
    print("   • Better performance for large datasets")
    print("   • Atomic transactions (data integrity)")
    print("   • SQL querying capabilities")
    print("   • Reduced file system clutter")
    print("   • Automatic indexing and optimization")
    print("   • Easier backup and version control")

    print(f"\nExample Usage:")
    print(f"   # Instead of: results.to_csv('results.csv')")
    print(f"   # Use: save_model_comparison_to_db(results, 'baseline')")
    print(f"   ")
    print(f"   # Instead of: pd.read_csv('results.csv')")
    print(f"   # Use: load_analysis_from_db('baseline_model_results')")

    return True


# Update the existing function to make it work with or without the parameter
def create_class_distribution_plot():
    """
    Create class distribution visualization.
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object for the class distribution plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # This is a template function - actual data will be passed when called
    return fig, (ax1, ax2)


def create_employee_population_comparison_plots(X, y, features, use_log=False):
    """
    Create comprehensive comparison plots showing differences between employee populations.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features dataset
    y : pandas.Series
        Target variable
    features : list
        List of features to compare
    use_log : bool
        Whether to apply log transformation for display
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing population comparison plots
    """
    n_features = len(features)
    fig, axes = plt.subplots(n_features, 3, figsize=(18, 5 * n_features))

    if n_features == 1:
        axes = axes.reshape(1, -1)

    for i, feature in enumerate(features):
        # Get data for each class
        data_stay = X[y == 0][feature]
        data_leave = X[y == 1][feature]

        # Apply log transformation if requested
        if use_log and (data_stay > 0).all() and (data_leave > 0).all():
            data_stay_plot = np.log1p(data_stay)
            data_leave_plot = np.log1p(data_leave)
            xlabel = f'{feature} (log scale)'
        else:
            data_stay_plot = data_stay
            data_leave_plot = data_leave
            xlabel = feature

        # 1. Histogram comparison
        ax1 = axes[i, 0]
        ax1.hist(data_stay_plot, alpha=0.7, bins=30, label='Stay (Class 0)',
                 color=TECHNOVA_COLORS['primary'], density=True)
        ax1.hist(data_leave_plot, alpha=0.7, bins=30, label='Leave (Class 1)',
                 color=TECHNOVA_COLORS['secondary'], density=True)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Density')
        ax1.set_title(f'Distribution Comparison: {feature}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Box plot comparison
        ax2 = axes[i, 1]
        box_data = [data_stay_plot, data_leave_plot]
        bp = ax2.boxplot(box_data, labels=['Stay', 'Leave'], patch_artist=True)
        bp['boxes'][0].set_facecolor(TECHNOVA_COLORS['primary'])
        bp['boxes'][1].set_facecolor(TECHNOVA_COLORS['secondary'])
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_alpha(0.7)
        ax2.set_ylabel(xlabel)
        ax2.set_title(f'Box Plot: {feature}')
        ax2.grid(True, alpha=0.3)

        # 3. Violin plot with statistics
        ax3 = axes[i, 2]
        violin_parts = ax3.violinplot([data_stay_plot, data_leave_plot], positions=[0, 1], widths=0.6)

        # Color the violin parts
        colors = [TECHNOVA_COLORS['primary'], TECHNOVA_COLORS['secondary']]
        for pc, color in zip(violin_parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(['Stay', 'Leave'])
        ax3.set_ylabel(xlabel)
        ax3.set_title(f'Violin Plot: {feature}')
        ax3.grid(True, alpha=0.3)

        # Add mean markers
        stay_mean = data_stay_plot.mean()
        leave_mean = data_leave_plot.mean()
        ax3.scatter([0, 1], [stay_mean, leave_mean], color='red', s=100, zorder=3, marker='D')

        # Add statistical annotation
        stats_text = f'Stay: μ={stay_mean:.2f}\nLeave: μ={leave_mean:.2f}\nΔ={abs(leave_mean - stay_mean):.2f}'
        ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

    plt.tight_layout()
    return fig


def create_feature_distribution_comparison(X, y, features, use_log=False):
    """
    Create comparison plots showing feature distributions between the two employee groups.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features dataset
    y : pandas.Series  
        Target variable
    features : list
        List of features to plot
    use_log : bool
        Whether to apply log transformation for display
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the comparison plots
    """
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, feature in enumerate(features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Get data for each class
        data_stay = X[y == 0][feature]
        data_leave = X[y == 1][feature]

        # Apply log transformation if requested
        if use_log and (data_stay > 0).all() and (data_leave > 0).all():
            data_stay = np.log1p(data_stay)
            data_leave = np.log1p(data_leave)
            xlabel = f'{feature} (log scale)'
        else:
            xlabel = feature

        # Create histogram comparison with improved styling
        ax.hist(data_stay, alpha=0.7, bins=30, label='Stay (Class 0)',
                color=TECHNOVA_COLORS['primary'], density=True)
        ax.hist(data_leave, alpha=0.7, bins=30, label='Leave (Class 1)',
                color=TECHNOVA_COLORS['secondary'], density=True)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution Comparison: {feature}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    return fig


def create_beeswarm_plot(X, y, features, use_log=False):
    """
    Create strip plots (formerly beeswarm) to show feature value distributions by class.

    Parameters:
    -----------
    X : pandas.DataFrame
        Features dataset
    y : pandas.Series
        Target variable
    features : list
        List of features to plot
    use_log : bool
        Whether to apply log transformation for display

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plots
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features))

    if n_features == 1:
        axes = [axes]

    # Combined dataset
    df_plot = X[features].copy()
    df_plot['target'] = y
    df_plot['target_label'] = y.map({0: 'Stay', 1: 'Leave'})

    for i, feature in enumerate(features):
        ax = axes[i]

        # Log transform if needed
        plot_data = df_plot.copy()
        if use_log and (plot_data[feature] > 0).all():
            plot_data[f'{feature}_display'] = np.log1p(plot_data[feature])
            plot_feature = f'{feature}_display'
            xlabel = f'{feature} (log scale)'
        else:
            plot_feature = feature
            xlabel = feature

        # Use stripplot instead of swarmplot
        sns.stripplot(
            data=plot_data,
            x=plot_feature,
            y='target_label',
            hue='target_label',
            dodge=True,
            jitter=0.25,
            size=3,
            alpha=0.6,
            palette=[TECHNOVA_COLORS['primary'], TECHNOVA_COLORS['secondary']],
            ax=ax
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Employee Group')
        ax.set_title(f'Strip Plot: {feature} by Employee Turnover Status')
        ax.grid(True, alpha=0.3)
        ax.legend_.remove()  # Remove legend for each subplot

    plt.tight_layout()
    return fig


def create_shapley_analysis_plots(model, X_sample, feature_names, max_display=15):
    """
    Create SHAP (Shapley) analysis plots for model interpretability.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model (should be tree-based for TreeExplainer)
    X_sample : pandas.DataFrame
        Sample of features for SHAP analysis
    feature_names : list
        Names of features
    max_display : int
        Maximum number of features to display
        
    Returns:
    --------
    tuple
        Tuple containing (shap_values, fig_summary, fig_bar)
    """
    # Initialize SHAP explainer
    if hasattr(model, 'predict_proba'):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, use positive class SHAP values
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    else:
        shap_values_pos = shap_values

    # Create summary plot
    fig_summary = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_pos, X_sample, max_display=max_display, show=False)
    plt.title('SHAP Summary Plot - Feature Impact on Turnover Prediction')
    plt.tight_layout()

    # Create feature importance plot
    fig_bar = plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values_pos, X_sample, plot_type="bar", max_display=max_display, show=False)
    plt.title('SHAP Feature Importance - Average Impact on Model Output')
    plt.tight_layout()

    return shap_values_pos, fig_summary, fig_bar


def create_class_imbalance_comprehensive_plot(X, y, most_imbalanced_features=None):
    """
    Create comprehensive class imbalance visualization including the most imbalanced features.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features dataset
    y : pandas.Series
        Target variable
    most_imbalanced_features : pandas.DataFrame, optional
        DataFrame containing most imbalanced features information.
        If None, will be calculated automatically.
        
    Returns:
    --------
    matplotlib.figure.Figure
        Comprehensive class imbalance visualization
    """
    # Calculate most imbalanced features if not provided
    if most_imbalanced_features is None:
        # Create a temporary processor to calculate imbalanced features
        temp_processor = HRDataProcessor(X, y)
        most_imbalanced_features = temp_processor.identify_most_imbalanced_features(top_n=12)

    fig = plt.figure(figsize=(20, 14))

    # Create subplots
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 2, 2], width_ratios=[1, 1, 1, 1])

    # 1. Overall class distribution (pie chart)
    ax1 = fig.add_subplot(gs[0, :2])
    class_counts = y.value_counts()
    colors = [TECHNOVA_COLORS['primary'], TECHNOVA_COLORS['secondary']]
    wedges, texts, autotexts = ax1.pie(class_counts.values, labels=['Stay (0)', 'Leave (1)'],
                                       colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Overall Class Distribution', fontsize=14, fontweight='bold')

    # 2. Class imbalance ratio
    ax2 = fig.add_subplot(gs[0, 2:])
    imbalance_ratio = class_counts[0] / class_counts[1]
    ax2.bar(['Imbalance Ratio'], [imbalance_ratio], color=TECHNOVA_COLORS['accent'], alpha=0.7)
    ax2.set_ylabel('Ratio (Stay:Leave)')
    ax2.set_title(f'Class Imbalance Ratio: {imbalance_ratio:.2f}:1', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Sample counts
    ax3 = fig.add_subplot(gs[1, :2])
    bars = ax3.bar(['Stay (Class 0)', 'Leave (Class 1)'], class_counts.values,
                   color=colors, alpha=0.8)
    ax3.set_ylabel('Number of Employees')
    ax3.set_title('Employee Count by Class', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 10,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    # 4. Class percentage breakdown
    ax4 = fig.add_subplot(gs[1, 2:])
    percentages = [class_counts[0] / len(y) * 100, class_counts[1] / len(y) * 100]
    bars2 = ax4.bar(['Stay (%)', 'Leave (%)'], percentages,
                    color=colors, alpha=0.8)
    ax4.set_ylabel('Percentage (%)')
    ax4.set_title('Class Distribution Percentage', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for bar, pct in zip(bars2, percentages):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 5. Top imbalanced features bar plot
    ax5 = fig.add_subplot(gs[2, :])
    top_features = most_imbalanced_features.head(12)
    bars = ax5.barh(range(len(top_features)), top_features['relative_difference_pct'],
                    color=TECHNOVA_COLORS['accent'], alpha=0.8)
    ax5.set_yticks(range(len(top_features)))
    ax5.set_yticklabels(top_features['feature'])
    ax5.set_xlabel('Relative Difference Between Classes (%)')
    ax5.set_title('Top 12 Most Imbalanced Features', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax5.text(width + 1, bar.get_y() + bar.get_height() / 2,
                 f'{width:.1f}%', ha='left', va='center', fontweight='bold', fontsize=9)

    # 6. Distribution plots for top 4 most imbalanced features
    top_4_features = most_imbalanced_features.head(4)
    for i, (_, row) in enumerate(top_4_features.iterrows()):
        feature = row['feature']
        ax = fig.add_subplot(gs[3, i])

        if feature in X.columns:
            data_stay = X[y == 0][feature]
            data_leave = X[y == 1][feature]

            # Create violin plots for better distribution visualization
            violin_parts = ax.violinplot([data_stay, data_leave], positions=[0, 1], widths=0.6)

            # Color the violin plots
            for pc, color in zip(violin_parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)

            # Add box plot overlay
            box_data = [data_stay, data_leave]
            bp = ax.boxplot(box_data, positions=[0, 1], widths=0.3, patch_artist=True)
            bp['boxes'][0].set_facecolor(colors[0])
            bp['boxes'][1].set_facecolor(colors[1])
            bp['boxes'][0].set_alpha(0.8)
            bp['boxes'][1].set_alpha(0.8)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Stay', 'Leave'])
            ax.set_title(f'{feature}\n(Diff: {row["relative_difference_pct"]:.1f}%)',
                         fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_advanced_beeswarm_plot(X, y, features, use_log=False, sample_size=1000):
    """
    Create enhanced beeswarm plots with statistical annotations.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features dataset
    y : pandas.Series
        Target variable
    features : list
        List of features to plot
    use_log : bool
        Whether to apply log transformation for display
    sample_size : int
        Maximum number of points to display for performance
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing enhanced beeswarm plots
    """
    # Sample data if needed
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[idx]
        y_sample = y.iloc[idx]
    else:
        X_sample = X
        y_sample = y

    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=(14, 5 * n_features))

    if n_features == 1:
        axes = [axes]

    # Create combined dataset
    df_plot = X_sample[features].copy()
    df_plot['target'] = y_sample
    df_plot['target_label'] = y_sample.map({0: 'Stay', 1: 'Leave'})

    for i, feature in enumerate(features):
        ax = axes[i]

        # Apply log transformation if requested
        if use_log and (df_plot[feature] > 0).all():
            plot_data = df_plot.copy()
            plot_data[f'{feature}_display'] = np.log1p(plot_data[feature])
            plot_feature = f'{feature}_display'
            xlabel = f'{feature} (log scale)'
        else:
            plot_data = df_plot.copy()
            plot_feature = feature
            xlabel = feature

        # Create enhanced beeswarm plot
        sns.swarmplot(data=plot_data, x=plot_feature, y='target_label',
                      alpha=0.6, size=4, ax=ax, palette=[TECHNOVA_COLORS['primary'], TECHNOVA_COLORS['secondary']])

        # Add statistical annotations
        stay_data = plot_data[plot_data['target_label'] == 'Stay'][plot_feature]
        leave_data = plot_data[plot_data['target_label'] == 'Leave'][plot_feature]

        # Calculate statistics
        stay_mean = stay_data.mean()
        leave_mean = leave_data.mean()
        stay_median = stay_data.median()
        leave_median = leave_data.median()

        # Add mean lines
        ax.axvline(stay_mean, color=TECHNOVA_COLORS['primary'], linestyle='--', alpha=0.8, linewidth=2)
        ax.axvline(leave_mean, color=TECHNOVA_COLORS['secondary'], linestyle='--', alpha=0.8, linewidth=2)

        # Add statistics text box
        stats_text = f'Stay: μ={stay_mean:.2f}, median={stay_median:.2f}\nLeave: μ={leave_mean:.2f}, median={leave_median:.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Employee Group')
        ax.set_title(f'Enhanced Beeswarm Plot: {feature} by Turnover Status')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_shapley_analysis_plots(model, X_sample, feature_names, max_display=15):
    """
    Create SHAP (Shapley) analysis plots for model interpretability.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model (should be tree-based for TreeExplainer)
    X_sample : pandas.DataFrame
        Sample of features for SHAP analysis
    feature_names : list
        Names of features
    max_display : int
        Maximum number of features to display
        
    Returns:
    --------
    tuple
        Tuple containing (shap_values, fig_summary, fig_bar)
    """
    # Initialize SHAP explainer
    if hasattr(model, 'predict_proba'):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, use positive class SHAP values
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    else:
        shap_values_pos = shap_values

    # Create summary plot
    fig_summary = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_pos, X_sample, max_display=max_display, show=False)
    plt.title('SHAP Summary Plot - Feature Impact on Turnover Prediction')
    plt.tight_layout()

    # Create feature importance plot
    fig_bar = plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values_pos, X_sample, plot_type="bar", max_display=max_display, show=False)
    plt.title('SHAP Feature Importance - Average Impact on Model Output')
    plt.tight_layout()

    return shap_values_pos, fig_summary, fig_bar


def create_log_transformation_comparison(X, y, features, sample_size=None):
    """
    Create side-by-side comparison of original vs log-transformed feature distributions.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features dataset
    y : pandas.Series
        Target variable
    features : list
        List of features to compare
    sample_size : int, optional
        Sample size for plotting (for performance)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure showing original vs log-transformed distributions
    """
    if sample_size and len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[idx]
        y_sample = y.iloc[idx]
    else:
        X_sample = X
        y_sample = y

    n_features = len(features)
    fig, axes = plt.subplots(n_features, 2, figsize=(16, 5 * n_features))

    if n_features == 1:
        axes = axes.reshape(1, -1)

    for i, feature in enumerate(features):
        if feature not in X_sample.columns:
            continue

        # Check if feature can be log-transformed
        if not (X_sample[feature] > 0).all():
            print(f"Skipping {feature} - contains non-positive values")
            continue

        # Original distribution
        ax_orig = axes[i, 0]
        data_stay_orig = X_sample[y_sample == 0][feature]
        data_leave_orig = X_sample[y_sample == 1][feature]

        ax_orig.hist(data_stay_orig, alpha=0.7, bins=30, label='Stay (Class 0)',
                     color='skyblue', density=True)
        ax_orig.hist(data_leave_orig, alpha=0.7, bins=30, label='Leave (Class 1)',
                     color='salmon', density=True)
        ax_orig.set_xlabel(feature)
        ax_orig.set_ylabel('Density')
        ax_orig.set_title(f'Original: {feature}')
        ax_orig.legend()
        ax_orig.grid(True, alpha=0.3)

        # Log-transformed distribution
        ax_log = axes[i, 1]
        data_stay_log = np.log1p(data_stay_orig)
        data_leave_log = np.log1p(data_leave_orig)

        ax_log.hist(data_stay_log, alpha=0.7, bins=30, label='Stay (Class 0)',
                    color='skyblue', density=True)
        ax_log.hist(data_leave_log, alpha=0.7, bins=30, label='Leave (Class 1)',
                    color='salmon', density=True)
        ax_log.set_xlabel(f'{feature} (log transformed)')
        ax_log.set_ylabel('Density')
        ax_log.set_title(f'Log-transformed: {feature}')
        ax_log.legend()
        ax_log.grid(True, alpha=0.3)

        # Add statistics
        orig_skew_stay = data_stay_orig.skew()
        orig_skew_leave = data_leave_orig.skew()
        log_skew_stay = data_stay_log.skew()
        log_skew_leave = data_leave_log.skew()

        ax_orig.text(0.02, 0.98, f'Skewness:\nStay: {orig_skew_stay:.2f}\nLeave: {orig_skew_leave:.2f}',
                     transform=ax_orig.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax_log.text(0.02, 0.98, f'Skewness:\nStay: {log_skew_stay:.2f}\nLeave: {log_skew_leave:.2f}',
                    transform=ax_log.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


def create_shapley_analysis_plots(model, X_test, features, max_samples=200):
    """
    Create comprehensive SHAP analysis plots including summary and bar plots.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model for SHAP analysis
    X_test : pandas.DataFrame
        Test features
    features : list
        Feature names for analysis
    max_samples : int, optional
        Maximum samples for SHAP analysis
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with SHAP analysis plots
    """
    import shap

    # Limit samples for performance
    if len(X_test) > max_samples:
        X_shap = X_test.sample(n=max_samples, random_state=42)
    else:
        X_shap = X_test

    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # Handle binary classification - use positive class
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1]  # Positive class (Leave)
    else:
        shap_values_plot = shap_values

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # SHAP Summary Plot (beeswarm)
    plt.subplot(2, 2, 1)
    shap.summary_plot(shap_values_plot, X_shap, plot_type="dot",
                      max_display=len(features), show=False,
                      color_bar_label="Feature Value")
    plt.title('SHAP Beeswarm Plot\n(Feature Impact Distribution)',
              fontsize=14, fontweight='bold', pad=20)

    # SHAP Bar Plot (mean absolute values)
    plt.subplot(2, 2, 2)
    shap.summary_plot(shap_values_plot, X_shap, plot_type="bar",
                      max_display=len(features), show=False)
    plt.title(' SHAP Feature Importance\n(Mean |SHAP Value|)',
              fontsize=14, fontweight='bold', pad=20)

    # SHAP Waterfall Plot (individual prediction)
    plt.subplot(2, 2, 3)
    # Select a random individual for waterfall plot
    individual_idx = np.random.randint(0, len(X_shap))
    
    # Fix for newer SHAP versions - handle binary classification properly
    if isinstance(shap_values_plot, np.ndarray) and len(shap_values_plot.shape) > 1:
        # For binary classification, use only the positive class values
        individual_shap = shap_values_plot[individual_idx]
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
    else:
        individual_shap = shap_values_plot[individual_idx]
        base_value = explainer.expected_value
    
    try:
        # Create a proper SHAP Explanation object for the waterfall plot
        explanation = shap.Explanation(
            values=individual_shap,
            base_values=base_value,
            data=X_shap.iloc[individual_idx].values,
            feature_names=X_shap.columns.tolist()
        )
        
        # Use the newer SHAP API directly
        shap.plots.waterfall(explanation, max_display=8, show=False)
        
    except Exception as e:
        print(f"SHAP waterfall plot failed: {e}")
        try:
            # Alternative: Use older waterfall_plot function
            if hasattr(shap, 'waterfall_plot'):
                shap.waterfall_plot(
                    base_value, individual_shap, X_shap.iloc[individual_idx],
                    feature_names=X_shap.columns.tolist(), max_display=8, show=False
                )
            else:
                raise Exception("No waterfall plot function available")
                
        except Exception as e2:
            print(f"Alternative waterfall plot also failed: {e2}")
            # Ultimate fallback: create a simple bar plot
            n_features_to_show = min(8, len(individual_shap))
            colors = ['red' if val > 0 else 'blue' for val in individual_shap[:n_features_to_show]]
            
            plt.barh(range(n_features_to_show), 
                     individual_shap[:n_features_to_show],
                     color=colors)
            plt.yticks(range(n_features_to_show), 
                       X_shap.columns[:n_features_to_show])
            plt.xlabel('SHAP Value')
            plt.title('Feature Impact (Individual)\nFallback Visualization', fontsize=12)
    plt.title('SHAP Waterfall Plot\n(Individual Prediction)',
              fontsize=14, fontweight='bold', pad=20)

    # SHAP Force Plot (sample of individuals)
    plt.subplot(2, 2, 4)
    # Create a simplified force plot representation
    feature_importance = np.mean(np.abs(shap_values_plot), axis=0)
    top_features_idx = np.argsort(feature_importance)[-8:]  # Top 8 features

    plt.barh(range(len(top_features_idx)),
             feature_importance[top_features_idx],
             color=[TECHNOVA_COLORS['primary'] if val > 0 else TECHNOVA_COLORS['secondary']
                    for val in feature_importance[top_features_idx]])
    plt.yticks(range(len(top_features_idx)),
               [X_shap.columns[i] for i in top_features_idx])
    plt.xlabel('Mean |SHAP Value|')
    plt.title('Top Feature Impact\n(Most Influential Features)',
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)

    plt.suptitle('SHAPLEY VALUE ANALYSIS - Model Interpretability',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    return shap_values_plot, fig,


def prepare_model_for_shap(X, y, test_size=0.2, random_state=42):
    """
    Prepare a trained model for SHAP analysis.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features dataset
    y : pandas.Series
        Target variable
    test_size : float
        Test set size
    random_state : int
        Random seed
        
    Returns:
    --------
    tuple
        (trained_model, X_test_sample, y_test_sample)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train a simple Random Forest for SHAP analysis
    model = RandomForestClassifier(n_estimators=100, random_state=random_state,
                                   class_weight='balanced')
    model.fit(X_train, y_train)

    # Take a smaller sample for SHAP (for performance)
    sample_size = min(100, len(X_test))
    if len(X_test) > sample_size:
        idx = np.random.choice(len(X_test), sample_size, replace=False)
        X_test_sample = X_test.iloc[idx]
        y_test_sample = y_test.iloc[idx]
    else:
        X_test_sample = X_test
        y_test_sample = y_test

    return model, X_test_sample, y_test_sample


def calculate_feature_importance_stats(X, y):
    """
    Calculate feature importance using multiple methods.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features dataset
    y : pandas.Series
        Target variable
        
    Returns:
    --------
    pandas.DataFrame
        Feature importance statistics
    """
    # Train Random Forest to get feature importances
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y)

    # Get feature importances
    importances = model.feature_importances_

    # Calculate correlation with target
    correlations = []
    for feature in X.columns:
        if X[feature].dtype in ['int64', 'float64']:
            corr = X[feature].corr(y)
            correlations.append(abs(corr))
        else:
            correlations.append(0)

    # Create results DataFrame
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'rf_importance': importances,
        'abs_correlation': correlations,
        'combined_score': (importances + np.array(correlations)) / 2
    })

    return importance_df.sort_values('combined_score', ascending=False)


# Additional utility functions for data validation and preprocessing

def validate_data_quality(X, y):
    """
    Validate data quality and return quality report.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features dataset
    y : pandas.Series
        Target variable
        
    Returns:
    --------
    dict
        Data quality report
    """
    report = {
        'total_samples': len(X),
        'total_features': len(X.columns),
        'missing_values': X.isnull().sum().sum(),
        'duplicate_rows': X.duplicated().sum(),
        'target_missing': y.isnull().sum(),
        'numerical_features': len(X.select_dtypes(include=[np.number]).columns),
        'categorical_features': len(X.select_dtypes(exclude=[np.number]).columns),
        'class_distribution': y.value_counts().to_dict(),
        'highly_correlated_pairs': [],
        'constant_features': [],
        'high_cardinality_features': []
    }

    # Check for highly correlated features
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        corr_matrix = X[numerical_cols].corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.9:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        report['highly_correlated_pairs'] = high_corr_pairs

    # Check for constant features
    for col in X.columns:
        if X[col].nunique() <= 1:
            report['constant_features'].append(col)

    # Check for high cardinality features
    for col in X.columns:
        if X[col].dtype == 'object' and X[col].nunique() > 50:
            report['high_cardinality_features'].append(col)

    return report


def print_data_quality_report(report):
    """
    Print formatted data quality report.
    
    Parameters:
    -----------
    report : dict
        Data quality report from validate_data_quality
    """
    print("=" * 50)
    print("DATA QUALITY REPORT")
    print("=" * 50)

    print(f" Dataset Overview:")
    print(f"   Total Samples: {report['total_samples']:,}")
    print(f"   Total Features: {report['total_features']:,}")
    print(f"   Numerical Features: {report['numerical_features']}")
    print(f"   Categorical Features: {report['categorical_features']}")

    print(f"\nTarget Variable:")
    for class_val, count in report['class_distribution'].items():
        pct = (count / report['total_samples']) * 100
        print(f"   Class {class_val}: {count:,} samples ({pct:.1f}%)")

    print(f"\n Data Quality Issues:")
    print(f"   Missing Values: {report['missing_values']:,}")
    print(f"   Duplicate Rows: {report['duplicate_rows']:,}")
    print(f"   Target Missing: {report['target_missing']:,}")
    print(f"   Constant Features: {len(report['constant_features'])}")
    print(f"   High Cardinality Features: {len(report['high_cardinality_features'])}")
    print(f"   Highly Correlated Pairs: {len(report['highly_correlated_pairs'])}")

    if report['constant_features']:
        print(f"\nConstant Features:")
        for feature in report['constant_features'][:5]:  # Show first 5
            print(f"   - {feature}")
        if len(report['constant_features']) > 5:
            print(f"   ... and {len(report['constant_features']) - 5} more")

    if report['highly_correlated_pairs']:
        print(f"\nHighly Correlated Feature Pairs (|r| > 0.9):")
        for feature1, feature2, corr in report['highly_correlated_pairs'][:3]:  # Show first 3
            print(f"   - {feature1} ↔ {feature2}: {corr:.3f}")
        if len(report['highly_correlated_pairs']) > 3:
            print(f"   ... and {len(report['highly_correlated_pairs']) - 3} more pairs")

    print("=" * 50)


def load_modeling_data_from_db(db_path=None):
    """
    Load features (X) and target (y) from database for modeling.
    
    Parameters:
    -----------
    db_path : str, optional
        Path to the SQLite database. If None, uses robust path detection.
        
    Returns:
    --------
    tuple
        (X, y) - Features DataFrame and target Series
    """
    if db_path is None:
        db_path = get_database_path()

    try:
        with sqlite3.connect(db_path) as conn:
            # Try different table names for compatibility
            try:
                # First try modern table names
                X = pd.read_sql_query("SELECT * FROM features_X", conn)
                y = pd.read_sql_query("SELECT * FROM target_y", conn)
                y = y.squeeze()  # Convert to Series
            except sqlite3.OperationalError:
                # Fallback to legacy table names
                X = pd.read_sql_query("SELECT * FROM features", conn)
                y = pd.read_sql_query("SELECT * FROM target", conn)
                y = y.squeeze()  # Convert to Series

            print(f"Data loaded from database:")
            print(f"   Database: {db_path}")
            print(f"   Features shape: {X.shape}")
            print(f"   Target shape: {y.shape}")
            print(f"   Target distribution: {y.value_counts().to_dict()}")
            print(f"   Turnover rate: {y.mean():.2%}")

            return X, y

    except sqlite3.OperationalError as e:
        print(f"Error loading data from database: {e}")
        print("   Make sure notebook 2 (feature engineering) has been executed to create the database tables.")
        return None, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None


def load_previous_model_results(table_name, db_path=None):
    """
    Load previous model results from database.
    
    Parameters:
    -----------
    table_name : str
        Name of the results table to load
    db_path : str, optional
        Path to the SQLite database. If None, uses robust path detection.
        
    Returns:
    --------
    pandas.DataFrame or None
        Previous model results or None if not found
    """
    if db_path is None:
        db_path = get_database_path()

    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            print(f"Loaded {len(df)} records from table '{table_name}'")
            return df

    except sqlite3.OperationalError:
        print(f"Table '{table_name}' not found in database")
        print("   This is expected if this is the first run of this analysis phase.")
        return None
    except Exception as e:
        print(f"Error loading table '{table_name}': {e}")
        return None


def save_model_results_to_db(results_df, table_name, db_path=None):
    """
    Save model results to database, replacing any existing table.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results DataFrame to save
    table_name : str
        Name of the table to create/replace
    db_path : str, optional
        Path to the SQLite database. If None, uses robust path detection.
    """
    if db_path is None:
        db_path = get_database_path()

    try:
        with sqlite3.connect(db_path) as conn:
            # Add timestamp
            results_df = results_df.copy()
            results_df['saved_timestamp'] = datetime.now().isoformat()

            # Save to database, replacing existing table
            results_df.to_sql(table_name, conn, index=False, if_exists='replace')

            print(f"Saved {len(results_df)} records to table '{table_name}'")

    except Exception as e:
        print(f"Error saving to database: {e}")


def save_feature_importance_to_db(feature_importance_df, table_name, db_path=None):
    """
    Save feature importance analysis to database.
    
    Parameters:
    -----------
    feature_importance_df : pandas.DataFrame
        Feature importance DataFrame
    table_name : str
        Name of the table to create/replace
    db_path : str, optional
        Path to the SQLite database. If None, uses robust path detection.
    """
    if db_path is None:
        db_path = get_database_path()

    try:
        with sqlite3.connect(db_path) as conn:
            # Add metadata
            feature_importance_df = feature_importance_df.copy()
            feature_importance_df['analysis_timestamp'] = datetime.now().isoformat()

            # Save to database
            feature_importance_df.to_sql(table_name, conn, index=False, if_exists='replace')

            print(f"Saved feature importance analysis to table '{table_name}'")

    except Exception as e:
        print(f"Error saving feature importance: {e}")


def get_database_tables(db_path=DB_PATH):
    """
    Get list of all tables in the database.
    
    Parameters:
    -----------
    db_path : str
        Path to the SQLite database
        
    Returns:
    --------
    list
        List of table names
    """
    try:
        with sqlite3.connect(db_path) as conn:
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", conn
            )
            return tables['name'].tolist()

    except Exception as e:
        print(f"Error getting database tables: {e}")
        return []


def print_database_status(db_path=DB_PATH):
    """
    Print current database status and available tables.
    
    Parameters:
    -----------
    db_path : str
        Path to the SQLite database
    """
    print("DATABASE STATUS")
    print("=" * 30)

    if not os.path.exists(db_path):
        print("Database file does not exist")
        print("Please run notebook 2 (feature engineering) first to create the database.")
        return

    tables = get_database_tables(db_path)

    if not tables:
        print("No tables found in database")
        return

    try:
        with sqlite3.connect(db_path) as conn:
            print(f"Available tables:")

            for table in sorted(tables):
                count_query = f"SELECT COUNT(*) as count FROM {table}"
                count_result = pd.read_sql_query(count_query, conn)
                count = count_result['count'].iloc[0]
                print(f"   • {table}: {count:,} records")

        # Database file size
        db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
        print(f"\nDatabase size: {db_size_mb:.1f} MB")

    except Exception as e:
        print(f"Error reading database status: {e}")

    print("=" * 30)
