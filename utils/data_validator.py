# ========================================
# FILE 5: utils/data_validator.py
# ========================================
"""
Data Validation and Cleaning Utilities
Handles data quality checks and cleaning operations.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Any


def validate_and_clean_data(df, dataset_name):
    """
    Validate and clean a dataset with comprehensive checks.

    Args:
        df (pd.DataFrame): Dataset to validate and clean
        dataset_name (str): Name of the dataset for logging

    Returns:
        tuple: (cleaned_df, validation_report)
    """
    validation_report = {
        'dataset_name': dataset_name,
        'original_rows': len(df),
        'original_columns': len(df.columns),
        'issues_found': [],
        'cleaning_actions': [],
        'final_rows': 0,
        'final_columns': 0
    }

    cleaned_df = df.copy()

    # 1. Check for completely empty rows
    empty_rows = cleaned_df.isnull().all(axis=1).sum()
    if empty_rows > 0:
        cleaned_df = cleaned_df.dropna(how='all')
        validation_report['issues_found'].append(f"Found {empty_rows} completely empty rows")
        validation_report['cleaning_actions'].append(f"Removed {empty_rows} empty rows")

    # 2. Check for duplicate rows
    duplicates = cleaned_df.duplicated().sum()
    if duplicates > 0:
        cleaned_df = cleaned_df.drop_duplicates()
        validation_report['issues_found'].append(f"Found {duplicates} duplicate rows")
        validation_report['cleaning_actions'].append(f"Removed {duplicates} duplicate rows")

    # 3. Clean column names
    original_columns = cleaned_df.columns.tolist()
    cleaned_df.columns = [clean_column_name(col) for col in cleaned_df.columns]

    if original_columns != cleaned_df.columns.tolist():
        validation_report['cleaning_actions'].append(
            "Cleaned column names (removed special characters, normalized spacing)")

    # 4. Check for columns with all null values
    null_columns = []
    for col in cleaned_df.columns:
        if cleaned_df[col].isnull().all():
            null_columns.append(col)

    if null_columns:
        cleaned_df = cleaned_df.drop(columns=null_columns)
        validation_report['issues_found'].append(
            f"Found {len(null_columns)} columns with all null values: {null_columns}")
        validation_report['cleaning_actions'].append(f"Removed {len(null_columns)} null columns")

    # 5. Validate specific columns based on expected patterns
    if 'spotify_id' in cleaned_df.columns:
        spotify_issues = validate_spotify_ids(cleaned_df['spotify_id'])
        if spotify_issues['invalid_count'] > 0:
            validation_report['issues_found'].append(
                f"Found {spotify_issues['invalid_count']} invalid Spotify IDs"
            )

    if 'track_id' in cleaned_df.columns:
        track_issues = validate_spotify_ids(cleaned_df['track_id'])
        if track_issues['invalid_count'] > 0:
            validation_report['issues_found'].append(
                f"Found {track_issues['invalid_count']} invalid track IDs"
            )

    # 6. Check for suspicious numeric values
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        outliers = detect_outliers(cleaned_df[col])
        if outliers['count'] > 0:
            validation_report['issues_found'].append(
                f"Column '{col}': {outliers['count']} potential outliers detected"
            )

    # 7. Check text columns for encoding issues
    text_columns = cleaned_df.select_dtypes(include=['object']).columns
    for col in text_columns:
        encoding_issues = detect_encoding_issues(cleaned_df[col])
        if encoding_issues > 0:
            validation_report['issues_found'].append(
                f"Column '{col}': {encoding_issues} rows with potential encoding issues"
            )

    # Update final counts
    validation_report['final_rows'] = len(cleaned_df)
    validation_report['final_columns'] = len(cleaned_df.columns)

    return cleaned_df, validation_report


def clean_column_name(column_name):
    """
    Clean and normalize a column name.

    Args:
        column_name (str): Original column name

    Returns:
        str: Cleaned column name
    """
    if not isinstance(column_name, str):
        return str(column_name)

    # Convert to lowercase
    cleaned = column_name.lower()

    # Replace spaces and special characters with underscores
    cleaned = re.sub(r'[^\w\s]', '_', cleaned)
    cleaned = re.sub(r'\s+', '_', cleaned)

    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)

    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')

    # Ensure it doesn't start with a number
    if cleaned and cleaned[0].isdigit():
        cleaned = 'col_' + cleaned

    return cleaned


def validate_spotify_ids(id_series):
    """
    Validate a series of Spotify IDs.

    Args:
        id_series (pd.Series): Series containing Spotify IDs

    Returns:
        dict: Validation results
    """
    results = {
        'total_count': len(id_series),
        'null_count': id_series.isnull().sum(),
        'valid_count': 0,
        'invalid_count': 0,
        'invalid_examples': []
    }

    # Check non-null values
    non_null_ids = id_series.dropna()

    for spotify_id in non_null_ids:
        if is_valid_spotify_id(str(spotify_id)):
            results['valid_count'] += 1
        else:
            results['invalid_count'] += 1
            if len(results['invalid_examples']) < 5:  # Store first 5 examples
                results['invalid_examples'].append(str(spotify_id))

    return results


def is_valid_spotify_id(spotify_id):
    """
    Check if a string is a valid Spotify ID format.

    Args:
        spotify_id (str): ID to validate

    Returns:
        bool: True if valid format, False otherwise
    """
    if not isinstance(spotify_id, str):
        return False

    # Remove common prefixes for validation
    cleaned_id = spotify_id.strip().lower()

    prefixes = [
        'https://open.spotify.com/track/',
        'http://open.spotify.com/track/',
        'spotify:track:',
        'spotify:'
    ]

    for prefix in prefixes:
        if cleaned_id.startswith(prefix):
            cleaned_id = cleaned_id[len(prefix):]
            break

    # Remove query parameters
    if '?' in cleaned_id:
        cleaned_id = cleaned_id.split('?')[0]

    # Check format: should be 22 alphanumeric characters
    return bool(re.match(r'^[a-zA-Z0-9]{22}$', cleaned_id))


def detect_outliers(numeric_series, method='iqr', threshold=3):
    """
    Detect outliers in a numeric series.

    Args:
        numeric_series (pd.Series): Numeric data to check
        method (str): Method to use ('iqr' or 'zscore')
        threshold (float): Threshold for outlier detection

    Returns:
        dict: Outlier detection results
    """
    results = {
        'count': 0,
        'indices': [],
        'values': [],
        'method': method
    }

    # Remove null values
    clean_series = numeric_series.dropna()

    if len(clean_series) == 0:
        return results

    if method == 'iqr':
        Q1 = clean_series.quantile(0.25)
        Q3 = clean_series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_mask = (clean_series < lower_bound) | (clean_series > upper_bound)

    elif method == 'zscore':
        z_scores = np.abs((clean_series - clean_series.mean()) / clean_series.std())
        outliers_mask = z_scores > threshold

    else:
        return results

    outlier_indices = clean_series[outliers_mask].index.tolist()
    outlier_values = clean_series[outliers_mask].tolist()

    results['count'] = len(outlier_indices)
    results['indices'] = outlier_indices[:10]  # Limit to first 10
    results['values'] = outlier_values[:10]  # Limit to first 10

    return results


def detect_encoding_issues(text_series):
    """
    Detect potential encoding issues in text data.

    Args:
        text_series (pd.Series): Text data to check

    Returns:
        int: Number of rows with potential encoding issues
    """
    issue_count = 0

    # Common indicators of encoding issues
    encoding_indicators = [
        '�',  # Replacement character
        'â€™',  # Curly apostrophe encoded as UTF-8 but read as Latin-1
        'â€œ',  # Left double quotation mark
        'â€\x9d',  # Right double quotation mark
        'Ã©',  # é encoded as UTF-8 but read as Latin-1
        'Ã¡',  # á encoded as UTF-8 but read as Latin-1
    ]

    for text in text_series.dropna():
        if isinstance(text, str):
            for indicator in encoding_indicators:
                if indicator in text:
                    issue_count += 1
                    break

    return issue_count


def validate_numeric_ranges(df, column_constraints):
    """
    Validate that numeric columns fall within expected ranges.

    Args:
        df (pd.DataFrame): DataFrame to validate
        column_constraints (dict): Dict mapping column names to (min, max) tuples

    Returns:
        dict: Validation results for each column
    """
    results = {}

    for column, (min_val, max_val) in column_constraints.items():
        if column not in df.columns:
            results[column] = {'status': 'column_not_found'}
            continue

        col_data = df[column].dropna()

        if len(col_data) == 0:
            results[column] = {'status': 'no_data'}
            continue

        out_of_range = ((col_data < min_val) | (col_data > max_val)).sum()

        results[column] = {
            'status': 'validated',
            'total_values': len(col_data),
            'out_of_range_count': int(out_of_range),
            'out_of_range_percentage': (out_of_range / len(col_data)) * 100,
            'min_found': float(col_data.min()),
            'max_found': float(col_data.max())
        }

    return results


def check_data_consistency(df1, df2, join_column):
    """
    Check consistency between two datasets on a join column.

    Args:
        df1 (pd.DataFrame): First dataset
        df2 (pd.DataFrame): Second dataset
        join_column (str): Column to check consistency on

    Returns:
        dict: Consistency check results
    """
    results = {
        'df1_unique_values': 0,
        'df2_unique_values': 0,
        'common_values': 0,
        'df1_only': 0,
        'df2_only': 0,
        'overlap_percentage': 0.0
    }

    if join_column not in df1.columns or join_column not in df2.columns:
        results['error'] = f"Column '{join_column}' not found in one or both datasets"
        return results

    df1_values = set(df1[join_column].dropna().astype(str))
    df2_values = set(df2[join_column].dropna().astype(str))

    common_values = df1_values & df2_values
    df1_only = df1_values - df2_values
    df2_only = df2_values - df1_values

    results['df1_unique_values'] = len(df1_values)
    results['df2_unique_values'] = len(df2_values)
    results['common_values'] = len(common_values)
    results['df1_only'] = len(df1_only)
    results['df2_only'] = len(df2_only)

    if len(df1_values) > 0:
        results['overlap_percentage'] = (len(common_values) / len(df1_values)) * 100

    return results


# Test function for development
def test_validation():
    """Test validation functions with sample data"""
    # Create test DataFrame with various issues
    test_data = pd.DataFrame({
        'spotify_id': ['4iV5W9uYEdYUVa79Axb7Rh', 'invalid_id', None, '4iV5W9uYEdYUVa79Axb7Rh'],
        'song name': ['Song 1', 'Song 2', 'Song 3', 'Song 1'],  # Space in column name
        'duration': [180, 210, 9999, 195],  # Outlier present
        'artist': ['Artist A', 'Artist B�', 'Artist C', 'Artist A'],  # Encoding issue
        'empty_col': [None, None, None, None]  # All null column
    })

    print("Testing data validation...")
    print(f"Original data shape: {test_data.shape}")

    # Test validation and cleaning
    cleaned_df, report = validate_and_clean_data(test_data, "Test Dataset")

    print(f"Cleaned data shape: {cleaned_df.shape}")
    print(f"\nValidation Report:")
    print(f"  Issues found: {len(report['issues_found'])}")
    for issue in report['issues_found']:
        print(f"    - {issue}")

    print(f"  Cleaning actions: {len(report['cleaning_actions'])}")
    for action in report['cleaning_actions']:
        print(f"    - {action}")

    # Test Spotify ID validation
    spotify_results = validate_spotify_ids(test_data['spotify_id'])
    print(f"\nSpotify ID validation:")
    print(f"  Valid: {spotify_results['valid_count']}")
    print(f"  Invalid: {spotify_results['invalid_count']}")
    print(f"  Examples of invalid: {spotify_results['invalid_examples']}")

    print("✓ All validation tests completed successfully")


if __name__ == "__main__":
    test_validation()