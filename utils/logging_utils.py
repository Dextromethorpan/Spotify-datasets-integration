"""
Logging Utilities for ETL Pipeline
Provides structured logging for different categories of events.
"""

import logging
import json
import numpy as np
from datetime import datetime
from pathlib import Path


def convert_numpy_types(obj):
    """
    Convert numpy data types to Python native types for JSON serialization.

    Args:
        obj: Object that might contain numpy types

    Returns:
        Object with numpy types converted to Python types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def setup_logger(name, log_file, level=logging.INFO):
    """
    Set up a logger with specified name and file.

    Args:
        name (str): Logger name
        log_file (str): Path to log file
        level: Logging level (default: INFO)

    Returns:
        logging.Logger: Configured logger instance
    """
    # Ensure log directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear any existing handlers to prevent duplicates
    logger.handlers.clear()

    # Add file handler
    logger.addHandler(file_handler)

    # Also add console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def log_counts(logger, operation_name, counts_dict):
    """
    Log count statistics in a structured format.

    Args:
        logger (logging.Logger): Logger instance
        operation_name (str): Name of the operation being logged
        counts_dict (dict): Dictionary of count statistics
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert numpy types to Python types for JSON serialization
    safe_counts = convert_numpy_types(counts_dict)

    log_entry = {
        'timestamp': timestamp,
        'operation': operation_name,
        'counts': safe_counts
    }

    # Log as both structured JSON and human-readable format
    logger.info(f"COUNTS - {operation_name}")
    try:
        logger.info(f"Details: {json.dumps(safe_counts, indent=2)}")
    except (TypeError, ValueError) as e:
        # Fallback if JSON serialization still fails
        logger.info(f"Details: {safe_counts}")
        logger.warning(f"JSON serialization failed: {e}")

    # Calculate percentages where applicable
    if 'total_rows' in safe_counts:
        total = safe_counts['total_rows']
        for key, value in safe_counts.items():
            if key != 'total_rows' and isinstance(value, (int, float)):
                percentage = (value / total * 100) if total > 0 else 0
                logger.info(f"  {key}: {value} ({percentage:.1f}%)")


def log_schema_collision(logger, collision_info):
    """
    Log schema collision details.

    Args:
        logger (logging.Logger): Logger instance
        collision_info (dict): Information about the schema collision
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert numpy types if present
    safe_collision_info = convert_numpy_types(collision_info)

    logger.warning(f"SCHEMA COLLISION - Column: {safe_collision_info['column']}")
    logger.warning(f"  Dataset 1 ({safe_collision_info['df1_name']}): {safe_collision_info['sample_df1_values']}")
    logger.warning(f"  Dataset 2 ({safe_collision_info['df2_name']}): {safe_collision_info['sample_df2_values']}")
    logger.warning(f"  Timestamp: {timestamp}")


def log_join_results(logger, join_operation, results):
    """
    Log the results of a join operation.

    Args:
        logger (logging.Logger): Logger instance
        join_operation (str): Description of the join operation
        results (dict): Join results including matches, failures, etc.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert numpy types
    safe_results = convert_numpy_types(results)

    logger.info(f"JOIN RESULTS - {join_operation}")
    logger.info(f"Timestamp: {timestamp}")

    for key, value in safe_results.items():
        logger.info(f"  {key}: {value}")

    # Calculate match rate if applicable
    if 'successful_matches' in safe_results and 'total_attempts' in safe_results:
        total = safe_results['total_attempts']
        matches = safe_results['successful_matches']
        match_rate = (matches / total * 100) if total > 0 else 0
        logger.info(f"  Match rate: {match_rate:.1f}%")


def log_data_quality_issues(logger, dataset_name, issues):
    """
    Log data quality issues found during processing.

    Args:
        logger (logging.Logger): Logger instance
        dataset_name (str): Name of the dataset
        issues (list): List of data quality issues
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.warning(f"DATA QUALITY ISSUES - {dataset_name}")
    logger.warning(f"Timestamp: {timestamp}")
    logger.warning(f"Total issues found: {len(issues)}")

    for issue in issues:
        logger.warning(f"  - {issue}")


def log_normalization_results(logger, normalization_type, results):
    """
    Log ID normalization results.

    Args:
        logger (logging.Logger): Logger instance
        normalization_type (str): Type of normalization performed
        results (dict): Normalization results
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert numpy types
    safe_results = convert_numpy_types(results)

    logger.info(f"NORMALIZATION - {normalization_type}")
    logger.info(f"Timestamp: {timestamp}")

    if 'total_ids' in safe_results:
        total = safe_results['total_ids']
        success = safe_results.get('successful_normalizations', 0)
        failed = safe_results.get('failed_normalizations', 0)

        success_rate = (success / total * 100) if total > 0 else 0

        logger.info(f"  Total IDs processed: {total}")
        logger.info(f"  Successful normalizations: {success} ({success_rate:.1f}%)")
        logger.info(f"  Failed normalizations: {failed}")

        # Log some examples of failed normalizations
        if 'failed_examples' in safe_results:
            logger.warning("  Failed normalization examples:")
            for example in safe_results['failed_examples'][:5]:  # Log first 5 examples
                logger.warning(f"    - '{example}'")


def log_processing_progress(logger, step_name, current, total, details=None):
    """
    Log processing progress for long-running operations.

    Args:
        logger (logging.Logger): Logger instance
        step_name (str): Name of the processing step
        current (int): Current progress count
        total (int): Total items to process
        details (str, optional): Additional details
    """
    percentage = (current / total * 100) if total > 0 else 0

    log_message = f"PROGRESS - {step_name}: {current}/{total} ({percentage:.1f}%)"
    if details:
        log_message += f" - {details}"

    logger.info(log_message)


def log_database_operation(logger, operation, db_path, table_name, row_count=None):
    """
    Log database operations.

    Args:
        logger (logging.Logger): Logger instance
        operation (str): Type of database operation
        db_path (str): Path to database file
        table_name (str): Name of table involved
        row_count (int, optional): Number of rows affected
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_message = f"DATABASE - {operation} on {table_name} in {db_path}"
    if row_count is not None:
        log_message += f" ({row_count} rows)"

    logger.info(log_message)
    logger.info(f"  Timestamp: {timestamp}")


def log_error_with_context(logger, error, context):
    """
    Log errors with additional context information.

    Args:
        logger (logging.Logger): Logger instance
        error (Exception): The error that occurred
        context (dict): Additional context information
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert numpy types in context
    safe_context = convert_numpy_types(context)

    logger.error(f"ERROR - {str(error)}")
    logger.error(f"  Error type: {type(error).__name__}")
    logger.error(f"  Timestamp: {timestamp}")

    for key, value in safe_context.items():
        logger.error(f"  {key}: {value}")


def create_summary_log(log_dir, pipeline_results):
    """
    Create a summary log file with overall pipeline results.

    Args:
        log_dir (str): Directory to store the summary log
        pipeline_results (dict): Results from the entire pipeline
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_file = Path(log_dir) / f"pipeline_summary_{timestamp}.log"

    # Ensure directory exists
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types
    safe_results = convert_numpy_types(pipeline_results)

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("SPOTIFY DATASETS INTEGRATION - PIPELINE SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for step_name, step_results in safe_results.items():
            f.write(f"{step_name.upper()}:\n")
            f.write("-" * 30 + "\n")

            if isinstance(step_results, dict):
                for key, value in step_results.items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  Result: {step_results}\n")

            f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("End of Summary\n")


# Test function for development
def test_logging():
    """Test logging functions with sample data"""
    import tempfile
    import os

    # Create temporary log directory
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, "test.log")

        # Test logger setup
        logger = setup_logger("test_logger", log_file)
        print("✓ Logger created")

        # Test count logging with numpy types
        test_counts = {
            'total_rows': np.int64(1000),  # Simulate numpy int64
            'valid_rows': np.int64(850),
            'invalid_rows': np.int64(150),
            'processed_rows': np.int64(800)
        }
        log_counts(logger, "Test Operation", test_counts)
        print("✓ Count logging tested")

        # Test schema collision logging
        collision_info = {
            'column': 'duration',
            'df1_name': 'Dataset1',
            'df2_name': 'Dataset2',
            'sample_df1_values': [180, 210, 195],
            'sample_df2_values': [185, 215, 200]
        }
        log_schema_collision(logger, collision_info)
        print("✓ Schema collision logging tested")

        # Test normalization results
        norm_results = {
            'total_ids': np.int64(100),  # Simulate numpy types
            'successful_normalizations': np.int64(95),
            'failed_normalizations': np.int64(5),
            'failed_examples': ['invalid_id_1', 'malformed_id_2']
        }
        log_normalization_results(logger, "Spotify ID Normalization", norm_results)
        print("✓ Normalization logging tested")

        print("✓ All logging tests completed successfully")


if __name__ == "__main__":
    test_logging()