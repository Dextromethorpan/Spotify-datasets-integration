# ========================================
# FILE 3: utils/database_utils.py
# ========================================
"""
Database Utilities for SQLite Operations
Handles connections, indexing, and batch operations.
"""

import sqlite3
import pandas as pd
import logging
from pathlib import Path


def create_connection(db_path):
    """
    Create a database connection to SQLite database.

    Args:
        db_path (str): Path to SQLite database file

    Returns:
        sqlite3.Connection or None: Database connection object
    """
    try:
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
        conn.execute("PRAGMA journal_mode = WAL")  # Enable WAL mode for better performance
        return conn
    except sqlite3.Error as e:
        logging.error(f"Error creating database connection to {db_path}: {e}")
        return None


def create_indexes(conn, table_name, columns):
    """
    Create indexes on specified columns for better query performance.

    Args:
        conn (sqlite3.Connection): Database connection
        table_name (str): Name of table to index
        columns (list): List of column names to create indexes on
    """
    try:
        cursor = conn.cursor()

        for column in columns:
            index_name = f"idx_{table_name}_{column}"
            sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({column})"
            cursor.execute(sql)
            logging.info(f"Created index {index_name} on {table_name}.{column}")

        conn.commit()

    except sqlite3.Error as e:
        logging.error(f"Error creating indexes: {e}")
        conn.rollback()


def batch_insert(conn, table_name, dataframe, batch_size=1000):
    """
    Insert DataFrame data in batches for better performance.

    Args:
        conn (sqlite3.Connection): Database connection
        table_name (str): Target table name
        dataframe (pd.DataFrame): Data to insert
        batch_size (int): Number of rows per batch

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Insert in batches
        total_rows = len(dataframe)
        batches_processed = 0

        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = dataframe.iloc[start_idx:end_idx]

            # Use to_sql with append mode for batches after first
            if_exists = 'replace' if start_idx == 0 else 'append'

            batch_df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            batches_processed += 1

            if batches_processed % 10 == 0:  # Log progress every 10 batches
                logging.info(f"Processed {batches_processed * batch_size} / {total_rows} rows")

        conn.commit()
        logging.info(f"Successfully inserted {total_rows} rows into {table_name}")
        return True

    except Exception as e:
        logging.error(f"Error in batch insert: {e}")
        conn.rollback()
        return False


def get_table_info(conn, table_name):
    """
    Get information about a table structure.

    Args:
        conn (sqlite3.Connection): Database connection
        table_name (str): Name of table to examine

    Returns:
        list: Table column information
    """
    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        return cursor.fetchall()
    except sqlite3.Error as e:
        logging.error(f"Error getting table info for {table_name}: {e}")
        return []


def get_table_stats(conn, table_name):
    """
    Get basic statistics about a table.

    Args:
        conn (sqlite3.Connection): Database connection
        table_name (str): Name of table to examine

    Returns:
        dict: Table statistics
    """
    try:
        cursor = conn.cursor()

        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]

        # Get column count
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        column_count = len(columns)

        # Get table size (approximation)
        cursor.execute(f"SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        db_size = cursor.fetchone()[0]

        return {
            'row_count': row_count,
            'column_count': column_count,
            'columns': [col[1] for col in columns],  # Column names
            'estimated_size_bytes': db_size
        }

    except sqlite3.Error as e:
        logging.error(f"Error getting stats for {table_name}: {e}")
        return {}


def execute_query(conn, query, params=None):
    """
    Execute a SQL query safely with optional parameters.

    Args:
        conn (sqlite3.Connection): Database connection
        query (str): SQL query to execute
        params (tuple, optional): Query parameters

    Returns:
        list: Query results or empty list if error
    """
    try:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        return cursor.fetchall()

    except sqlite3.Error as e:
        logging.error(f"Error executing query: {e}")
        return []


def optimize_database(conn):
    """
    Optimize database performance by running maintenance commands.

    Args:
        conn (sqlite3.Connection): Database connection
    """
    try:
        cursor = conn.cursor()

        # Analyze database for query optimizer
        cursor.execute("ANALYZE")

        # Vacuum to reclaim space and defragment
        cursor.execute("VACUUM")

        logging.info("Database optimization completed")

    except sqlite3.Error as e:
        logging.error(f"Error optimizing database: {e}")


def backup_database(source_path, backup_path):
    """
    Create a backup of the database.

    Args:
        source_path (str): Path to source database
        backup_path (str): Path for backup database

    Returns:
        bool: True if backup successful, False otherwise
    """
    try:
        # Ensure backup directory exists
        Path(backup_path).parent.mkdir(parents=True, exist_ok=True)

        # Connect to source database
        source_conn = sqlite3.connect(source_path)

        # Create backup connection
        backup_conn = sqlite3.connect(backup_path)

        # Perform backup
        source_conn.backup(backup_conn)

        # Close connections
        backup_conn.close()
        source_conn.close()

        logging.info(f"Database backed up from {source_path} to {backup_path}")
        return True

    except Exception as e:
        logging.error(f"Error backing up database: {e}")
        return False


def create_table_from_dataframe(conn, table_name, dataframe, primary_key=None):
    """
    Create a table based on DataFrame structure with optional primary key.

    Args:
        conn (sqlite3.Connection): Database connection
        table_name (str): Name of table to create
        dataframe (pd.DataFrame): DataFrame to base table structure on
        primary_key (str, optional): Column name to use as primary key

    Returns:
        bool: True if table created successfully, False otherwise
    """
    try:
        cursor = conn.cursor()

        # Map pandas dtypes to SQLite types
        dtype_mapping = {
            'object': 'TEXT',
            'int64': 'INTEGER',
            'float64': 'REAL',
            'bool': 'INTEGER',
            'datetime64[ns]': 'TIMESTAMP'
        }

        # Build column definitions
        columns = []
        for col_name, dtype in dataframe.dtypes.items():
            sql_type = dtype_mapping.get(str(dtype), 'TEXT')

            if col_name == primary_key:
                columns.append(f"{col_name} {sql_type} PRIMARY KEY")
            else:
                columns.append(f"{col_name} {sql_type}")

        # Create table SQL
        create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
        cursor.execute(create_sql)
        conn.commit()

        logging.info(f"Created table {table_name} with {len(columns)} columns")
        return True

    except Exception as e:
        logging.error(f"Error creating table {table_name}: {e}")
        return False


# Test function for development
def test_database_operations():
    """Test database operations with sample data"""
    import tempfile
    import os

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as tmp:
        db_path = tmp.name

    try:
        # Test connection
        conn = create_connection(db_path)
        if not conn:
            print("✗ Failed to create connection")
            return
        print("✓ Database connection created")

        # Test with sample data
        sample_data = pd.DataFrame({
            'id': ['abc123', 'def456', 'ghi789'],
            'name': ['Song 1', 'Song 2', 'Song 3'],
            'artist': ['Artist A', 'Artist B', 'Artist C'],
            'duration': [180, 210, 195]
        })

        # Test table creation
        if create_table_from_dataframe(conn, 'test_songs', sample_data, 'id'):
            print("✓ Table created successfully")

        # Test batch insert
        if batch_insert(conn, 'test_songs', sample_data):
            print("✓ Batch insert successful")

        # Test index creation
        create_indexes(conn, 'test_songs', ['id', 'artist'])
        print("✓ Indexes created")

        # Test table stats
        stats = get_table_stats(conn, 'test_songs')
        print(f"✓ Table stats: {stats['row_count']} rows, {stats['column_count']} columns")

        conn.close()
        print("✓ All database operations completed successfully")

    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == "__main__":
    test_database_operations()