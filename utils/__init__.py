# ========================================
# FILE 1: utils/__init__.py
# ========================================
"""
Spotify ETL Utilities Package

This package contains utility functions for the Spotify datasets integration project.

Modules:
- id_normalizer: Functions for normalizing Spotify IDs
- database_utils: Database operations and SQLite utilities
- logging_utils: Structured logging for ETL operations
- data_validator: Data validation and cleaning functions
"""

__version__ = "1.0.0"
__author__ = "Spotify ETL Project"

# Import key functions for easy access
from .id_normalizer import normalize_spotify_id, validate_spotify_id_format
from .database_utils import create_connection, create_indexes, batch_insert
from .logging_utils import setup_logger, log_counts, log_schema_collision
from .data_validator import validate_and_clean_data, is_valid_spotify_id

__all__ = [
    'normalize_spotify_id',
    'validate_spotify_id_format',
    'create_connection',
    'create_indexes',
    'batch_insert',
    'setup_logger',
    'log_counts',
    'log_schema_collision',
    'validate_and_clean_data',
    'is_valid_spotify_id'
]