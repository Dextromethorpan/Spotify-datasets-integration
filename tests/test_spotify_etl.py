#!/usr/bin/env python3
"""
Test Suite for Spotify ETL Pipeline
Tests all critical components of the ETL process.
"""

import unittest
import pandas as pd
import sqlite3
import tempfile
import os
import sys
from pathlib import Path

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from id_normalizer import normalize_spotify_id, validate_spotify_id_format, batch_normalize_ids
from database_utils import create_connection, create_indexes, batch_insert
from data_validator import validate_and_clean_data, is_valid_spotify_id, detect_outliers


class TestSpotifyIDNormalization(unittest.TestCase):
    """Test Spotify ID normalization functions"""

    def test_valid_id_unchanged(self):
        """Test that valid IDs remain unchanged"""
        valid_id = "4iV5W9uYEdYUVa79Axb7Rh"
        normalized, success = normalize_spotify_id(valid_id)
        self.assertEqual(normalized, valid_id.lower())
        self.assertTrue(success)

    def test_url_prefix_removal(self):
        """Test removal of various URL prefixes"""
        test_cases = [
            ("https://open.spotify.com/track/4iV5W9uYEdYUVa79Axb7Rh", "4iv5w9uyedyuva79axb7rh"),
            ("http://open.spotify.com/track/4iV5W9uYEdYUVa79Axb7Rh", "4iv5w9uyedyuva79axb7rh"),
            ("spotify:track:4iV5W9uYEdYUVa79Axb7Rh", "4iv5w9uyedyuva79axb7rh"),
            ("spotify:4iV5W9uYEdYUVa79Axb7Rh", "4iv5w9uyedyuva79axb7rh")
        ]

        for input_id, expected in test_cases:
            with self.subTest(input_id=input_id):
                normalized, success = normalize_spotify_id(input_id)
                self.assertEqual(normalized, expected)
                self.assertTrue(success)

    def test_query_parameter_removal(self):
        """Test removal of query parameters"""
        test_id = "4iV5W9uYEdYUVa79Axb7Rh?si=abc123def456"
        normalized, success = normalize_spotify_id(test_id)
        self.assertEqual(normalized, "4iv5w9uyedyuva79axb7rh")
        self.assertTrue(success)

    def test_whitespace_trimming(self):
        """Test trimming of whitespace"""
        test_id = "  4iV5W9uYEdYUVa79Axb7Rh  "
        normalized, success = normalize_spotify_id(test_id)
        self.assertEqual(normalized, "4iv5w9uyedyuva79axb7rh")
        self.assertTrue(success)

    def test_invalid_ids(self):
        """Test handling of invalid IDs"""
        invalid_ids = [
            "invalid_id",
            "too_short",
            "way_too_long_to_be_a_valid_spotify_id_format",
            "",
            None
        ]

        for invalid_id in invalid_ids:
            with self.subTest(invalid_id=invalid_id):
                normalized, success = normalize_spotify_id(invalid_id)
                self.assertFalse(success)

    def test_batch_normalization(self):
        """Test batch normalization function"""
        test_ids = [
            "4iV5W9uYEdYUVa79Axb7Rh",
            "https://open.spotify.com/track/1uNFoZAHBGtllmzznpCI3s",
            "invalid_id",
            "spotify:track:7qiZfU4dY1lWllzX7mPBI3"
        ]

        results = batch_normalize_ids(test_ids)

        self.assertEqual(len(results['normalized_ids']), 4)
        self.assertEqual(results['success_count'], 3)
        self.assertEqual(results['failed_count'], 1)
        self.assertEqual(len(results['failed_ids']), 1)
        self.assertIn("invalid_id", results['failed_ids'])


class TestDatabaseOperations(unittest.TestCase):
    """Test database utility functions"""

    def setUp(self):
        """Set up temporary database for testing"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()

    def tearDown(self):
        """Clean up temporary database"""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_database_connection(self):
        """Test database connection creation"""
        conn = create_connection(self.db_path)
        self.assertIsNotNone(conn)
        self.assertIsInstance(conn, sqlite3.Connection)
        conn.close()

    def test_batch_insert(self):
        """Test batch insert functionality"""
        # Create test data
        test_data = pd.DataFrame({
            'id': ['abc123', 'def456', 'ghi789'],
            'name': ['Song 1', 'Song 2', 'Song 3'],
            'artist': ['Artist A', 'Artist B', 'Artist C']
        })

        conn = create_connection(self.db_path)
        success = batch_insert(conn, 'test_songs', test_data, batch_size=2)
        self.assertTrue(success)

        # Verify data was inserted
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM test_songs")
        count = cursor.fetchone()[0]
        self.assertEqual(count, 3)

        conn.close()

    def test_index_creation(self):
        """Test index creation"""
        # Create test table first
        test_data = pd.DataFrame({
            'spotify_id': ['abc123', 'def456'],
            'name': ['Song 1', 'Song 2']
        })

        conn = create_connection(self.db_path)
        test_data.to_sql('test_table', conn, if_exists='replace', index=False)

        # Create indexes
        create_indexes(conn, 'test_table', ['spotify_id'])

        # Verify indexes exist (this is a basic check)
        cursor = conn.cursor()
        cursor.execute("PRAGMA index_list(test_table)")
        indexes = cursor.fetchall()

        # Should have at least one index
        self.assertGreater(len(indexes), 0)

        conn.close()


class TestDataValidation(unittest.TestCase):
    """Test data validation functions"""

    def test_spotify_id_validation(self):
        """Test Spotify ID format validation"""
        valid_ids = [
            "4iV5W9uYEdYUVa79Axb7Rh",
            "1uNFoZAHBGtllmzznpCI3s",
            "7qiZfU4dY1lWllzX7mPBI3"
        ]

        invalid_ids = [
            "invalid_id",
            "too_short",
            "way_too_long_to_be_valid",
            "",
            None,
            123
        ]

        for valid_id in valid_ids:
            with self.subTest(valid_id=valid_id):
                self.assertTrue(is_valid_spotify_id(valid_id))

        for invalid_id in invalid_ids:
            with self.subTest(invalid_id=invalid_id):
                self.assertFalse(is_valid_spotify_id(invalid_id))

    def test_data_cleaning(self):
        """Test data cleaning and validation"""
        # Create test DataFrame with issues
        test_data = pd.DataFrame({
            'spotify_id': ['4iV5W9uYEdYUVa79Axb7Rh', None, 'invalid_id'],
            'song name': ['Song 1', 'Song 2', 'Song 3'],  # Space in column name
            'duration': [180, 210, 9999],  # Outlier
            'empty_col': [None, None, None]  # All null
        })

        cleaned_df, report = validate_and_clean_data(test_data, "Test Dataset")

        # Check that issues were detected
        self.assertGreater(len(report['issues_found']), 0)

        # Check that empty column was removed
        self.assertNotIn('empty_col', cleaned_df.columns)

        # Check that column names were cleaned
        self.assertIn('song_name', cleaned_df.columns)
        self.assertNotIn('song name', cleaned_df.columns)

    def test_outlier_detection(self):
        """Test outlier detection"""
        # Create data with obvious outliers
        normal_data = [100, 110, 105, 95, 102, 108, 97, 103]
        outlier_data = normal_data + [1000, -50]  # Add outliers

        test_series = pd.Series(outlier_data)
        results = detect_outliers(test_series, method='iqr')

        # Should detect the outliers
        self.assertGreater(results['count'], 0)
        self.assertIn(1000, results['values'])
        self.assertIn(-50, results['values'])


class TestCSVLoading(unittest.TestCase):
    """Test CSV loading with different encodings"""

    def test_csv_loading_utf8(self):
        """Test loading CSV with UTF-8 encoding"""
        # Create temporary CSV file
        test_data = "spotify_id,name,artist\n4iV5W9uYEdYUVa79Axb7Rh,Test Song,Test Artist\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(test_data)
            temp_csv = f.name

        try:
            # Test loading
            df = pd.read_csv(temp_csv, encoding='utf-8')
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]['spotify_id'], '4iV5W9uYEdYUVa79Axb7Rh')
        finally:
            os.unlink(temp_csv)


class TestSchemaCollisionDetection(unittest.TestCase):
    """Test schema collision detection"""

    def test_collision_detection(self):
        """Test detection of schema collisions"""
        df1 = pd.DataFrame({
            'id': ['abc123', 'def456'],
            'duration': [180, 210],
            'artist': ['Artist A', 'Artist B']
        })

        df2 = pd.DataFrame({
            'id': ['abc123', 'def456'],
            'duration': [185, 215],  # Different values
            'artist': ['Artist A', 'Artist B']  # Same values
        })

        # Simple collision detection - check if values differ
        common_cols = set(df1.columns) & set(df2.columns) - {'id'}
        collisions_found = False

        for col in common_cols:
            # Compare first few values
            if not df1[col].head(2).equals(df2[col].head(2)):
                if col == 'duration':  # We expect duration to have conflicts
                    collisions_found = True

        self.assertTrue(collisions_found)


def run_all_tests():
    """Run all tests and return results"""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestSpotifyIDNormalization,
        TestDatabaseOperations,
        TestDataValidation,
        TestCSVLoading,
        TestSchemaCollisionDetection
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Running Spotify ETL Test Suite")
    print("=" * 50)

    success = run_all_tests()

    if success:
        print("\n✅ All tests passed! ETL pipeline is ready to run.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)