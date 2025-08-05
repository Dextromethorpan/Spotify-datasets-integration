#!/usr/bin/env python3
"""
Spotify Datasets Integration Project - Main ETL Script
Processes three Spotify datasets and creates unified SQLite databases.
Updated with robust column detection for real datasets.
"""

import os
import sys
import pandas as pd
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path

# Add utils to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from id_normalizer import normalize_spotify_id
from database_utils import create_connection, create_indexes, batch_insert
from logging_utils import setup_logger, log_counts, log_schema_collision
from data_validator import validate_and_clean_data


class SpotifyETL:
    def __init__(self):
        self.setup_directories()
        self.loggers = self.setup_loggers()
        self.column_mappings = self.setup_column_mappings()

    def setup_directories(self):
        """Create all required directories"""
        dirs = [
            'data', 'results', 'logs', 'samples', 'tests', 'utils',
            'logs/null_missing_data', 'logs/schema_collisions', 'logs/join_inconsistencies'
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def setup_loggers(self):
        """Setup all loggers for different categories"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        loggers = {
            'null_data': setup_logger('null_data', f'logs/null_missing_data/processing_{timestamp}.log'),
            'schema': setup_logger('schema', f'logs/schema_collisions/conflicts_{timestamp}.log'),
            'joins': setup_logger('joins', f'logs/join_inconsistencies/joins_{timestamp}.log')
        }
        return loggers

    def setup_column_mappings(self):
        """Setup column mappings for different datasets"""
        return {
            'chordonomicon': {
                'spotify_id_patterns': ['spotify_song_id', 'spotify_id', 'track_id', 'id', 'spotify_track_id', 'uri'],
                'song_name_patterns': ['title', 'name', 'song_name', 'track_name', 'song', 'track'],
                'artist_patterns': ['artist', 'artist_name', 'artists', 'performer']
            },
            'kaggle': {
                'spotify_id_patterns': ['track_id', 'spotify_id', 'id', 'spotify_track_id', 'uri'],
                'song_name_patterns': ['track_name', 'name', 'title', 'song_name', 'song', 'track'],
                'artist_patterns': ['artists', 'artist_name', 'artist', 'performer'],  # Kaggle uses 'artists'
                'key_patterns': ['key', 'musical_key', 'music_key', 'tonic']
            },
            'maharshipandya': {
                'spotify_id_patterns': ['track_id', 'spotify_id', 'id', 'spotify_track_id', 'uri'],
                'song_name_patterns': ['track_name', 'name', 'title', 'song_name', 'song', 'track'],
                'artist_patterns': ['artists', 'artist_name', 'artist', 'performer'],  # Maharshipandya uses 'artists'
                'key_patterns': ['key', 'musical_key', 'music_key', 'tonic']
            }
        }

    def find_column(self, df, patterns, column_type="column"):
        """Find a column based on pattern matching"""
        for pattern in patterns:
            if pattern in df.columns:
                print(f"  ✅ Found {column_type}: '{pattern}'")
                return pattern
        return None

    def load_dataset(self, filepath, dataset_name):
        """Load dataset with error handling"""
        try:
            print(f"Loading {dataset_name} from {filepath}...")

            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
                    print(f"✓ Loaded {len(df)} rows from {dataset_name}")
                    self.loggers['joins'].info(f"Successfully loaded {dataset_name}: {len(df)} rows")
                    return df
                except UnicodeDecodeError:
                    continue

            raise Exception(f"Could not decode {filepath} with any encoding")

        except Exception as e:
            print(f"✗ Error loading {dataset_name}: {e}")
            self.loggers['joins'].error(f"Failed to load {dataset_name}: {e}")
            return None

    def step1_extract_valid_spotify_ids(self):
        """Step 1: Extract songs with valid spotify_id from Chordonomicon"""
        print("\n=== STEP 1: Extracting valid Spotify IDs ===")

        # Load Chordonomicon dataset
        chord_df = self.load_dataset('data/chordonomicon.csv', 'Chordonomicon')
        if chord_df is None:
            return None

        # Find the Spotify ID column
        print("🔍 Detecting column structure...")
        mappings = self.column_mappings['chordonomicon']
        spotify_id_column = self.find_column(chord_df, mappings['spotify_id_patterns'], 'Spotify ID column')

        if spotify_id_column is None:
            print("❌ No Spotify ID column found. Available columns:")
            print(f"   {list(chord_df.columns)}")
            self.loggers['joins'].error("No Spotify ID column found in Chordonomicon")
            return None

        # Show sample values to verify
        sample_values = chord_df[spotify_id_column].dropna().head(3).tolist()
        print(f"📋 Sample Spotify IDs: {sample_values}")

        # Count null spotify_ids before filtering
        null_count = chord_df[spotify_id_column].isnull().sum()
        total_count = len(chord_df)

        print(f"📊 Data overview: {total_count:,} total rows, {null_count:,} null IDs")

        # Filter for non-null spotify_ids
        valid_df = chord_df[chord_df[spotify_id_column].notna()].copy()
        valid_count = len(valid_df)

        print(f"✅ Found {valid_count:,} songs with valid Spotify IDs")

        # Normalize spotify_ids with progress tracking
        print("🔧 Normalizing Spotify IDs...")
        normalized_results = []

        total_ids = len(valid_df)
        for idx, spotify_id in enumerate(valid_df[spotify_id_column]):
            normalized_id, success = normalize_spotify_id(spotify_id)
            valid_df.iloc[idx, valid_df.columns.get_loc(spotify_id_column)] = normalized_id
            normalized_results.append(success)

            # Progress update for large datasets
            if (idx + 1) % 50000 == 0 or (idx + 1) == total_ids:
                progress = (idx + 1) / total_ids * 100
                print(f"  Progress: {idx + 1:,}/{total_ids:,} IDs processed ({progress:.1f}%)")

        normalized_success = sum(normalized_results)
        normalized_failed = len(normalized_results) - normalized_success

        print(f"✅ Normalization complete: {normalized_success:,} successful, {normalized_failed:,} failed")

        # Rename column to standardize for later steps
        if spotify_id_column != 'spotify_id':
            valid_df = valid_df.rename(columns={spotify_id_column: 'spotify_id'})
            print(f"🔄 Renamed '{spotify_id_column}' to 'spotify_id' for consistency")

        # Log results
        log_counts(self.loggers['null_data'], 'Chordonomicon spotify_id extraction', {
            'total_rows': total_count,
            'null_spotify_ids': null_count,
            'valid_spotify_ids': valid_count,
            'normalized_success': normalized_success,
            'normalized_failed': normalized_failed
        })

        # Save to SQLite
        print("💾 Saving to database...")
        db_path = 'results/spotify_id_songs.sqlite'
        conn = create_connection(db_path)
        if conn:
            valid_df.to_sql('songs', conn, if_exists='replace', index=False)
            create_indexes(conn, 'songs', ['spotify_id'])
            conn.close()
            print(f"✅ Saved {valid_count:,} valid songs to {db_path}")

        return valid_df

    def detect_schema_collisions(self, df1, df2, df1_name, df2_name):
        """Detect and resolve schema collisions between datasets"""
        collisions = []
        common_cols = set(df1.columns) & set(df2.columns)

        for col in common_cols:
            if col in ['spotify_id', 'track_id']:  # Skip ID columns
                continue

            # Sample some rows to check for conflicts
            sample_size = min(100, len(df1), len(df2))
            df1_sample = df1[col].dropna().head(sample_size)
            df2_sample = df2[col].dropna().head(sample_size)

            # Check if values differ
            if len(df1_sample) > 0 and len(df2_sample) > 0:
                # Simple conflict detection - check if first few values differ
                conflicts_found = False
                for i in range(min(5, len(df1_sample), len(df2_sample))):
                    if str(df1_sample.iloc[i]) != str(df2_sample.iloc[i]):
                        conflicts_found = True
                        break

                if conflicts_found:
                    collision_info = {
                        'column': col,
                        'df1_name': df1_name,
                        'df2_name': df2_name,
                        'sample_df1_values': df1_sample.head(3).tolist(),
                        'sample_df2_values': df2_sample.head(3).tolist()
                    }
                    collisions.append(collision_info)
                    log_schema_collision(self.loggers['schema'], collision_info)

        return collisions

    def resolve_schema_collisions(self, df1, df2, df1_name, df2_name, join_on_left, join_on_right):
        """Resolve schema collisions by renaming conflicting columns"""
        # Merge datasets
        print(f"🔗 Joining {df1_name} with {df2_name}...")
        merged = pd.merge(df1, df2, left_on=join_on_left, right_on=join_on_right, how='inner',
                          suffixes=(f'_{df1_name}', f'_{df2_name}'))

        # Detect collisions in merged data
        collisions = self.detect_schema_collisions(df1, df2, df1_name, df2_name)

        print(f"✓ Found {len(collisions)} schema collisions")
        if collisions:
            collision_cols = [c['column'] for c in collisions]
            print(f"  Conflicting columns: {collision_cols}")

        return merged

    def step2_match_with_kaggle(self, valid_songs_df):
        """Step 2: Match with Kaggle dataset"""
        print("\n=== STEP 2: Matching with Kaggle dataset ===")

        # Load Kaggle dataset
        kaggle_df = self.load_dataset('data/kaggle_spotify.csv', 'Kaggle Spotify')
        if kaggle_df is None:
            return None

        # Find columns in Kaggle dataset
        print("🔍 Detecting Kaggle dataset structure...")
        mappings = self.column_mappings['kaggle']

        track_id_column = self.find_column(kaggle_df, mappings['spotify_id_patterns'], 'track ID column')
        key_column = self.find_column(kaggle_df, mappings['key_patterns'], 'key column')

        if track_id_column is None:
            print("❌ No track ID column found in Kaggle dataset")
            self.loggers['joins'].error("No track ID column found in Kaggle dataset")
            return None

        # Normalize track_ids in Kaggle dataset
        print("🔧 Normalizing Kaggle track IDs...")
        kaggle_df[track_id_column] = kaggle_df[track_id_column].apply(
            lambda x: normalize_spotify_id(x)[0] if pd.notna(x) else x
        )

        # Filter for non-null keys if key column exists
        kaggle_filtered = kaggle_df.copy()
        kaggle_null_keys = 0

        if key_column:
            kaggle_filtered = kaggle_df[kaggle_df[key_column].notna()].copy()
            kaggle_null_keys = len(kaggle_df) - len(kaggle_filtered)
            print(f"📊 Filtered out {kaggle_null_keys:,} rows with null keys")
        else:
            print("⚠️  No key column found - proceeding without key filtering")

        # Standardize column name for joining
        if track_id_column != 'track_id':
            kaggle_filtered = kaggle_filtered.rename(columns={track_id_column: 'track_id'})

        # Perform merge with collision resolution
        matched_df = self.resolve_schema_collisions(
            valid_songs_df, kaggle_filtered,
            'Chordonomicon', 'Kaggle',
            'spotify_id', 'track_id'
        )

        # Log matching results
        log_counts(self.loggers['joins'], 'Kaggle dataset matching', {
            'chordonomicon_songs': len(valid_songs_df),
            'kaggle_total_rows': len(kaggle_df),
            'kaggle_null_keys': kaggle_null_keys,
            'kaggle_valid_rows': len(kaggle_filtered),
            'successful_matches': len(matched_df),
            'failed_matches': len(valid_songs_df) - len(matched_df)
        })

        # Save to SQLite
        db_path = 'results/spotify_id_songs_in_kaggle.sqlite'
        conn = create_connection(db_path)
        if conn:
            matched_df.to_sql('songs', conn, if_exists='replace', index=False)
            create_indexes(conn, 'songs', ['spotify_id', 'track_id'])
            conn.close()
            print(f"✅ Saved {len(matched_df):,} matched songs to {db_path}")

        return matched_df

    def step3_match_with_maharshipandya(self, valid_songs_df):
        """Step 3: Match with Maharshipandya dataset"""
        print("\n=== STEP 3: Matching with Maharshipandya dataset ===")

        # Load Maharshipandya dataset
        maharshi_df = self.load_dataset('data/maharshipandya_spotify.csv', 'Maharshipandya Spotify')
        if maharshi_df is None:
            return None

        # Find columns in Maharshipandya dataset
        print("🔍 Detecting Maharshipandya dataset structure...")
        mappings = self.column_mappings['maharshipandya']

        track_id_column = self.find_column(maharshi_df, mappings['spotify_id_patterns'], 'track ID column')
        key_column = self.find_column(maharshi_df, mappings['key_patterns'], 'key column')

        if track_id_column is None:
            print("❌ No track ID column found in Maharshipandya dataset")
            self.loggers['joins'].error("No track ID column found in Maharshipandya dataset")
            return None

        # Normalize track_ids
        print("🔧 Normalizing Maharshipandya track IDs...")
        maharshi_df[track_id_column] = maharshi_df[track_id_column].apply(
            lambda x: normalize_spotify_id(x)[0] if pd.notna(x) else x
        )

        # Filter for non-null keys if key column exists
        maharshi_filtered = maharshi_df.copy()
        maharshi_null_keys = 0

        if key_column:
            maharshi_filtered = maharshi_df[maharshi_df[key_column].notna()].copy()
            maharshi_null_keys = len(maharshi_df) - len(maharshi_filtered)
            print(f"📊 Filtered out {maharshi_null_keys:,} rows with null keys")
        else:
            print("⚠️  No key column found - proceeding without key filtering")

        # Standardize column name for joining
        if track_id_column != 'track_id':
            maharshi_filtered = maharshi_filtered.rename(columns={track_id_column: 'track_id'})

        # Perform merge with collision resolution
        matched_df = self.resolve_schema_collisions(
            valid_songs_df, maharshi_filtered,
            'Chordonomicon', 'Maharshipandya',
            'spotify_id', 'track_id'
        )

        # Log matching results
        log_counts(self.loggers['joins'], 'Maharshipandya dataset matching', {
            'chordonomicon_songs': len(valid_songs_df),
            'maharshipandya_total_rows': len(maharshi_df),
            'maharshipandya_null_keys': maharshi_null_keys,
            'maharshipandya_valid_rows': len(maharshi_filtered),
            'successful_matches': len(matched_df),
            'failed_matches': len(valid_songs_df) - len(matched_df)
        })

        # Save to SQLite
        db_path = 'results/spotify_id_songs_in_maharshipandya.sqlite'
        conn = create_connection(db_path)
        if conn:
            matched_df.to_sql('songs', conn, if_exists='replace', index=False)
            create_indexes(conn, 'songs', ['spotify_id', 'track_id'])
            conn.close()
            print(f"✅ Saved {len(matched_df):,} matched songs to {db_path}")

        return matched_df

    def step4_combine_datasets(self, kaggle_matched_df, maharshi_matched_df):
        """Step 4: Combine both matched datasets"""
        print("\n=== STEP 4: Combining all datasets ===")

        if kaggle_matched_df is None or maharshi_matched_df is None:
            print("⚠️  One or more datasets failed to load - creating partial combination")

            # Handle partial combinations
            if kaggle_matched_df is not None and maharshi_matched_df is None:
                combined_df = kaggle_matched_df.copy()
                print("✓ Using only Kaggle matched data")
            elif maharshi_matched_df is not None and kaggle_matched_df is None:
                combined_df = maharshi_matched_df.copy()
                print("✓ Using only Maharshipandya matched data")
            else:
                print("❌ No datasets available for combination")
                return None
        else:
            # Combine datasets and remove duplicates by spotify_id
            combined_df = pd.concat([kaggle_matched_df, maharshi_matched_df], ignore_index=True)

        # Remove duplicates by spotify_id, keeping first occurrence
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['spotify_id'], keep='first')
        after_dedup = len(combined_df)
        duplicates_removed = before_dedup - after_dedup

        print(f"🔄 Removed {duplicates_removed:,} duplicates")

        # Log combination results
        log_counts(self.loggers['joins'], 'Dataset combination', {
            'kaggle_matched_songs': len(kaggle_matched_df) if kaggle_matched_df is not None else 0,
            'maharshipandya_matched_songs': len(maharshi_matched_df) if maharshi_matched_df is not None else 0,
            'total_before_dedup': before_dedup,
            'duplicates_removed': duplicates_removed,
            'final_unique_songs': after_dedup
        })

        # Save to SQLite
        db_path = 'results/spotify_id_songs_from_all_datasets.sqlite'
        conn = create_connection(db_path)
        if conn:
            combined_df.to_sql('songs', conn, if_exists='replace', index=False)
            create_indexes(conn, 'songs', ['spotify_id'])
            conn.close()
            print(f"✅ Saved {after_dedup:,} unique songs to {db_path}")

        return combined_df

    def generate_samples(self):
        """Generate JSON samples from all databases"""
        print("\n=== Generating sample data ===")

        db_files = [
            'results/spotify_id_songs.sqlite',
            'results/spotify_id_songs_in_kaggle.sqlite',
            'results/spotify_id_songs_in_maharshipandya.sqlite',
            'results/spotify_id_songs_from_all_datasets.sqlite'
        ]

        for db_file in db_files:
            if os.path.exists(db_file):
                try:
                    conn = sqlite3.connect(db_file)
                    df = pd.read_sql_query("SELECT * FROM songs LIMIT 5", conn)
                    conn.close()

                    # Convert to JSON
                    sample_data = df.to_dict('records')

                    # Save sample
                    db_name = os.path.basename(db_file).replace('.sqlite', '')
                    sample_file = f'samples/{db_name}_sample.json'

                    with open(sample_file, 'w', encoding='utf-8') as f:
                        json.dump(sample_data, f, indent=2, default=str)

                    print(f"✓ Generated sample: {sample_file}")

                except Exception as e:
                    print(f"✗ Error generating sample for {db_file}: {e}")

    def run(self):
        """Run the complete ETL pipeline"""
        print("🎵 Starting Spotify Datasets Integration ETL Pipeline 🎵")
        print("=" * 60)

        try:
            # Step 1: Extract valid Spotify IDs
            valid_songs = self.step1_extract_valid_spotify_ids()
            if valid_songs is None:
                print("✗ Step 1 failed - cannot continue")
                return False

            # Step 2: Match with Kaggle
            kaggle_matched = self.step2_match_with_kaggle(valid_songs)

            # Step 3: Match with Maharshipandya
            maharshi_matched = self.step3_match_with_maharshipandya(valid_songs)

            # Step 4: Combine datasets
            combined = self.step4_combine_datasets(kaggle_matched, maharshi_matched)

            # Generate samples
            self.generate_samples()

            print("\n" + "=" * 60)
            print("🎉 ETL Pipeline completed successfully!")
            print("Check the /results folder for SQLite databases")
            print("Check the /logs folder for detailed processing logs")
            print("Check the /samples folder for JSON sample data")

            return True

        except Exception as e:
            print(f"\n✗ ETL Pipeline failed: {e}")
            self.loggers['joins'].error(f"ETL Pipeline failed: {e}")
            return False


def main():
    """Main execution function"""
    etl = SpotifyETL()
    success = etl.run()

    if success:
        print("\n✅ All done! Your integrated Spotify datasets are ready.")
    else:
        print("\n❌ ETL process encountered errors. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()