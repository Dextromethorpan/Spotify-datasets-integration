#!/usr/bin/env python3
"""
Spotify ETL Runner Script
Provides a user-friendly interface to run the ETL pipeline with options.
"""

import argparse
import sys
import os
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import pandas
        import sqlite3
        print("✓ All required dependencies are available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False


def check_data_files():
    """Check if data files exist"""
    required_files = [
        'data/chordonomicon.csv',
        'data/kaggle_spotify.csv',
        'data/maharshipandya_spotify.csv'
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("⚠️  Missing data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nThe pipeline will use mock data for missing files.")
        print("To use your own data, place CSV files in the /data/ folder.")
        return False
    else:
        print("✓ All data files found")
        return True


def run_tests():
    """Run the test suite"""
    print("\n🧪 Running test suite...")
    try:
        # Import and run tests
        sys.path.append('tests')
        from test_spotify_etl import run_all_tests

        success = run_all_tests()
        if success:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed!")
            return False
    except Exception as e:
        print(f"✗ Error running tests: {e}")
        return False


def run_etl_pipeline():
    """Run the main ETL pipeline"""
    print("\n🎵 Starting Spotify ETL Pipeline...")
    try:
        from main_script import SpotifyETL

        etl = SpotifyETL()
        success = etl.run()

        if success:
            print("\n🎉 ETL Pipeline completed successfully!")
            print("\n📊 Results summary:")
            print("  • Check /results/ folder for SQLite databases")
            print("  • Check /logs/ folder for detailed processing logs")
            print("  • Check /samples/ folder for JSON sample data")
            return True
        else:
            print("\n❌ ETL Pipeline failed. Check logs for details.")
            return False

    except Exception as e:
        print(f"\n✗ Error running ETL pipeline: {e}")
        return False


def show_results_summary():
    """Show summary of generated results"""
    result_files = [
        'results/spotify_id_songs.sqlite',
        'results/spotify_id_songs_in_kaggle.sqlite',
        'results/spotify_id_songs_in_maharshipandya.sqlite',
        'results/spotify_id_songs_from_all_datasets.sqlite'
    ]

    print("\n📊 Generated databases:")
    for file_path in result_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            size_kb = size / 1024
            print(f"  ✓ {file_path} ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {file_path} (not found)")

    # Show sample files
    sample_files = list(Path('samples').glob('*.json')) if os.path.exists('samples') else []
    if sample_files:
        print(f"\n📋 Sample files generated: {len(sample_files)}")
        for sample_file in sample_files:
            print(f"  • {sample_file}")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Spotify Datasets Integration ETL Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_etl.py                    # Run full pipeline
  python run_etl.py --test-only        # Run tests only
  python run_etl.py --skip-tests       # Skip tests, run ETL directly
  python run_etl.py --check-only       # Check setup without running
        """
    )

    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Run tests only, skip ETL pipeline'
    )

    parser.add_argument(
        '--skip-tests',
        action='store_true',
        help='Skip tests, run ETL pipeline directly'
    )

    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Check dependencies and data files only'
    )

    args = parser.parse_args()

    print("🎵 Spotify Datasets Integration ETL")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check data files
    data_files_ok = check_data_files()

    # If only checking, exit here
    if args.check_only:
        if data_files_ok:
            print("\n✅ All checks passed! Ready to run ETL pipeline.")
            sys.exit(0)
        else:
            print("\n⚠️  Setup incomplete but can run with mock data.")
            sys.exit(0)

    # Run tests unless skipped
    if not args.skip_tests and not args.test_only:
        if not run_tests():
            print("\n⚠️  Tests failed but continuing with ETL...")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
    elif args.test_only:
        success = run_tests()
        sys.exit(0 if success else 1)

    # Run ETL pipeline
    if not args.test_only:
        success = run_etl_pipeline()

        if success:
            show_results_summary()

            print("\n🚀 Next steps:")
            print("  • Explore the SQLite databases in /results/")
            print("  • Review processing logs in /logs/")
            print("  • Check sample data in /samples/")
            print("  • Replace mock data with your real datasets in /data/")

            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()