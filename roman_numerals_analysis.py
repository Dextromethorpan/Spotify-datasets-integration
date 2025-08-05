#!/usr/bin/env python3
"""
Spotify Roman Numeral Analysis Pipeline
Standalone script that creates roman numeral chord analysis from the integrated database.
Creates new database, samples, and logs.
"""

import os
import sys
import pandas as pd
import sqlite3
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add utils to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from database_utils import create_connection, create_indexes
from logging_utils import setup_logger, log_counts

class RomanNumeralAnalysisPipeline:
    def __init__(self):
        self.setup_directories()
        self.loggers = self.setup_loggers()
        self.setup_chord_converter()

    def setup_directories(self):
        dirs = [
            'results', 'samples', 'logs',
            'logs/roman_numeral_analysis'
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def setup_loggers(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        loggers = {
            'conversion': setup_logger('chord_conversion', f'logs/roman_numeral_analysis/chord_conversion_{timestamp}.log'),
            'harmonic': setup_logger('harmonic_analysis', f'logs/roman_numeral_analysis/harmonic_analysis_{timestamp}.log'),
            'summary': setup_logger('roman_summary', f'logs/roman_numeral_analysis/roman_processing_summary_{timestamp}.log')
        }
        return loggers

    def setup_chord_converter(self):
        self.pitch_class_to_note = {
            0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
            6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
        }
        self.major_scale_romans = {
            0: 'I', 1: 'bII', 2: 'ii', 3: 'bIII', 4: 'iii', 5: 'IV',
            6: 'bV', 7: 'V', 8: 'bVI', 9: 'vi', 10: 'bVII', 11: 'vii°'
        }
        self.minor_scale_romans = {
            0: 'i', 1: 'bII', 2: 'ii°', 3: 'III', 4: 'iv', 5: 'V',
            6: 'bVI', 7: 'VII', 8: 'viii', 9: 'VI', 10: 'VII', 11: 'vii°'
        }

    def load_source_database(self):
        source_db = 'results/spotify_id_songs_from_all_datasets.sqlite'
        if not os.path.exists(source_db):
            print(f"❌ Source database not found: {source_db}")
            return None
        try:
            print(f"📊 Loading source database: {source_db}")
            conn = sqlite3.connect(source_db)
            query = "SELECT * FROM songs WHERE chords IS NOT NULL AND key IS NOT NULL"
            df = pd.read_sql_query(query, conn)
            conn.close()
            print(f"✅ Loaded {len(df):,} songs with chord and key data")
            self.loggers['conversion'].info(f"Loaded {len(df)} songs for roman numeral analysis")
            return df
        except Exception as e:
            print(f"❌ Error loading source database: {e}")
            self.loggers['conversion'].error(f"Failed to load source database: {e}")
            return None

    def parse_chord_progression_with_structure(self, progression_str: str) -> List[str]:
        if not progression_str or not isinstance(progression_str, str):
            return []
        parts = re.split(r'(<[^>]+>)', progression_str.strip())
        result = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.startswith('<') and part.endswith('>'):
                result.append(part)
            else:
                chords = re.split(r'[\s\-\|,]+', part)
                result.extend([ch.strip() for ch in chords if ch.strip()])
        return result

    def parse_chord_symbol(self, chord_str: str) -> Tuple[Optional[str], Optional[str]]:
        if not chord_str or not isinstance(chord_str, str):
            return None, None
        match = re.match(r'^([A-G][#b]?)(.*)$', chord_str.strip())
        return (match.group(1), match.group(2).strip()) if match else (None, None)

    def note_to_pitch_class(self, note: str) -> Optional[int]:
        for pc, name in self.pitch_class_to_note.items():
            if note == name:
                return pc
        return None

    def get_scale_degree(self, chord_root_pc: int, key_pc: int) -> int:
        return (chord_root_pc - key_pc) % 12

    def chord_quality_to_roman_modifier(self, quality: str) -> str:
        if not quality:
            return ""
        quality = quality.lower()
        if 'maj7' in quality:
            return 'M7'
        if '7' in quality:
            return '7'
        if 'dim' in quality or '°' in quality:
            return '°'
        if 'aug' in quality or '+' in quality:
            return '+'
        if 'sus4' in quality:
            return 'sus4'
        if 'sus2' in quality:
            return 'sus2'
        if '6' in quality:
            return '6'
        if '9' in quality:
            return '9'
        return ""

    def convert_chord_to_roman(self, chord_str: str, key_pc: int, mode: int = 1) -> str:
        root, quality = self.parse_chord_symbol(chord_str)
        if not root:
            return chord_str
        root_pc = self.note_to_pitch_class(root)
        if root_pc is None:
            return chord_str
        scale_degree = self.get_scale_degree(root_pc, key_pc)
        base_roman = (self.major_scale_romans if mode == 1 else self.minor_scale_romans).get(scale_degree, '?')
        modifier = self.chord_quality_to_roman_modifier(quality)
        return base_roman + modifier

    def convert_progression_to_roman_with_structure(self, elements: List[str], key_pc: int, mode: int = 1) -> List[str]:
        return [e if e.startswith('<') and e.endswith('>') else self.convert_chord_to_roman(e, key_pc, mode) for e in elements]

    def format_progression_with_key_context(self, roman_elements: List[str], key_name: str, mode_name: str) -> str:
        result = []
        section = []
        for el in roman_elements:
            if el.startswith('<'):
                if section:
                    result.append(' '.join(section))
                    section = []
                result.append(el)
            else:
                section.append(el)
        if section:
            result.append(' '.join(section))
        return f"Key of {key_name} {mode_name}: {' '.join(result)}"

    def create_harmonic_function_sequence(self, roman_progression: List[str]) -> str:
        function_map = {
            'I': 'T', 'i': 'T', 'ii': 'PD', 'ii°': 'PD', 'IV': 'PD', 'iv': 'PD',
            'V': 'D', 'V7': 'D', 'vii°': 'D', 'vi': 'T', 'VI': 'T', 'iii': 'T', 'III': 'T'
        }
        return ' - '.join([function_map.get(re.sub(r'[^IViv°]+', '', r), 'X') for r in roman_progression])

    def process_songs_for_roman_analysis(self, df):
        print("🎼 Converting chord progressions to roman numerals (preserving song structure)...")
        df['conversion_success'] = False
        df['harmonic_analysis_with_key'] = None
        df['harmonic_function_sequence'] = None
        df['key_used_for_analysis'] = None
        df['mode_used_for_analysis'] = None

        success = 0
        fail = 0

        for idx, row in df.iterrows():
            try:
                key_pc = None
                mode = 1
                if pd.notna(row.get('key_Kaggle')):
                    key_pc = int(row['key_Kaggle'])
                    mode = int(row.get('mode_Kaggle', 1))
                elif pd.notna(row.get('key_Maharshipandya')):
                    key_pc = int(row['key_Maharshipandya'])
                    mode = int(row.get('mode_Maharshipandya', 1))
                elif pd.notna(row.get('key')):
                    key_pc = int(row['key'])
                    mode = int(row.get('mode', 1))
                if key_pc is None:
                    fail += 1
                    continue
                chords = row.get('chords')
                if not chords:
                    fail += 1
                    continue
                elements = self.parse_chord_progression_with_structure(chords)
                roman_elements = self.convert_progression_to_roman_with_structure(elements, key_pc, mode)
                key_name = self.pitch_class_to_note.get(key_pc, '?')
                mode_name = 'Major' if mode == 1 else 'Minor'
                with_key = self.format_progression_with_key_context(roman_elements, key_name, mode_name)
                function_seq = self.create_harmonic_function_sequence([e for e in roman_elements if not e.startswith('<')])

                df.at[idx, 'conversion_success'] = True
                df.at[idx, 'harmonic_analysis_with_key'] = with_key
                df.at[idx, 'harmonic_function_sequence'] = function_seq
                df.at[idx, 'key_used_for_analysis'] = key_name
                df.at[idx, 'mode_used_for_analysis'] = mode_name
                success += 1
            except Exception as e:
                fail += 1
                self.loggers['conversion'].error(f"Failed conversion: {e}")

        print(f"✅ Roman numeral conversion complete!")
        print(f"   Success: {success:,} songs")
        print(f"   Failed: {fail:,} songs")
        return df

    def save_roman_analysis_database(self, df):
        print("💾 Creating roman numeral analysis database...")
        db_path = 'results/spotify_songs_with_roman_analysis.sqlite'
        try:
            conn = create_connection(db_path)
            if not conn:
                print("❌ Failed to create database connection")
                return False
            df.to_sql('songs_with_roman_analysis', conn, if_exists='replace', index=False)
            create_indexes(conn, 'songs_with_roman_analysis', [
                'spotify_id', 'key_used_for_analysis', 'mode_used_for_analysis',
                'harmonic_analysis_with_key', 'conversion_success'
            ])
            conn.close()
            print(f"✅ Saved roman numeral analysis database: {db_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving database: {e}")
            return False

    def run(self):
        print("\n🎼 Starting Roman Numeral Analysis Pipeline 🎼")
        print("=" * 60)
        df = self.load_source_database()
        if df is None:
            return False
        df = self.process_songs_for_roman_analysis(df)
        return self.save_roman_analysis_database(df)

def main():
    pipeline = RomanNumeralAnalysisPipeline()
    if pipeline.run():
        print("\n✅ Roman numeral analysis complete! Database 5 is ready for music research.")
    else:
        print("\n❌ Roman numeral analysis failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
