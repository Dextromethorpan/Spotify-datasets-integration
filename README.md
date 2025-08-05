The project was done to match three databases:
.https://huggingface.co/datasets/ailsntua/Chordonomicon/viewer/default/train,
.https://www.kaggle.com/datasets/priyamchoksi/spotify-dataset-114k-songs,
.https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/viewer.

In the end, there is one database that contain the interaction of the three databases. This one is:
results/spotify_songs_with_roman_analysis.sqlite.

'results/spotify_songs_with_roman_analysis.sqlite' was created by roman_numerals_analysis.py.

There is a need to check the conversion from chords into roman numerals.
For that, two functions were used:

```python
     def create_harmonic_function_sequence(self, roman_progression: List[str]) -> str:
        function_map = {
            'I': 'T', 'i': 'T', 'ii': 'PD', 'ii°': 'PD', 'IV': 'PD', 'iv': 'PD',
            'V': 'D', 'V7': 'D', 'vii°': 'D', 'vi': 'T', 'VI': 'T', 'iii': 'T', 'III': 'T'
        }
        return ' - '.join([function_map.get(re.sub(r'[^IViv°]+', '', r), 'X') for r in roman_progression])
```


```python
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
```
