# ========================================
# FILE 2: utils/id_normalizer.py
# ========================================
"""
Spotify ID Normalization Utilities
Handles various Spotify ID formats and normalizes them to clean format.
"""

import re
import logging


def normalize_spotify_id(spotify_id):
    """
    Normalize Spotify ID by removing prefixes and cleaning format.

    Args:
        spotify_id (str): Raw Spotify ID in various formats

    Returns:
        tuple: (normalized_id, success_flag)
            - normalized_id: Clean Spotify ID or original if failed
            - success_flag: True if normalization succeeded, False otherwise
    """
    if not spotify_id or not isinstance(spotify_id, str):
        return spotify_id, False

    # Store original for fallback
    original_id = spotify_id

    try:
        # Trim whitespace
        cleaned_id = spotify_id.strip()

        # Convert to lowercase
        cleaned_id = cleaned_id.lower()

        # Remove common Spotify URL prefixes
        prefixes_to_remove = [
            'https://open.spotify.com/track/',
            'http://open.spotify.com/track/',
            'spotify:track:',
            'spotify:',
        ]

        for prefix in prefixes_to_remove:
            if cleaned_id.startswith(prefix):
                cleaned_id = cleaned_id[len(prefix):]
                break

        # Remove query parameters (everything after ?)
        if '?' in cleaned_id:
            cleaned_id = cleaned_id.split('?')[0]

        # Remove trailing slashes
        cleaned_id = cleaned_id.rstrip('/')

        # Validate format - Spotify track IDs should be 22 characters, alphanumeric
        if re.match(r'^[a-zA-Z0-9]{22}$', cleaned_id):
            return cleaned_id, True
        else:
            # If doesn't match expected format, check if it's close
            if len(cleaned_id) == 22 and re.match(r'^[a-zA-Z0-9]+$', cleaned_id):
                return cleaned_id, True
            else:
                # Return original if normalization resulted in invalid format
                return original_id, False

    except Exception as e:
        # Return original if any error occurs
        return original_id, False


def validate_spotify_id_format(spotify_id):
    """
    Validate if a string matches expected Spotify ID format.

    Args:
        spotify_id (str): Spotify ID to validate

    Returns:
        bool: True if valid format, False otherwise
    """
    if not spotify_id or not isinstance(spotify_id, str):
        return False

    # Standard Spotify track ID format: 22 alphanumeric characters
    return bool(re.match(r'^[a-zA-Z0-9]{22}$', spotify_id))


def extract_id_from_url(spotify_url):
    """
    Extract Spotify ID from various URL formats.

    Args:
        spotify_url (str): Spotify URL in various formats

    Returns:
        str or None: Extracted ID or None if not found
    """
    if not spotify_url or not isinstance(spotify_url, str):
        return None

    # Patterns to match Spotify URLs
    patterns = [
        r'https?://open\.spotify\.com/track/([a-zA-Z0-9]{22})',
        r'spotify:track:([a-zA-Z0-9]{22})',
        r'/track/([a-zA-Z0-9]{22})',
        r'track/([a-zA-Z0-9]{22})',
    ]

    for pattern in patterns:
        match = re.search(pattern, spotify_url)
        if match:
            return match.group(1)

    return None


def batch_normalize_ids(id_list):
    """
    Normalize a list of Spotify IDs in batch.

    Args:
        id_list (list): List of Spotify IDs to normalize

    Returns:
        dict: Results containing normalized IDs and statistics
    """
    results = {
        'normalized_ids': [],
        'original_ids': [],
        'success_count': 0,
        'failed_count': 0,
        'failed_ids': []
    }

    for original_id in id_list:
        normalized_id, success = normalize_spotify_id(original_id)

        results['normalized_ids'].append(normalized_id)
        results['original_ids'].append(original_id)

        if success:
            results['success_count'] += 1
        else:
            results['failed_count'] += 1
            results['failed_ids'].append(original_id)

    return results


# Test function for development
def test_normalization():
    """Test the normalization function with various inputs"""
    test_cases = [
        "https://open.spotify.com/track/4iV5W9uYEdYUVa79Axb7Rh",
        "spotify:track:4iV5W9uYEdYUVa79Axb7Rh",
        "4iV5W9uYEdYUVa79Axb7Rh",
        "  4iV5W9uYEdYUVa79Axb7Rh  ",
        "HTTPS://OPEN.SPOTIFY.COM/TRACK/4IV5W9UYEDYUVA79AXB7RH",
        "invalid_id",
        "",
        None,
        "4iV5W9uYEdYUVa79Axb7Rh?si=abc123",
        "https://open.spotify.com/track/4iV5W9uYEdYUVa79Axb7Rh/",
    ]

    print("Testing Spotify ID normalization:")
    print("-" * 50)

    for test_id in test_cases:
        normalized, success = normalize_spotify_id(test_id)
        status = "✓" if success else "✗"
        print(f"{status} '{test_id}' → '{normalized}' (Success: {success})")


if __name__ == "__main__":
    test_normalization()
