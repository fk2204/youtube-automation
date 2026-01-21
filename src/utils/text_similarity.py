"""
Text Similarity Utilities for Semantic Caching

This module provides various text similarity calculations for the semantic
caching system. It includes multiple algorithms that can be combined for
more accurate similarity detection.

Features:
- Word overlap (Jaccard) similarity
- Character n-gram similarity
- Levenshtein edit distance and similarity
- Combined weighted similarity

Usage:
    from src.utils.text_similarity import (
        word_overlap_similarity,
        character_ngram_similarity,
        levenshtein_similarity,
        combined_similarity,
    )

    # Calculate word overlap
    score = word_overlap_similarity("hello world", "hello there")

    # Calculate n-gram similarity
    score = character_ngram_similarity("hello world", "hello there", n=3)

    # Calculate Levenshtein similarity
    score = levenshtein_similarity("hello", "hallo")

    # Combined similarity with custom weights
    score = combined_similarity(text1, text2, weights={
        "word_overlap": 0.4,
        "ngram": 0.3,
        "levenshtein": 0.3,
    })
"""

import re
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words.

    Args:
        text: Input text to tokenize

    Returns:
        List of lowercase word tokens
    """
    # Convert to lowercase and extract words
    text = text.lower()
    # Remove punctuation and split on whitespace
    words = re.findall(r'\b[a-z0-9]+\b', text)
    return words


def get_word_set(text: str) -> Set[str]:
    """
    Get a set of unique words from text.

    Args:
        text: Input text

    Returns:
        Set of unique lowercase words
    """
    return set(tokenize(text))


def word_overlap_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity based on word overlap.

    Jaccard similarity = |A intersection B| / |A union B|

    This measures the proportion of shared words between two texts.
    A score of 1.0 means identical word sets, 0.0 means no overlap.

    Args:
        text1: First text to compare
        text2: Second text to compare

    Returns:
        Similarity score between 0.0 and 1.0

    Examples:
        >>> word_overlap_similarity("hello world", "hello there")
        0.333...  # 1 shared word out of 3 unique words
        >>> word_overlap_similarity("the quick brown fox", "the quick brown fox")
        1.0
        >>> word_overlap_similarity("hello", "goodbye")
        0.0
    """
    if not text1 or not text2:
        return 0.0

    words1 = get_word_set(text1)
    words2 = get_word_set(text2)

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0


def get_ngrams(text: str, n: int = 3) -> Set[str]:
    """
    Extract character n-grams from text.

    Args:
        text: Input text
        n: Size of n-grams (default 3 for trigrams)

    Returns:
        Set of n-gram strings

    Examples:
        >>> sorted(get_ngrams("hello", 2))
        ['el', 'he', 'll', 'lo']
    """
    text = text.lower().strip()
    # Pad text for edge n-grams
    padded = f" {text} "

    ngrams = set()
    for i in range(len(padded) - n + 1):
        ngrams.add(padded[i:i + n])

    return ngrams


def character_ngram_similarity(text1: str, text2: str, n: int = 3) -> float:
    """
    Calculate similarity based on character n-grams.

    Uses Jaccard similarity on n-gram sets. This method is more robust
    to minor spelling variations than word-based similarity.

    Args:
        text1: First text to compare
        text2: Second text to compare
        n: Size of n-grams (default 3 for trigrams)

    Returns:
        Similarity score between 0.0 and 1.0

    Examples:
        >>> character_ngram_similarity("hello", "hallo", n=2)
        0.5  # Shares 'al', 'lo' but not 'he', 'el', 'll', 'ha'
        >>> character_ngram_similarity("python", "python")
        1.0
    """
    if not text1 or not text2:
        return 0.0

    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)

    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2

    return len(intersection) / len(union) if union else 0.0


@lru_cache(maxsize=1024)
def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings.

    The Levenshtein distance is the minimum number of single-character
    edits (insertions, deletions, substitutions) needed to transform
    one string into another.

    Uses dynamic programming with space optimization (only stores
    two rows at a time).

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance (minimum number of edits)

    Examples:
        >>> levenshtein_distance("kitten", "sitting")
        3  # k->s, e->i, add g
        >>> levenshtein_distance("hello", "hello")
        0
        >>> levenshtein_distance("", "hello")
        5
    """
    # Ensure s1 is the shorter string for space efficiency
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    m, n = len(s1), len(s2)

    # Handle edge cases
    if m == 0:
        return n
    if n == 0:
        return m

    # Use two rows instead of full matrix for space efficiency
    # prev_row represents row i-1, curr_row represents row i
    prev_row = list(range(n + 1))
    curr_row = [0] * (n + 1)

    for i in range(1, m + 1):
        curr_row[0] = i

        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                # Characters match, no edit needed
                curr_row[j] = prev_row[j - 1]
            else:
                # Minimum of: insertion, deletion, substitution
                curr_row[j] = 1 + min(
                    curr_row[j - 1],    # Insertion
                    prev_row[j],        # Deletion
                    prev_row[j - 1],    # Substitution
                )

        # Swap rows
        prev_row, curr_row = curr_row, prev_row

    # Result is in prev_row (after final swap)
    return prev_row[n]


def levenshtein_similarity(text1: str, text2: str) -> float:
    """
    Convert Levenshtein distance to similarity score 0-1.

    Similarity = 1 - (distance / max_length)

    A score of 1.0 means identical strings, 0.0 means completely different.

    Args:
        text1: First text to compare
        text2: Second text to compare

    Returns:
        Similarity score between 0.0 and 1.0

    Examples:
        >>> levenshtein_similarity("hello", "hello")
        1.0
        >>> levenshtein_similarity("hello", "hallo")
        0.8  # 1 edit out of 5 characters
        >>> levenshtein_similarity("", "hello")
        0.0
    """
    if not text1 and not text2:
        return 1.0  # Both empty, considered identical

    if not text1 or not text2:
        return 0.0  # One empty, completely different

    # Normalize texts for comparison
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()

    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 1.0

    distance = levenshtein_distance(text1, text2)
    return 1.0 - (distance / max_len)


def sequence_matcher_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity using Python's difflib SequenceMatcher.

    This finds the longest contiguous matching subsequence that contains
    no "junk" elements, then recursively finds matches in the parts
    before and after.

    Args:
        text1: First text to compare
        text2: Second text to compare

    Returns:
        Similarity score between 0.0 and 1.0
    """
    try:
        from difflib import SequenceMatcher

        if not text1 or not text2:
            return 0.0 if text1 != text2 else 1.0

        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()

        return SequenceMatcher(None, text1, text2).ratio()
    except ImportError:
        # Fallback to Levenshtein if difflib not available
        return levenshtein_similarity(text1, text2)


def cosine_word_similarity(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity based on word frequency vectors.

    Treats each text as a bag of words and computes the cosine of
    the angle between their frequency vectors.

    Args:
        text1: First text to compare
        text2: Second text to compare

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0

    words1 = tokenize(text1)
    words2 = tokenize(text2)

    if not words1 or not words2:
        return 0.0

    # Build frequency dictionaries
    freq1: Dict[str, int] = {}
    freq2: Dict[str, int] = {}

    for word in words1:
        freq1[word] = freq1.get(word, 0) + 1

    for word in words2:
        freq2[word] = freq2.get(word, 0) + 1

    # Get all unique words
    all_words = set(freq1.keys()) | set(freq2.keys())

    # Calculate dot product and magnitudes
    dot_product = 0.0
    magnitude1 = 0.0
    magnitude2 = 0.0

    for word in all_words:
        v1 = freq1.get(word, 0)
        v2 = freq2.get(word, 0)

        dot_product += v1 * v2
        magnitude1 += v1 * v1
        magnitude2 += v2 * v2

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    import math
    return dot_product / (math.sqrt(magnitude1) * math.sqrt(magnitude2))


def prefix_similarity(text1: str, text2: str, prefix_length: int = 50) -> float:
    """
    Calculate similarity based on common prefix.

    Useful for prompts that often start with similar instructions.

    Args:
        text1: First text to compare
        text2: Second text to compare
        prefix_length: Length of prefix to compare

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0

    # Normalize texts
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()

    # Get prefixes
    prefix1 = text1[:prefix_length]
    prefix2 = text2[:prefix_length]

    # Calculate character overlap in prefix
    min_len = min(len(prefix1), len(prefix2))
    if min_len == 0:
        return 0.0

    matches = sum(1 for c1, c2 in zip(prefix1, prefix2) if c1 == c2)
    return matches / min_len


def combined_similarity(
    text1: str,
    text2: str,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Combine multiple similarity measures with weighted average.

    This provides a more robust similarity score by combining different
    algorithms that capture different aspects of similarity.

    Available methods:
    - "word_overlap": Jaccard similarity on words
    - "ngram": Character n-gram similarity
    - "levenshtein": Levenshtein edit distance similarity
    - "sequence": difflib SequenceMatcher ratio
    - "cosine": Cosine similarity on word frequencies
    - "prefix": Common prefix similarity

    Args:
        text1: First text to compare
        text2: Second text to compare
        weights: Dict mapping method names to weights (must sum to 1.0)
                Default: {"word_overlap": 0.3, "ngram": 0.3, "levenshtein": 0.2, "sequence": 0.2}

    Returns:
        Weighted similarity score between 0.0 and 1.0

    Examples:
        >>> combined_similarity("hello world", "hello there")
        0.4...  # Depends on weights
        >>> combined_similarity("python code", "python code")
        1.0
    """
    if weights is None:
        weights = {
            "word_overlap": 0.3,
            "ngram": 0.3,
            "levenshtein": 0.2,
            "sequence": 0.2,
        }

    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0

    weights = {k: v / total_weight for k, v in weights.items()}

    # Method name to function mapping
    methods = {
        "word_overlap": word_overlap_similarity,
        "ngram": lambda t1, t2: character_ngram_similarity(t1, t2, n=3),
        "levenshtein": levenshtein_similarity,
        "sequence": sequence_matcher_similarity,
        "cosine": cosine_word_similarity,
        "prefix": prefix_similarity,
    }

    combined_score = 0.0

    for method_name, weight in weights.items():
        if method_name in methods and weight > 0:
            score = methods[method_name](text1, text2)
            combined_score += score * weight

    return combined_score


def fast_similarity(text1: str, text2: str) -> float:
    """
    Fast similarity check for initial filtering.

    Uses quick heuristics before more expensive calculations.
    Good for filtering candidates before full similarity calculation.

    Args:
        text1: First text to compare
        text2: Second text to compare

    Returns:
        Approximate similarity score between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0

    # Quick length check
    len1, len2 = len(text1), len(text2)
    if len1 == 0 or len2 == 0:
        return 0.0

    length_ratio = min(len1, len2) / max(len1, len2)

    # If lengths are very different, texts are likely different
    if length_ratio < 0.5:
        return length_ratio * 0.5

    # Quick word overlap (faster than full Jaccard)
    words1 = set(text1.lower().split()[:20])  # First 20 words
    words2 = set(text2.lower().split()[:20])

    if not words1 or not words2:
        return length_ratio * 0.3

    overlap = len(words1 & words2)
    max_overlap = max(len(words1), len(words2))
    word_score = overlap / max_overlap if max_overlap > 0 else 0

    # Combine length ratio and word overlap
    return (length_ratio * 0.3 + word_score * 0.7)


def find_similar_texts(
    query: str,
    candidates: List[Tuple[str, any]],
    threshold: float = 0.8,
    max_results: int = 5,
    similarity_func: Optional[callable] = None,
) -> List[Tuple[str, any, float]]:
    """
    Find texts similar to query from a list of candidates.

    Args:
        query: The query text to find similar matches for
        candidates: List of (text, data) tuples to search
        threshold: Minimum similarity threshold (0.0-1.0)
        max_results: Maximum number of results to return
        similarity_func: Custom similarity function (default: combined_similarity)

    Returns:
        List of (text, data, similarity_score) tuples, sorted by score descending

    Examples:
        >>> candidates = [
        ...     ("Generate a title for python tutorial", "resp1"),
        ...     ("Generate a title for javascript guide", "resp2"),
        ...     ("Write a poem about nature", "resp3"),
        ... ]
        >>> results = find_similar_texts("Generate title for python video", candidates)
        >>> results[0][2]  # First result's similarity score
        0.85  # High similarity with python tutorial prompt
    """
    if similarity_func is None:
        similarity_func = combined_similarity

    results = []

    for text, data in candidates:
        # Use fast similarity for initial filtering
        quick_score = fast_similarity(query, text)

        # Skip obviously dissimilar texts
        if quick_score < threshold * 0.7:
            continue

        # Calculate full similarity
        score = similarity_func(query, text)

        if score >= threshold:
            results.append((text, data, score))

    # Sort by score descending
    results.sort(key=lambda x: x[2], reverse=True)

    return results[:max_results]


# ============================================================
# Specialized Similarity Functions for Prompts
# ============================================================

def prompt_similarity(prompt1: str, prompt2: str) -> float:
    """
    Calculate similarity optimized for AI prompts.

    This function is tuned for comparing prompts, which often have:
    - Similar instruction patterns
    - Variable content sections
    - Common template structures

    Args:
        prompt1: First prompt to compare
        prompt2: Second prompt to compare

    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Use weights optimized for prompts
    weights = {
        "word_overlap": 0.35,  # Important for instruction keywords
        "ngram": 0.25,         # Catches similar phrases
        "sequence": 0.25,      # Good for template matching
        "prefix": 0.15,        # Prompts often start similarly
    }

    return combined_similarity(prompt1, prompt2, weights)


def normalize_prompt_for_comparison(prompt: str) -> str:
    """
    Normalize a prompt for comparison by removing variable parts.

    This helps find similar prompts that differ only in specific
    values like topic names, numbers, etc.

    Args:
        prompt: The prompt to normalize

    Returns:
        Normalized prompt string
    """
    import re

    # Convert to lowercase
    normalized = prompt.lower()

    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)

    # Replace numbers with placeholder
    normalized = re.sub(r'\b\d+\b', '<NUM>', normalized)

    # Replace quoted strings with placeholder
    normalized = re.sub(r'"[^"]*"', '<QUOTED>', normalized)
    normalized = re.sub(r"'[^']*'", '<QUOTED>', normalized)

    # Remove common variable parts (URLs, paths, etc.)
    normalized = re.sub(r'https?://\S+', '<URL>', normalized)
    normalized = re.sub(r'\S+\.(com|org|net|io)\S*', '<URL>', normalized)

    # Strip and return
    return normalized.strip()


# ============================================================
# Test Functions
# ============================================================

if __name__ == "__main__":
    # Test the similarity functions
    print("=" * 50)
    print("TEXT SIMILARITY TESTS")
    print("=" * 50)

    # Test pairs
    test_pairs = [
        ("Generate a script about Python programming", "Generate a script about JavaScript programming"),
        ("Generate a title for: AI tutorial", "Generate a title for: Machine Learning guide"),
        ("Write 5 YouTube tags for a video about cooking", "Write 5 YouTube tags for a video about baking"),
        ("Create an engaging hook for a finance video", "Create an engaging hook for a money video"),
        ("hello world", "goodbye world"),
        ("the quick brown fox", "the quick brown fox"),
    ]

    print("\n--- Word Overlap Similarity ---")
    for t1, t2 in test_pairs:
        score = word_overlap_similarity(t1, t2)
        print(f"{score:.3f}: '{t1[:40]}...' vs '{t2[:40]}...'")

    print("\n--- Character N-gram Similarity (n=3) ---")
    for t1, t2 in test_pairs:
        score = character_ngram_similarity(t1, t2, n=3)
        print(f"{score:.3f}: '{t1[:40]}...' vs '{t2[:40]}...'")

    print("\n--- Levenshtein Similarity ---")
    for t1, t2 in test_pairs:
        score = levenshtein_similarity(t1, t2)
        print(f"{score:.3f}: '{t1[:40]}...' vs '{t2[:40]}...'")

    print("\n--- Combined Similarity ---")
    for t1, t2 in test_pairs:
        score = combined_similarity(t1, t2)
        print(f"{score:.3f}: '{t1[:40]}...' vs '{t2[:40]}...'")

    print("\n--- Prompt-Optimized Similarity ---")
    for t1, t2 in test_pairs:
        score = prompt_similarity(t1, t2)
        print(f"{score:.3f}: '{t1[:40]}...' vs '{t2[:40]}...'")

    print("\n--- Find Similar Texts ---")
    query = "Generate a title for a Python tutorial video"
    candidates = [
        ("Generate a title for a JavaScript tutorial", "js_response"),
        ("Generate a title for a Python guide", "py_response"),
        ("Write a description for a cooking channel", "cook_response"),
        ("Create tags for a programming video", "tags_response"),
        ("Generate a title for a Python programming tutorial", "py2_response"),
    ]

    results = find_similar_texts(query, candidates, threshold=0.5, max_results=3)
    print(f"\nQuery: '{query}'")
    print("Top matches:")
    for text, data, score in results:
        print(f"  {score:.3f}: '{text}' -> {data}")

    print("\n" + "=" * 50)
    print("Tests completed!")
