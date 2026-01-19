"""
Script Validator Module

Validates, cleans, and improves YouTube scripts to ensure professional quality
and TTS compatibility. Catches common AI-generated script issues.

Usage:
    from src.content.script_validator import ScriptValidator

    validator = ScriptValidator()

    # Clean a script
    cleaned_text = validator.clean_script(raw_script)

    # Validate a script
    result = validator.validate_script(script_text, niche="finance")
    if not result.is_valid:
        print(f"Issues found: {result.issues}")

    # Improve a script
    improved_text = validator.improve_script(script_text, niche="finance")

    # Check TTS compatibility
    tts_issues = validator.check_tts_compatibility(script_text)
"""

import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ValidationResult:
    """Result of script validation."""
    is_valid: bool
    score: int  # 0-100
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "score": self.score,
            "issues": self.issues,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "stats": self.stats
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        status = "VALID" if self.is_valid else "INVALID"
        lines = [
            f"Script Validation: {status} (Score: {self.score}/100)",
            ""
        ]

        if self.issues:
            lines.append(f"Issues ({len(self.issues)}):")
            for issue in self.issues:
                lines.append(f"  [X] {issue}")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  [!] {warning}")

        if self.suggestions:
            lines.append(f"\nSuggestions ({len(self.suggestions)}):")
            for suggestion in self.suggestions[:5]:
                lines.append(f"  [i] {suggestion}")

        if self.stats:
            lines.append(f"\nStats:")
            for key, value in self.stats.items():
                lines.append(f"  - {key}: {value}")

        return "\n".join(lines)


class ScriptValidator:
    """
    Validates, cleans, and improves YouTube scripts for professional quality
    and TTS compatibility.

    Features:
    - Remove timestamps, formatting artifacts, stage directions
    - Validate word count, sentence length, hook presence
    - Improve scripts for natural TTS delivery
    - Check for awkward TTS phrases
    """

    # ============================================================
    # Patterns to Clean/Remove
    # ============================================================

    # Timestamp patterns (various formats)
    TIMESTAMP_PATTERNS = [
        r'\[?\d{1,2}:\d{2}(?::\d{2})?\]?',  # [00:00], 0:00, 00:00:00
        r'\[\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\]',  # [00:00-00:15]
        r'\(\d{1,2}:\d{2}\)',  # (00:00)
        r'\d{1,2}:\d{2}(?::\d{2})?\s*[-:]\s*',  # 00:00 - or 00:00:
        r'^\s*\d{1,2}:\d{2}\s+',  # Line starting with timestamp
    ]

    # Chapter markers that shouldn't be read
    CHAPTER_MARKER_PATTERNS = [
        r'^#{1,3}\s*Chapter\s*\d*:?\s*',  # ### Chapter 1:
        r'^Chapter\s+\d+:?\s*',  # Chapter 1:
        r'^\*\*Chapter\s+\d+\*\*',  # **Chapter 1**
        r'^Section\s+\d+:?\s*',  # Section 1:
        r'^Part\s+\d+:?\s*',  # Part 1:
    ]

    # Markdown formatting
    MARKDOWN_PATTERNS = [
        r'\*\*([^*]+)\*\*',  # **bold** -> text
        r'\*([^*]+)\*',  # *italic* -> text
        r'__([^_]+)__',  # __bold__ -> text
        r'_([^_]+)_',  # _italic_ -> text
        r'^#{1,6}\s*',  # # Headers
        r'\[([^\]]+)\]\([^)]+\)',  # [text](url) -> text
        r'```[^`]*```',  # Code blocks
        r'`([^`]+)`',  # `inline code` -> text
        r'^[-*+]\s+',  # Bullet points
        r'^\d+\.\s+',  # Numbered lists
        r'^>\s*',  # Block quotes
    ]

    # Stage directions and production notes
    STAGE_DIRECTION_PATTERNS = [
        r'\[pause\]',
        r'\[music\]',
        r'\[sfx[^\]]*\]',
        r'\[sound[^\]]*\]',
        r'\[cut\]',
        r'\[transition\]',
        r'\[zoom\]',
        r'\[pan\]',
        r'\[fade\]',
        r'\[b-?roll[^\]]*\]',
        r'\[broll[^\]]*\]',
        r'\[text[^\]]*\]',
        r'\[graphic[^\]]*\]',
        r'\[visual[^\]]*\]',
        r'\[animation[^\]]*\]',
        r'\[screen[^\]]*\]',
        r'\[show[^\]]*\]',
        r'\[display[^\]]*\]',
        r'\[insert[^\]]*\]',
        r'\[overlay[^\]]*\]',
        r'\[lower third[^\]]*\]',
        r'\[title card[^\]]*\]',
        r'\[on-?screen[^\]]*\]',
        r'\[end card[^\]]*\]',
        r'\[subscribe[^\]]*\]',
        r'\[like[^\]]*\]',
        r'\[bell[^\]]*\]',
        r'\[outro[^\]]*\]',
        r'\[intro[^\]]*\]',
        r'\[hook[^\]]*\]',
        r'\[cta[^\]]*\]',
        r'\[action[^\]]*\]',
        r'\[note[^\]]*\]',
        r'\[edit[^\]]*\]',
        r'\[camera[^\]]*\]',
        r'\[close-?up[^\]]*\]',
        r'\[wide shot[^\]]*\]',
        r'\(pause\)',
        r'\(beat\)',
        r'\(silence\)',
        r'\.\.\.\s*\[',  # ... [something]
    ]

    # Meta-text phrases that shouldn't be spoken
    META_TEXT_PATTERNS = [
        r"In this (?:section|video|part|episode|chapter),?\s*(?:we'll|we will|I'll|I will|you'll|you will)?",
        r"As (?:I |we )?mentioned (?:earlier|before|previously|above)",
        r"As (?:I |we )?(?:said|stated|noted) (?:earlier|before|previously)",
        r"Let me (?:explain|tell you|show you|break down)",
        r"(?:Now |So |Okay |Alright ),?let's (?:talk about|discuss|look at|dive into|get into|move on to)",
        r"Moving on to",
        r"(?:Now |Next ),?(?:we'll|we will|I'll|I will) (?:discuss|talk about|cover|explore)",
        r"(?:First|Second|Third|Fourth|Fifth|Finally|Lastly),?\s*(?:up|off|of all)?,?\s*(?:we have|is|let's)",
        r"Without further ado",
        r"Let's (?:get started|begin|dive (?:right )?in)",
        r"(?:So |Now )?what (?:is|are) (?:we |you )?(?:going to |gonna )?(?:talk|learn|discuss|cover)",
        r"Before (?:we |I )?(?:begin|start|get started|dive in|continue)",
        r"(?:So |Now )?let's (?:jump|hop|get) (?:right )?into (?:it|this|that)",
        r"In (?:today's|this) (?:video|episode|tutorial)",
        r"Welcome (?:back )?to (?:the|my|our) (?:channel|video|tutorial)",
        r"(?:Hey |Hi )?(?:everyone|guys|folks|there),?\s*(?:welcome|it's|I'm)",
        r"Don't forget to (?:like|subscribe|hit|smash|click)",
        r"(?:Hit|Smash|Click|Press) (?:the|that) (?:like|subscribe|notification|bell)",
        r"(?:Make sure (?:to|you) )?(?:like|subscribe|comment|share)",
        r"If you (?:enjoyed|liked|found this helpful)",
        r"See you in the next (?:video|one|episode)",
        r"Thanks for watching",
        r"Until next time",
        r"(?:That's|This) (?:all|it) for (?:today|now|this video)",
        r"Let me know (?:in the comments|what you think|your thoughts)",
    ]

    # Filler words and phrases
    FILLER_PATTERNS = [
        r'\b(?:um|uh|uhm|umm|er|erm|ah|ahh)\b',
        r'\byou know\b',
        r'\bI mean\b',
        r'\blike,?\s+(?=\w)',  # "like" as filler, not comparison
        r'\bbasically\b',
        r'\bactually,?\s*(?=\w)',  # "actually" at start of sentence
        r'\bliterally\b',
        r'\bhonestly\b',
        r'\bso,?\s+(?=anyway|basically|yeah)',
        r'\bjust\s+(?=kind of|sort of)',
        r'\bkind of\b',
        r'\bsort of\b',
        r'\bI guess\b',
        r'\bI think,?\s*(?=I think)',  # repeated "I think"
        r'\breally,?\s+really\b',  # repeated "really"
        r'\bvery,?\s+very\b',  # repeated "very"
        r'\bpretty much\b',
        r'\bat the end of the day\b',
        r'\bto be honest\b',
        r'\bto be fair\b',
        r'\bin my opinion\b',
        r'\bif you will\b',
        r'\bper se\b',
        r'\bas such\b',
    ]

    # URL and link patterns
    URL_PATTERNS = [
        r'https?://[^\s<>"{}|\\^`\[\]]+',
        r'www\.[^\s<>"{}|\\^`\[\]]+',
        r'[a-zA-Z0-9.-]+\.(com|org|net|io|co|ai|edu|gov|info|biz|me|tv|app|dev|tech|store|shop|blog)(?:/[^\s]*)?',
    ]

    # Emoji pattern (matches most emoji)
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )

    # ============================================================
    # TTS Problem Patterns
    # ============================================================

    # Awkward phrases for TTS
    TTS_AWKWARD_PATTERNS = [
        (r'\b(?:e\.g\.|i\.e\.|etc\.)\b', "Use full phrases instead of abbreviations"),
        (r'\b(?:vs\.?|versus)\b', "Spell out 'versus' completely"),
        (r'\b(?:approx\.?|approximately)\b', "Use 'about' or 'around' instead"),
        (r'\b(?:govt\.?|government)\b', "Avoid abbreviations"),
        (r'\b(?:dept\.?|department)\b', "Avoid abbreviations"),
        (r'\b\d+(?:st|nd|rd|th)\b', "Ordinals may be mispronounced"),
        (r'\$\d+(?:,\d{3})*(?:\.\d{2})?[kKmMbB]?\b', "Currency with K/M/B may be mispronounced"),
        (r'\b\d+%', "Percentages should be written as 'X percent'"),
        (r'\b(?:Dr\.|Mr\.|Mrs\.|Ms\.)\b', "Titles may be mispronounced"),
        (r'\b[A-Z]{2,}\b', "Acronyms may be mispronounced"),
        (r'[/\\]', "Slashes may cause TTS issues"),
        (r'\([^)]{50,}\)', "Long parenthetical may cause awkward pauses"),
        (r';\s*', "Semicolons may cause unnatural pauses"),
        (r':\s*(?=[a-z])', "Colons before lowercase may be awkward"),
        (r'—|--', "Em dashes may cause unnatural pauses"),
        (r'\.{3,}', "Multiple periods (ellipsis) may be read awkwardly"),
        (r'\?{2,}', "Multiple question marks are unnatural"),
        (r'!{2,}', "Multiple exclamation marks are unnatural"),
        (r'["\'][^"\']{100,}["\']', "Very long quotes may be awkward"),
    ]

    # ============================================================
    # Validation Thresholds
    # ============================================================

    # Word count requirements
    MIN_WORD_COUNT_REGULAR = 400  # Minimum for regular videos
    MAX_WORD_COUNT_REGULAR = 2000  # Maximum for regular videos
    MIN_WORD_COUNT_SHORT = 30  # Minimum for Shorts
    MAX_WORD_COUNT_SHORT = 150  # Maximum for Shorts

    # Sentence requirements
    MAX_SENTENCE_LENGTH = 35  # Words per sentence (TTS clarity)
    OPTIMAL_SENTENCE_LENGTH = 15  # Optimal for TTS

    # Hook requirements
    HOOK_MAX_SECONDS = 5  # Hook should be in first 5 seconds
    HOOK_MAX_WORDS = 25  # Approximately 5 seconds of speech

    # CTA requirements
    CTA_MIN_POSITION_PERCENT = 25  # CTA shouldn't appear before 25%

    # Validation threshold
    PASSING_SCORE = 70

    def __init__(self, keep_emoji: bool = False):
        """
        Initialize the ScriptValidator.

        Args:
            keep_emoji: If True, don't remove emoji from scripts
        """
        self.keep_emoji = keep_emoji
        self._compile_patterns()
        logger.info("ScriptValidator initialized")

    def _compile_patterns(self):
        """Compile all regex patterns for efficiency."""
        # Compile timestamp patterns
        self._timestamp_regex = [re.compile(p, re.IGNORECASE | re.MULTILINE)
                                  for p in self.TIMESTAMP_PATTERNS]

        # Compile chapter marker patterns
        self._chapter_regex = [re.compile(p, re.IGNORECASE | re.MULTILINE)
                                for p in self.CHAPTER_MARKER_PATTERNS]

        # Compile markdown patterns
        self._markdown_regex = [(re.compile(p, re.MULTILINE), r'\1' if '(' in p else '')
                                 for p in self.MARKDOWN_PATTERNS]

        # Compile stage direction patterns
        self._stage_regex = [re.compile(p, re.IGNORECASE)
                              for p in self.STAGE_DIRECTION_PATTERNS]

        # Compile meta-text patterns
        self._meta_regex = [re.compile(p, re.IGNORECASE)
                             for p in self.META_TEXT_PATTERNS]

        # Compile filler patterns
        self._filler_regex = [re.compile(p, re.IGNORECASE)
                               for p in self.FILLER_PATTERNS]

        # Compile URL patterns
        self._url_regex = [re.compile(p, re.IGNORECASE)
                            for p in self.URL_PATTERNS]

        # Compile TTS awkward patterns
        self._tts_awkward_regex = [(re.compile(p, re.IGNORECASE), msg)
                                    for p, msg in self.TTS_AWKWARD_PATTERNS]

    # ============================================================
    # Main Public Methods
    # ============================================================

    def clean_script(self, text: str) -> str:
        """
        Clean a script by removing all unwanted elements.

        Removes:
        - Timestamps and timelines
        - Chapter markers
        - Markdown formatting
        - Stage directions
        - Meta-text
        - Filler phrases
        - URLs
        - Emoji (unless keep_emoji=True)

        Args:
            text: Raw script text

        Returns:
            Cleaned script text ready for TTS
        """
        if not text:
            return ""

        cleaned = text

        # Remove timestamps
        cleaned = self.remove_timestamps(cleaned)

        # Remove chapter markers
        for regex in self._chapter_regex:
            cleaned = regex.sub('', cleaned)

        # Clean markdown (replace with captured text where applicable)
        for regex, replacement in self._markdown_regex:
            cleaned = regex.sub(replacement, cleaned)

        # Remove stage directions
        for regex in self._stage_regex:
            cleaned = regex.sub('', cleaned)

        # Remove meta-text
        cleaned = self.remove_meta_text(cleaned)

        # Remove filler words
        for regex in self._filler_regex:
            cleaned = regex.sub('', cleaned)

        # Remove URLs
        for regex in self._url_regex:
            cleaned = regex.sub('', cleaned)

        # Remove emoji (unless keeping them)
        if not self.keep_emoji:
            cleaned = self.EMOJI_PATTERN.sub('', cleaned)

        # Clean up whitespace
        cleaned = self._normalize_whitespace(cleaned)

        # Remove duplicate sentences
        cleaned = self._remove_duplicate_sentences(cleaned)

        return cleaned.strip()

    def validate_script(
        self,
        text: str,
        niche: str = "default",
        is_short: bool = False
    ) -> ValidationResult:
        """
        Validate a script for quality and TTS compatibility.

        Checks:
        - Word count (minimum and maximum)
        - Sentence length
        - Hook presence in first sentences
        - CTA placement (not too early)
        - Duplicate sentences
        - TTS compatibility issues

        Args:
            text: Script text to validate
            niche: Content niche (finance, psychology, storytelling)
            is_short: Whether this is a YouTube Short

        Returns:
            ValidationResult with score, issues, warnings, and suggestions
        """
        issues = []
        warnings = []
        suggestions = []
        stats = {}

        # Calculate initial score
        score = 100

        # Get word and sentence counts
        words = text.split()
        word_count = len(words)
        sentences = self._split_sentences(text)
        sentence_count = len(sentences)

        stats["word_count"] = word_count
        stats["sentence_count"] = sentence_count
        stats["niche"] = niche
        stats["is_short"] = is_short

        # ============================================================
        # Word Count Validation
        # ============================================================
        if is_short:
            min_words = self.MIN_WORD_COUNT_SHORT
            max_words = self.MAX_WORD_COUNT_SHORT
        else:
            min_words = self.MIN_WORD_COUNT_REGULAR
            max_words = self.MAX_WORD_COUNT_REGULAR

        if word_count < min_words:
            issues.append(f"Script too short: {word_count} words (minimum: {min_words})")
            score -= 20
        elif word_count > max_words:
            warnings.append(f"Script very long: {word_count} words (recommended max: {max_words})")
            score -= 5

        # ============================================================
        # Sentence Length Validation
        # ============================================================
        long_sentences = []
        for i, sentence in enumerate(sentences):
            sentence_words = len(sentence.split())
            if sentence_words > self.MAX_SENTENCE_LENGTH:
                long_sentences.append((i + 1, sentence_words, sentence[:50]))

        if long_sentences:
            stats["long_sentences"] = len(long_sentences)
            if len(long_sentences) > 3:
                issues.append(f"{len(long_sentences)} sentences too long for TTS (max {self.MAX_SENTENCE_LENGTH} words)")
                score -= 10
            else:
                warnings.append(f"{len(long_sentences)} sentences could be shorter for better TTS")
                score -= 5
            for sent_num, sent_len, preview in long_sentences[:3]:
                suggestions.append(f"Sentence {sent_num} ({sent_len} words): Break up '{preview}...'")

        # Calculate average sentence length
        if sentence_count > 0:
            avg_sentence_len = word_count / sentence_count
            stats["avg_sentence_length"] = round(avg_sentence_len, 1)
            if avg_sentence_len > 25:
                warnings.append(f"Average sentence length high ({avg_sentence_len:.1f} words)")
                suggestions.append("Use shorter sentences for better TTS clarity")

        # ============================================================
        # Hook Validation (first 2 sentences)
        # ============================================================
        if sentence_count >= 2:
            hook_text = ' '.join(sentences[:2])
            hook_words = len(hook_text.split())
            stats["hook_word_count"] = hook_words

            # Check for engaging hook elements
            hook_lower = hook_text.lower()
            has_question = '?' in hook_text
            has_you = 'you' in hook_lower
            has_number = bool(re.search(r'\d+', hook_text))
            has_power_word = any(pw in hook_lower for pw in [
                'secret', 'shocking', 'truth', 'never', 'always', 'mistake',
                'wrong', 'hidden', 'revealed', 'proven', 'exactly'
            ])

            hook_elements = sum([has_question, has_you, has_number, has_power_word])
            stats["hook_elements"] = hook_elements

            if hook_elements < 2:
                warnings.append("Hook may be weak (missing engagement triggers)")
                suggestions.append("Add question, 'you', specific number, or power word to hook")
                score -= 5

            if hook_words > self.HOOK_MAX_WORDS * 2:
                warnings.append(f"Hook section too long ({hook_words} words)")
                suggestions.append(f"Keep first 2 sentences under {self.HOOK_MAX_WORDS * 2} words total")
                score -= 3

        # ============================================================
        # CTA Placement Validation
        # ============================================================
        cta_patterns = [
            r'\b(?:subscribe|like|comment|share|hit|smash|bell)\b',
            r'\bnotification\b',
            r'\bfollow\s+(?:us|me)\b'
        ]
        cta_regex = re.compile('|'.join(cta_patterns), re.IGNORECASE)

        # Find first CTA position
        cta_match = cta_regex.search(text)
        if cta_match and word_count > 0:
            cta_position = len(text[:cta_match.start()].split())
            cta_percent = (cta_position / word_count) * 100
            stats["cta_position_percent"] = round(cta_percent, 1)

            if cta_percent < self.CTA_MIN_POSITION_PERCENT:
                issues.append(f"CTA too early at {cta_percent:.0f}% (should be after {self.CTA_MIN_POSITION_PERCENT}%)")
                suggestions.append("Move subscribe/like CTA to after 30% of the script")
                score -= 10

        # ============================================================
        # Duplicate Sentence Check
        # ============================================================
        sentence_set = set()
        duplicates = []
        for sentence in sentences:
            normalized = sentence.lower().strip()
            if len(normalized) > 20:  # Only check substantial sentences
                if normalized in sentence_set:
                    duplicates.append(sentence[:50])
                else:
                    sentence_set.add(normalized)

        if duplicates:
            stats["duplicate_sentences"] = len(duplicates)
            issues.append(f"{len(duplicates)} duplicate sentence(s) found")
            score -= len(duplicates) * 3
            for dup in duplicates[:2]:
                suggestions.append(f"Remove duplicate: '{dup}...'")

        # ============================================================
        # TTS Compatibility Check
        # ============================================================
        tts_issues = self.check_tts_compatibility(text)
        if tts_issues:
            stats["tts_issues"] = len(tts_issues)
            if len(tts_issues) > 5:
                warnings.append(f"{len(tts_issues)} potential TTS pronunciation issues")
                score -= 8
            elif tts_issues:
                warnings.append(f"{len(tts_issues)} potential TTS issues found")
                score -= 3
            for issue in tts_issues[:3]:
                suggestions.append(f"TTS: {issue}")

        # ============================================================
        # Check for Remaining Artifacts
        # ============================================================
        artifact_checks = [
            (r'\[.*?\]', "Square bracket text found (stage directions?)"),
            (r'\(.*?pause.*?\)', "Pause direction found"),
            (r'\*\*.*?\*\*', "Bold markdown found"),
            (r'^#+ ', "Header markdown found"),
        ]
        for pattern, message in artifact_checks:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                warnings.append(message)
                score -= 2

        # ============================================================
        # Repetitive Phrase Check
        # ============================================================
        repetitive = self._find_repetitive_phrases(text)
        if repetitive:
            stats["repetitive_phrases"] = len(repetitive)
            warnings.append(f"{len(repetitive)} repetitive phrases found")
            for phrase, count in repetitive[:3]:
                suggestions.append(f"Phrase '{phrase}' appears {count} times - vary language")
            score -= len(repetitive) * 2

        # ============================================================
        # Final Score and Result
        # ============================================================
        score = max(0, min(100, score))
        is_valid = score >= self.PASSING_SCORE and len(issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            score=score,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
            stats=stats
        )

    def improve_script(self, text: str, niche: str = "default") -> str:
        """
        Improve a script for better TTS delivery and engagement.

        Improvements:
        - Break long sentences
        - Add natural pauses (commas, periods)
        - Fix common TTS issues
        - Ensure conversational tone
        - Remove redundancy

        Args:
            text: Script text to improve
            niche: Content niche for context

        Returns:
            Improved script text
        """
        if not text:
            return ""

        # First, clean the script
        improved = self.clean_script(text)

        # Break long sentences
        improved = self._break_long_sentences(improved)

        # Fix TTS-unfriendly patterns
        improved = self._fix_tts_patterns(improved)

        # Add natural pauses where needed
        improved = self._add_natural_pauses(improved)

        # Normalize whitespace again
        improved = self._normalize_whitespace(improved)

        return improved.strip()

    def remove_timestamps(self, text: str) -> str:
        """
        Remove all timestamp patterns from text.

        Handles:
        - [00:00]
        - 0:00
        - 00:00:00
        - 00:00 - 00:15
        - (00:00)
        - Lines starting with timestamps

        Args:
            text: Text with timestamps

        Returns:
            Text with timestamps removed
        """
        cleaned = text
        for regex in self._timestamp_regex:
            cleaned = regex.sub('', cleaned)
        return cleaned

    def remove_meta_text(self, text: str) -> str:
        """
        Remove meta-text phrases that shouldn't be spoken.

        Removes:
        - "In this section..."
        - "As mentioned earlier..."
        - "Let's talk about..."
        - Welcome/outro phrases
        - Subscribe/like reminders

        Args:
            text: Text with meta-text

        Returns:
            Text with meta-text removed
        """
        cleaned = text
        for regex in self._meta_regex:
            cleaned = regex.sub('', cleaned)
        return cleaned

    def check_tts_compatibility(self, text: str) -> List[str]:
        """
        Check text for TTS pronunciation issues.

        Checks for:
        - Abbreviations (e.g., i.e., etc.)
        - Acronyms
        - Currency with K/M/B
        - Percentages without "percent"
        - Slashes
        - Multiple punctuation
        - Long parentheticals

        Args:
            text: Text to check

        Returns:
            List of TTS compatibility issues found
        """
        issues = []

        for regex, message in self._tts_awkward_regex:
            matches = regex.findall(text)
            if matches:
                # Limit to first 3 unique matches
                unique_matches = list(set(matches))[:3]
                for match in unique_matches:
                    issues.append(f"{message}: '{match}'")

        return issues

    # ============================================================
    # Helper Methods
    # ============================================================

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove spaces before punctuation
        text = re.sub(r' +([.,!?;:])', r'\1', text)
        # Ensure space after punctuation (except for abbreviations)
        text = re.sub(r'([.,!?;:])(?=[A-Za-z])', r'\1 ', text)
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter
        # Handle abbreviations and decimal numbers
        text = re.sub(r'(\d)\.(\d)', r'\1<DECIMAL>\2', text)
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Co|vs|etc|e\.g|i\.e)\.', r'\1<ABBR>', text, flags=re.IGNORECASE)

        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+\s+', text)

        # Restore abbreviations and decimals
        sentences = [s.replace('<DECIMAL>', '.').replace('<ABBR>', '.') for s in sentences]

        # Filter out empty sentences
        return [s.strip() for s in sentences if s.strip()]

    def _remove_duplicate_sentences(self, text: str) -> str:
        """Remove duplicate sentences from text."""
        sentences = self._split_sentences(text)
        seen = set()
        unique = []

        for sentence in sentences:
            normalized = sentence.lower().strip()
            if normalized not in seen or len(normalized) < 20:
                unique.append(sentence)
                seen.add(normalized)

        return '. '.join(unique)

    def _find_repetitive_phrases(self, text: str, min_length: int = 4, min_count: int = 3) -> List[Tuple[str, int]]:
        """Find phrases that repeat too often."""
        words = text.lower().split()
        phrase_counts = {}

        # Check n-grams of different lengths
        for n in range(min_length, min_length + 3):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i + n])
                # Skip phrases that are mostly stop words
                if not self._is_meaningful_phrase(phrase):
                    continue
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        # Filter to repetitive phrases
        repetitive = [(p, c) for p, c in phrase_counts.items() if c >= min_count]
        repetitive.sort(key=lambda x: x[1], reverse=True)

        return repetitive[:5]  # Return top 5

    def _is_meaningful_phrase(self, phrase: str) -> bool:
        """Check if phrase is meaningful (not mostly stop words)."""
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about',
                      'into', 'over', 'after', 'and', 'but', 'or', 'as', 'if',
                      'than', 'because', 'while', 'although', 'though', 'that',
                      'which', 'who', 'whom', 'this', 'these', 'those', 'it', 'its'}

        words = phrase.split()
        non_stop = [w for w in words if w not in stop_words]
        return len(non_stop) >= len(words) * 0.4

    def _break_long_sentences(self, text: str) -> str:
        """Break long sentences into shorter ones."""
        sentences = self._split_sentences(text)
        result = []

        for sentence in sentences:
            words = sentence.split()
            if len(words) <= self.MAX_SENTENCE_LENGTH:
                result.append(sentence)
            else:
                # Try to break at natural points
                broken = self._smart_break_sentence(sentence)
                result.extend(broken)

        return '. '.join(result)

    def _smart_break_sentence(self, sentence: str) -> List[str]:
        """Break a long sentence at natural points."""
        words = sentence.split()
        if len(words) <= self.MAX_SENTENCE_LENGTH:
            return [sentence]

        # Look for break points
        break_points = [
            r'\s+(?:and|but|or|so|yet|however|therefore|moreover|furthermore|additionally)\s+',
            r'\s*,\s*(?:which|who|that|where|when)\s+',
            r'\s*,\s+',
            r'\s+-\s+',
        ]

        for pattern in break_points:
            parts = re.split(pattern, sentence, maxsplit=1)
            if len(parts) == 2 and all(len(p.split()) >= 5 for p in parts):
                # Recursively break if still too long
                result = []
                for part in parts:
                    result.extend(self._smart_break_sentence(part.strip()))
                return result

        # If no good break point, just split at middle
        mid = len(words) // 2
        return [
            ' '.join(words[:mid]),
            ' '.join(words[mid:])
        ]

    def _fix_tts_patterns(self, text: str) -> str:
        """Fix patterns that cause TTS issues."""
        # Replace abbreviations
        replacements = [
            (r'\be\.g\.\b', 'for example'),
            (r'\bi\.e\.\b', 'that is'),
            (r'\betc\.\b', 'and so on'),
            (r'\bvs\.?\b', 'versus'),
            (r'\bapprox\.?\b', 'approximately'),
            (r'(\d+)%', r'\1 percent'),
            (r'\$(\d+)k\b', r'$\1 thousand', re.IGNORECASE),
            (r'\$(\d+)m\b', r'$\1 million', re.IGNORECASE),
            (r'\$(\d+)b\b', r'$\1 billion', re.IGNORECASE),
            (r'\.{3,}', '.'),  # Multiple periods to single
            (r'\?{2,}', '?'),  # Multiple question marks
            (r'!{2,}', '!'),  # Multiple exclamation marks
            (r'—', ', '),  # Em dash to comma
            (r'--', ', '),  # Double dash to comma
        ]

        fixed = text
        for pattern, replacement, *flags in replacements:
            flag = flags[0] if flags else 0
            fixed = re.sub(pattern, replacement, fixed, flags=flag)

        return fixed

    def _add_natural_pauses(self, text: str) -> str:
        """Add natural pauses for better TTS delivery."""
        # Add commas after introductory phrases
        introductory_phrases = [
            r'^(However)\s+',
            r'^(Therefore)\s+',
            r'^(Furthermore)\s+',
            r'^(Moreover)\s+',
            r'^(In fact)\s+',
            r'^(Actually)\s+',
            r'^(Surprisingly)\s+',
            r'^(Interestingly)\s+',
            r'^(First)\s+(?=[A-Z])',
            r'^(Second)\s+(?=[A-Z])',
            r'^(Third)\s+(?=[A-Z])',
            r'^(Finally)\s+(?=[A-Z])',
            r'^(Now)\s+(?=[A-Z])',
            r'^(So)\s+(?=[A-Z])',
        ]

        result = text
        for pattern in introductory_phrases:
            result = re.sub(pattern, r'\1, ', result, flags=re.MULTILINE)

        return result


# ============================================================
# Convenience Functions
# ============================================================

def clean_script(text: str, keep_emoji: bool = False) -> str:
    """
    Quick function to clean a script.

    Args:
        text: Raw script text
        keep_emoji: Whether to keep emoji

    Returns:
        Cleaned script text
    """
    validator = ScriptValidator(keep_emoji=keep_emoji)
    return validator.clean_script(text)


def validate_script(text: str, niche: str = "default", is_short: bool = False) -> ValidationResult:
    """
    Quick function to validate a script.

    Args:
        text: Script text to validate
        niche: Content niche
        is_short: Whether this is a YouTube Short

    Returns:
        ValidationResult
    """
    validator = ScriptValidator()
    return validator.validate_script(text, niche=niche, is_short=is_short)


def improve_script(text: str, niche: str = "default") -> str:
    """
    Quick function to improve a script.

    Args:
        text: Script text to improve
        niche: Content niche

    Returns:
        Improved script text
    """
    validator = ScriptValidator()
    return validator.improve_script(text, niche=niche)


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("""
Script Validator - Clean and validate YouTube scripts

Usage:
    python -m src.content.script_validator <script_file> [options]

Options:
    --clean       Clean the script (remove timestamps, formatting, etc.)
    --validate    Validate the script (check word count, TTS compatibility, etc.)
    --improve     Improve the script (break sentences, fix TTS issues, etc.)
    --niche       Content niche: finance, psychology, storytelling, default
    --short       Validate as YouTube Short
    --json        Output as JSON

Examples:
    python -m src.content.script_validator script.txt --clean
    python -m src.content.script_validator script.txt --validate --niche finance
    python -m src.content.script_validator script.txt --improve
        """)
        sys.exit(0)

    script_file = sys.argv[1]
    do_clean = "--clean" in sys.argv
    do_validate = "--validate" in sys.argv
    do_improve = "--improve" in sys.argv
    is_short = "--short" in sys.argv
    output_json = "--json" in sys.argv

    # Get niche
    niche = "default"
    if "--niche" in sys.argv:
        niche_idx = sys.argv.index("--niche")
        if niche_idx + 1 < len(sys.argv):
            niche = sys.argv[niche_idx + 1]

    # Default to validate if no action specified
    if not any([do_clean, do_validate, do_improve]):
        do_validate = True

    try:
        with open(script_file, 'r', encoding='utf-8') as f:
            script_text = f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {script_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    validator = ScriptValidator()

    if do_clean:
        cleaned = validator.clean_script(script_text)
        if output_json:
            print(json.dumps({"cleaned_script": cleaned}))
        else:
            print("=" * 60)
            print("CLEANED SCRIPT:")
            print("=" * 60)
            print(cleaned)

    if do_validate:
        result = validator.validate_script(script_text, niche=niche, is_short=is_short)
        if output_json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("\n" + "=" * 60)
            print(result.summary())
            print("=" * 60)

    if do_improve:
        improved = validator.improve_script(script_text, niche=niche)
        if output_json:
            print(json.dumps({"improved_script": improved}))
        else:
            print("\n" + "=" * 60)
            print("IMPROVED SCRIPT:")
            print("=" * 60)
            print(improved)
