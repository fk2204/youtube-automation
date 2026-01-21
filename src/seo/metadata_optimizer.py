"""
Title/Description Optimizer

Implement IMPACT formula for titles, front-load keywords, auto-generate
timestamps/chapters, optimize description structure, and use power words.

Features:
- IMPACT title formula (Immediate, Massive, Perceived, Action, Clear, Transform)
- Keyword front-loading (first 40 chars)
- Auto-generated timestamps and chapters
- 200-300 word optimized descriptions
- Power words database for engagement

Usage:
    optimizer = MetadataOptimizer()

    # Optimize title
    title = optimizer.optimize_title("How to Make Money", keywords=["passive income", "2026"])

    # Generate description
    description = optimizer.generate_description(
        topic="passive income",
        keywords=["investing", "make money online"],
        video_duration=600
    )

    # Add chapters
    description_with_chapters = optimizer.add_chapters(description, timestamps)
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from loguru import logger


@dataclass
class OptimizedMetadata:
    """Optimized video metadata."""
    title: str
    description: str
    tags: List[str]
    title_score: float  # 0-100
    keyword_density: float
    chapters: List[Dict]


class MetadataOptimizer:
    """
    Optimize YouTube titles and descriptions for SEO and CTR.
    """

    # Power words that increase CTR
    POWER_WORDS = {
        "curiosity": ["secret", "hidden", "revealed", "truth", "exposed", "untold", "leaked"],
        "urgency": ["now", "today", "2026", "new", "breaking", "urgent", "limited"],
        "value": ["free", "proven", "guaranteed", "easy", "simple", "fast", "ultimate"],
        "emotion": ["shocking", "insane", "unbelievable", "amazing", "incredible", "mind-blowing"],
        "authority": ["expert", "professional", "advanced", "complete", "definitive"],
        "numbers": ["7", "10", "5", "3", "100%", "$10K", "1M+"],
    }

    # Negative words to avoid (clickbait but hurt retention)
    AVOID_WORDS = ["scam", "fake", "lie", "hate", "worst", "terrible"]

    # IMPACT formula components
    IMPACT_COMPONENTS = {
        "I": "Immediate",  # Create urgency
        "M": "Massive",    # Show big benefit
        "P": "Perceived",  # Specific outcome
        "A": "Action",     # Use action verbs
        "C": "Clear",      # Easy to understand
        "T": "Transform"   # Promise transformation
    }

    def __init__(self):
        """Initialize metadata optimizer."""
        logger.info("[MetadataOptimizer] Initialized")

    def optimize_title(
        self,
        base_title: str,
        keywords: List[str],
        max_length: int = 70,
        front_load_keywords: bool = True
    ) -> str:
        """
        Optimize title using IMPACT formula and keyword front-loading.

        Args:
            base_title: Original title
            keywords: Target keywords
            max_length: Max title length
            front_load_keywords: Put main keyword in first 40 chars

        Returns:
            Optimized title
        """
        # Start with base title
        title = base_title.strip()

        # Front-load primary keyword (first 40 chars for mobile visibility)
        if front_load_keywords and keywords:
            primary_keyword = keywords[0]
            if primary_keyword.lower() not in title[:40].lower():
                title = f"{primary_keyword.title()}: {title}"

        # Add power words if not present
        has_power_word = any(
            word.lower() in title.lower()
            for category in self.POWER_WORDS.values()
            for word in category
        )

        if not has_power_word:
            # Add a subtle power word
            power_word = "proven" if "finance" in title.lower() else "ultimate"
            title = f"{title} ({power_word.title()})"

        # Add current year for freshness
        if "2026" not in title and "2025" not in title:
            title = f"{title} 2026"

        # Trim if too long
        if len(title) > max_length:
            title = title[:max_length - 3] + "..."

        score = self._score_title(title, keywords)
        logger.info(f"[MetadataOptimizer] Title score: {score:.1f}/100")

        return title

    def _score_title(self, title: str, keywords: List[str]) -> float:
        """Score title based on best practices (0-100)."""
        score = 0.0
        title_lower = title.lower()

        # Length score (50-70 chars is optimal)
        length = len(title)
        if 50 <= length <= 70:
            score += 20
        elif 40 <= length <= 80:
            score += 10

        # Keyword presence (30 points)
        keywords_found = sum(1 for kw in keywords if kw.lower() in title_lower)
        score += min(30, keywords_found * 15)

        # Power word presence (20 points)
        power_words_found = sum(
            1 for category in self.POWER_WORDS.values()
            for word in category if word in title_lower
        )
        score += min(20, power_words_found * 10)

        # Number presence (15 points)
        if re.search(r'\d+', title):
            score += 15

        # Freshness (year) (10 points)
        if "2026" in title or "2025" in title:
            score += 10

        # Avoid clickbait (5 points)
        if not any(word in title_lower for word in self.AVOID_WORDS):
            score += 5

        return min(100.0, score)

    def score_title(self, title: str, keywords: List[str] = None) -> float:
        """
        Public method to score a title.

        Args:
            title: Title to score
            keywords: Optional keywords for scoring

        Returns:
            Score from 0-100
        """
        return self._score_title(title, keywords or [])

    def generate_title_variants(
        self,
        topic: str,
        keywords: List[str],
        count: int = 3
    ) -> List[str]:
        """
        Generate multiple title variants for A/B testing.

        Args:
            topic: Base topic/title
            keywords: Target keywords
            count: Number of variants to generate

        Returns:
            List of title variants
        """
        variants = []

        # Variant 1: Original optimized
        v1 = self.optimize_title(topic, keywords)
        variants.append(v1)

        # Variant 2: Question format
        if count >= 2:
            question_starters = ["How to", "Why", "What Makes", "Is"]
            for starter in question_starters:
                if not topic.lower().startswith(starter.lower()):
                    v2 = f"{starter} {topic}?"
                    v2 = self.optimize_title(v2, keywords)
                    if v2 not in variants:
                        variants.append(v2)
                        break

        # Variant 3: Number-based format
        if count >= 3 and not re.search(r'\d+', topic):
            v3 = f"7 {topic} Tips That Actually Work"
            v3 = self.optimize_title(v3, keywords)
            if v3 not in variants:
                variants.append(v3)

        # Variant 4: Contrarian format
        if count >= 4:
            v4 = f"Why Everything You Know About {topic} Is Wrong"
            v4 = self.optimize_title(v4, keywords)
            if v4 not in variants:
                variants.append(v4)

        return variants[:count]

    def generate_description(
        self,
        topic: str,
        keywords: List[str],
        video_duration: int = 600,
        target_words: int = 250,
        include_chapters: bool = True,
        chapters: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate optimized description (200-300 words).

        Args:
            topic: Video topic
            keywords: Target keywords
            video_duration: Video length in seconds
            target_words: Target word count
            include_chapters: Include chapter timestamps
            chapters: Chapter data if available

        Returns:
            Optimized description
        """
        description_parts = []

        # 1. Hook (first 2-3 lines visible without "show more")
        hook = self._generate_description_hook(topic, keywords)
        description_parts.append(hook)

        # 2. Value proposition
        value = f"\n\nIn this video about {keywords[0]}, you'll discover:"
        bullets = self._generate_value_bullets(topic, keywords)
        description_parts.append(value)
        description_parts.append(bullets)

        # 3. Chapters (if provided)
        if include_chapters and chapters:
            chapter_text = self._format_chapters(chapters)
            description_parts.append("\n\n" + chapter_text)

        # 4. Keywords naturally integrated
        keyword_section = self._generate_keyword_section(keywords)
        description_parts.append("\n\n" + keyword_section)

        # 5. Call to action
        cta = "\n\n🔔 Subscribe for more content on " + ", ".join(keywords[:3])
        cta += "\n👍 Like if you found this helpful"
        cta += "\n💬 Comment your questions below"
        description_parts.append(cta)

        # 6. Hashtags (max 3 for YouTube)
        hashtags = "\n\n" + " ".join([f"#{kw.replace(' ', '')}" for kw in keywords[:3]])
        description_parts.append(hashtags)

        description = "".join(description_parts)

        # Calculate keyword density
        words = description.lower().split()
        keyword_mentions = sum(1 for word in words for kw in keywords if kw.lower() in word)
        density = (keyword_mentions / len(words)) * 100 if words else 0

        logger.info(
            f"[MetadataOptimizer] Description generated: {len(words)} words, "
            f"{density:.1f}% keyword density"
        )

        return description

    def _generate_description_hook(self, topic: str, keywords: List[str]) -> str:
        """Generate engaging description hook."""
        templates = [
            f"Want to master {keywords[0]}? You're in the right place.",
            f"Everything you need to know about {keywords[0]} in one video.",
            f"This {topic} guide will transform how you think about {keywords[0]}.",
            f"Discover the secrets of {keywords[0]} that actually work.",
        ]
        return templates[hash(topic) % len(templates)]

    def _generate_value_bullets(self, topic: str, keywords: List[str]) -> str:
        """Generate value bullet points."""
        bullets = [
            f"✓ How to get started with {keywords[0]}",
            f"✓ Common mistakes to avoid",
            f"✓ Proven strategies for success",
            f"✓ Real-world examples and case studies",
        ]
        return "\n".join(bullets)

    def _format_chapters(self, chapters: List[Dict]) -> str:
        """Format chapters as timestamps."""
        if not chapters:
            return ""

        chapter_text = "⏱️ CHAPTERS:\n"
        for chapter in chapters:
            timestamp = self._format_timestamp(chapter.get("start", 0))
            title = chapter.get("title", "Chapter")
            chapter_text += f"{timestamp} - {title}\n"

        return chapter_text

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    def _generate_keyword_section(self, keywords: List[str]) -> str:
        """Generate keyword-rich section."""
        return (
            f"Topics covered: {', '.join(keywords)}. "
            f"Learn more about {keywords[0]} and related concepts. "
            f"This comprehensive guide to {keywords[0]} is perfect for beginners and experts alike."
        )

    def auto_generate_chapters(
        self,
        script: str,
        video_duration: int,
        chapter_count: int = 5
    ) -> List[Dict]:
        """
        Auto-generate chapters from script.

        Args:
            script: Video script
            video_duration: Total duration in seconds
            chapter_count: Target number of chapters

        Returns:
            List of chapter dictionaries
        """
        # Split script into sections
        paragraphs = [p.strip() for p in script.split("\n\n") if p.strip()]

        if len(paragraphs) < chapter_count:
            chapter_count = len(paragraphs)

        # Distribute chapters evenly
        chapters = []
        section_size = len(paragraphs) // chapter_count

        for i in range(chapter_count):
            start_idx = i * section_size
            section = paragraphs[start_idx] if start_idx < len(paragraphs) else ""

            # Extract first sentence as chapter title
            sentences = re.split(r'[.!?]', section)
            title = sentences[0].strip()[:60] if sentences else f"Chapter {i+1}"

            # Calculate timestamp
            timestamp = (video_duration / chapter_count) * i

            chapters.append({
                "start": timestamp,
                "title": title,
                "index": i
            })

        # Always add intro and outro
        if chapters:
            chapters[0]["title"] = "Introduction"
            chapters.append({
                "start": video_duration * 0.95,
                "title": "Wrap Up & Next Steps",
                "index": len(chapters)
            })

        logger.info(f"[MetadataOptimizer] Generated {len(chapters)} chapters")
        return chapters

    def optimize_tags(
        self,
        keywords: List[str],
        topic: str,
        max_tags: int = 15
    ) -> List[str]:
        """
        Generate optimized tag list.

        Args:
            keywords: Target keywords
            topic: Video topic
            max_tags: Maximum number of tags

        Returns:
            List of optimized tags
        """
        tags = []

        # Add exact keywords
        tags.extend(keywords)

        # Add topic variations
        tags.append(f"{topic} tutorial")
        tags.append(f"{topic} guide")
        tags.append(f"how to {topic}")

        # Add niche-specific tags
        tags.append(f"{topic} 2026")
        tags.append(f"{topic} for beginners")

        # Remove duplicates and limit
        tags = list(dict.fromkeys(tags))  # Preserve order
        tags = tags[:max_tags]

        logger.info(f"[MetadataOptimizer] Generated {len(tags)} tags")
        return tags

    def create_complete_metadata(
        self,
        topic: str,
        keywords: List[str],
        script: str,
        video_duration: int
    ) -> OptimizedMetadata:
        """
        Generate complete optimized metadata package.

        Args:
            topic: Video topic
            keywords: Target keywords
            script: Video script
            video_duration: Duration in seconds

        Returns:
            OptimizedMetadata object
        """
        # Generate chapters
        chapters = self.auto_generate_chapters(script, video_duration)

        # Optimize title
        title = self.optimize_title(f"How to {topic}", keywords)
        title_score = self._score_title(title, keywords)

        # Generate description
        description = self.generate_description(
            topic=topic,
            keywords=keywords,
            video_duration=video_duration,
            chapters=chapters
        )

        # Calculate keyword density
        words = description.lower().split()
        keyword_mentions = sum(1 for word in words for kw in keywords if kw.lower() in word)
        density = (keyword_mentions / len(words)) * 100 if words else 0

        # Generate tags
        tags = self.optimize_tags(keywords, topic)

        metadata = OptimizedMetadata(
            title=title,
            description=description,
            tags=tags,
            title_score=title_score,
            keyword_density=density,
            chapters=chapters
        )

        logger.success(
            f"[MetadataOptimizer] Complete metadata generated "
            f"(Title score: {title_score:.1f}/100, Keyword density: {density:.1f}%)"
        )

        return metadata


# CLI
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("""
Metadata Optimizer - Title & Description SEO

Commands:
    title <base_title> <keyword1> [keyword2...]
        Optimize a title

    description <topic> <keyword1> [keyword2...] [--duration <seconds>]
        Generate optimized description

    complete <topic> <keyword1> [keyword2...] [--script <file>] [--duration <seconds>]
        Generate complete metadata package

Examples:
    python -m src.seo.metadata_optimizer title "Make Money" "passive income" "2026"
    python -m src.seo.metadata_optimizer description "investing" "stocks" "beginners" --duration 600
    python -m src.seo.metadata_optimizer complete "passive income" "make money" --script script.txt --duration 600
        """)
    else:
        optimizer = MetadataOptimizer()
        cmd = sys.argv[1]

        if cmd == "title" and len(sys.argv) >= 4:
            base_title = sys.argv[2]
            keywords = sys.argv[3:]
            title = optimizer.optimize_title(base_title, keywords)
            print(f"\nOptimized Title:\n{title}\n")

        elif cmd == "description" and len(sys.argv) >= 4:
            topic = sys.argv[2]
            keywords = []
            duration = 600

            for i, arg in enumerate(sys.argv[3:]):
                if arg == "--duration":
                    duration = int(sys.argv[3 + i + 1])
                    break
                keywords.append(arg)

            description = optimizer.generate_description(topic, keywords, duration)
            print(f"\nOptimized Description:\n{description}\n")

        elif cmd == "complete" and len(sys.argv) >= 4:
            topic = sys.argv[2]
            keywords = []
            script_file = None
            duration = 600

            i = 3
            while i < len(sys.argv):
                if sys.argv[i] == "--script":
                    script_file = sys.argv[i + 1]
                    i += 2
                elif sys.argv[i] == "--duration":
                    duration = int(sys.argv[i + 1])
                    i += 2
                else:
                    keywords.append(sys.argv[i])
                    i += 1

            script = ""
            if script_file:
                with open(script_file, "r", encoding="utf-8") as f:
                    script = f.read()
            else:
                script = f"This is a video about {topic}. " * 50  # Mock script

            metadata = optimizer.create_complete_metadata(topic, keywords, script, duration)
            print("\nComplete Metadata Package:\n")
            print(f"Title: {metadata.title}")
            print(f"Title Score: {metadata.title_score:.1f}/100")
            print(f"\nDescription:\n{metadata.description}")
            print(f"\nTags: {', '.join(metadata.tags)}")
            print(f"\nChapters: {len(metadata.chapters)}")
            for chapter in metadata.chapters:
                print(f"  {optimizer._format_timestamp(chapter['start'])} - {chapter['title']}")
