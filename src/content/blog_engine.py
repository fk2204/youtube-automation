"""
Blog content creation engine.

Generates SEO-optimized blog articles from topics or video scripts.
Supports output for Medium, LinkedIn articles, Quora answers, and
Pinterest pin descriptions.

Uses the existing ScriptWriter AI providers for generation and
KeywordIntelligence for SEO optimization.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class BlogArticle:
    """Generated blog article."""
    title: str
    content: str  # Markdown format
    meta_description: str = ""
    keywords: List[str] = field(default_factory=list)
    headings: List[str] = field(default_factory=list)
    word_count: int = 0
    reading_time_minutes: int = 0
    niche: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Platform-specific variants
    medium_content: str = ""
    linkedin_content: str = ""
    quora_answer: str = ""
    twitter_thread: List[str] = field(default_factory=list)
    pinterest_descriptions: List[str] = field(default_factory=list)


class BlogEngine:
    """Creates blog articles and platform-specific text variants.

    Usage:
        engine = BlogEngine()
        article = await engine.create_article(
            topic="5 Passive Income Strategies",
            niche="finance",
        )
        print(article.content)        # Full markdown article
        print(article.twitter_thread)  # Thread-ready tweets
    """

    def __init__(self, provider: str = "groq"):
        self._provider = provider
        self._writer = None

    def _get_writer(self):
        if self._writer is None:
            try:
                from src.content.script_writer import ScriptWriter
                self._writer = ScriptWriter(provider=self._provider)
            except ImportError:
                logger.warning("ScriptWriter not available")
        return self._writer

    async def create_article(
        self,
        topic: str,
        niche: str = "general",
        channel: str = "",
        style: str = "educational",
        word_count_target: int = 1500,
    ) -> BlogArticle:
        """Generate a full blog article from a topic.

        Args:
            topic: The article topic
            niche: Content niche (finance, psychology, etc.)
            channel: Channel ID for branding
            style: Writing style (educational, casual, professional)
            word_count_target: Target word count

        Returns:
            BlogArticle with all platform variants.
        """
        writer = self._get_writer()

        # Generate main article via AI
        prompt = self._build_blog_prompt(topic, niche, style, word_count_target)
        raw_content = self._generate_with_ai(prompt)

        # Parse and structure
        article = self._parse_article(raw_content, topic, niche)

        # Generate platform variants
        article.medium_content = self._adapt_for_medium(article)
        article.linkedin_content = self._adapt_for_linkedin(article)
        article.quora_answer = self._adapt_for_quora(article)
        article.twitter_thread = self._create_twitter_thread(article)
        article.pinterest_descriptions = self._create_pin_descriptions(article)

        # SEO optimization
        await self._optimize_seo(article)

        # Save to file
        output_path = Path("output/blogs")
        output_path.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", topic.lower())[:50]
        file_path = output_path / f"{slug}_{datetime.utcnow().strftime('%Y%m%d')}.md"
        file_path.write_text(article.content, encoding="utf-8")
        logger.info(f"Blog saved: {file_path}")

        return article

    async def from_script(self, script, niche: str = "general") -> BlogArticle:
        """Convert a VideoScript to a blog article.

        Takes the existing script data and reformats it as a blog post
        with proper headings, links, and SEO structure.
        """
        title = script.title if hasattr(script, "title") else str(script)
        sections = script.sections if hasattr(script, "sections") else []

        # Build markdown from script sections
        content_parts = [f"# {title}\n"]
        for section in sections:
            heading = section.heading if hasattr(section, "heading") else ""
            body = section.content if hasattr(section, "content") else str(section)
            if heading:
                content_parts.append(f"\n## {heading}\n")
            content_parts.append(f"{body}\n")

        raw_content = "\n".join(content_parts)

        article = self._parse_article(raw_content, title, niche)
        article.medium_content = self._adapt_for_medium(article)
        article.linkedin_content = self._adapt_for_linkedin(article)
        article.quora_answer = self._adapt_for_quora(article)
        article.twitter_thread = self._create_twitter_thread(article)
        article.pinterest_descriptions = self._create_pin_descriptions(article)

        return article

    def _build_blog_prompt(self, topic: str, niche: str, style: str, word_count: int) -> str:
        return f"""Write a comprehensive blog article about: {topic}

Niche: {niche}
Style: {style}
Target length: {word_count} words

Requirements:
- Use markdown formatting with clear headings (##)
- Include an engaging introduction with a hook
- Use bullet points and numbered lists where appropriate
- Include actionable takeaways
- End with a strong conclusion and call to action
- Optimize for SEO (use the topic keywords naturally)
- Write in an authoritative but accessible tone

Format the output as a complete markdown article."""

    def _generate_with_ai(self, prompt: str) -> str:
        writer = self._get_writer()
        if not writer:
            return f"# Article\n\n{prompt}\n\n*Content generation requires AI provider.*"

        try:
            from src.content.script_writer import get_provider
            provider = get_provider(self._provider)
            return provider.generate(prompt)
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            return f"# Article\n\n*Generation failed: {e}*"

    def _parse_article(self, raw_content: str, topic: str, niche: str) -> BlogArticle:
        """Parse raw AI output into structured BlogArticle."""
        # Extract title from first # heading or use topic
        title = topic
        lines = raw_content.split("\n")
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break

        # Extract headings
        headings = [line[3:].strip() for line in lines if line.startswith("## ")]

        # Word count
        words = len(raw_content.split())
        reading_time = max(1, words // 200)

        # Meta description (first paragraph that's not a heading)
        meta = ""
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and len(stripped) > 50:
                meta = stripped[:160]
                break

        return BlogArticle(
            title=title,
            content=raw_content,
            meta_description=meta,
            headings=headings,
            word_count=words,
            reading_time_minutes=reading_time,
            niche=niche,
        )

    def _adapt_for_medium(self, article: BlogArticle) -> str:
        """Adapt for Medium (supports full markdown, add tags)."""
        # Medium supports markdown natively
        footer = f"\n\n---\n\n*Originally published on our blog. Follow for more {article.niche} insights.*"
        return article.content + footer

    def _adapt_for_linkedin(self, article: BlogArticle) -> str:
        """Adapt for LinkedIn articles (2000 char limit for posts, longer for articles)."""
        # LinkedIn articles: Take first 1500 words + CTA
        lines = article.content.split("\n")
        linkedin_lines = []
        word_count = 0
        for line in lines:
            words = len(line.split())
            if word_count + words > 1500:
                break
            linkedin_lines.append(line)
            word_count += words

        content = "\n".join(linkedin_lines)
        content += f"\n\n---\n\nWhat are your thoughts on this? Comment below.\n\n#{article.niche} #insights #learning"
        return content

    def _adapt_for_quora(self, article: BlogArticle) -> str:
        """Adapt as a Quora answer (conversational, direct, ~500 words)."""
        # Take key points and present as answer
        sections = article.content.split("\n## ")
        if len(sections) > 1:
            # Use first 2-3 sections as answer
            answer_parts = sections[:3]
            answer = "\n\n".join(answer_parts)
        else:
            answer = article.content

        # Trim to ~500 words
        words = answer.split()
        if len(words) > 500:
            answer = " ".join(words[:500]) + "..."

        return answer

    def _create_twitter_thread(self, article: BlogArticle) -> List[str]:
        """Create a Twitter/X thread from article key points."""
        thread = []

        # Tweet 1: Hook
        thread.append(f"{article.title}\n\nA thread 🧵")

        # Extract key points from headings/sections
        sections = article.content.split("\n## ")
        for i, section in enumerate(sections[1:6], 2):  # Max 5 content tweets
            # Get first line of section as point
            lines = section.strip().split("\n")
            heading = lines[0].strip() if lines else ""
            body = " ".join(lines[1:3]).strip() if len(lines) > 1 else ""

            tweet = f"{i}/ {heading}"
            if body:
                tweet += f"\n\n{body[:200]}"

            # Respect 280 char limit
            if len(tweet) > 280:
                tweet = tweet[:277] + "..."

            thread.append(tweet)

        # Final tweet: CTA
        thread.append(f"If you found this useful:\n\n• Follow for more {article.niche} insights\n• Repost to share\n• Comment your thoughts")

        return thread

    def _create_pin_descriptions(self, article: BlogArticle) -> List[str]:
        """Create Pinterest pin descriptions (SEO-rich, 500 chars max)."""
        descriptions = []

        # Pin 1: Main article pin
        desc = f"{article.title} | Learn the key insights about {article.niche}. "
        if article.headings:
            desc += "Topics covered: " + ", ".join(article.headings[:3]) + ". "
        desc += f"#{article.niche} #tips #guide"
        descriptions.append(desc[:500])

        # Pin 2-3: Section-specific pins
        for heading in article.headings[:2]:
            pin = f"{heading} - Expert {article.niche} insights. Read our full guide for actionable tips. #{article.niche} #{heading.lower().replace(' ', '')}"
            descriptions.append(pin[:500])

        return descriptions

    async def _optimize_seo(self, article: BlogArticle) -> None:
        """Apply SEO optimization using KeywordIntelligence."""
        try:
            from src.seo.keyword_intelligence import KeywordIntelligence
            ki = KeywordIntelligence()
            result = ki.full_analysis(article.title, niche=article.niche)
            if hasattr(result, "keywords"):
                article.keywords = result.keywords[:10]
        except (ImportError, Exception) as e:
            logger.debug(f"SEO optimization skipped: {e}")
            article.keywords = [article.niche, article.title.split()[0].lower()]
