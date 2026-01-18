"""
Stock Footage Module - Pexels API Integration

Fetches relevant stock videos and images for video production.
FREE: 20,000 requests/month, 200/hour

Get API key: https://www.pexels.com/api/

Usage:
    from src.content.stock_footage import StockFootage

    stock = StockFootage()
    videos = stock.search_videos("passive income", count=5)
    stock.download_video(videos[0], "output/clip1.mp4")
"""

import os
import json
import requests
import random
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class StockVideo:
    """Stock video metadata."""
    id: int
    url: str
    duration: int  # seconds
    width: int
    height: int
    preview_url: str
    download_url: str  # Best quality
    photographer: str
    tags: List[str]


@dataclass
class StockImage:
    """Stock image metadata."""
    id: int
    url: str
    width: int
    height: int
    download_url: str
    photographer: str
    alt: str


class StockFootage:
    """
    Pexels API client for stock videos and images.

    Free tier: 200 requests/hour, 20,000/month
    """

    BASE_URL = "https://api.pexels.com"

    # Topic-specific keyword mappings for smarter stock footage matching
    NICHE_KEYWORDS = {
        "psychology": {
            "narcissism": ["manipulative person", "toxic relationship", "gaslighting", "emotional abuse", "fake smile", "controlling behavior", "sad person", "argument couple"],
            "anxiety": ["stressed person", "worried face", "panic", "breathing exercise", "calm meditation"],
            "motivation": ["success celebration", "goal achievement", "determined person", "workout motivation"],
            "depression": ["sad person", "loneliness", "dark room", "isolation", "empty room"],
            "relationships": ["couple talking", "family conflict", "communication", "listening", "emotional support"],
            "trauma": ["broken glass", "dark memories", "healing process", "therapy session", "recovery"],
            "self_esteem": ["mirror reflection", "confident person", "self doubt", "body image", "positive affirmation"],
            "default": ["brain", "thinking person", "psychology session", "mental health"]
        },
        "storytelling": {
            "disappearance": ["missing person poster", "police investigation", "abandoned place", "old photographs", "detective", "search party"],
            "crime": ["crime scene tape", "police lights", "courtroom", "evidence", "investigation"],
            "mystery": ["dark forest", "fog", "abandoned building", "eerie location", "night scene"],
            "horror": ["haunted house", "dark corridor", "creepy atmosphere", "night forest", "shadow figure"],
            "true_crime": ["police car", "handcuffs", "jail cell", "trial courtroom", "news broadcast"],
            "historical": ["old film footage", "vintage photos", "historical documents", "archive footage", "sepia tone"],
            "default": ["documentary footage", "historical photos", "newspaper", "timeline"]
        },
        "finance": {
            "investing": ["stock market", "trading screen", "money growth", "investment chart"],
            "budgeting": ["calculator money", "savings jar", "piggy bank", "budget planning"],
            "passive_income": ["laptop money", "rental property", "dividend stocks", "online business"],
            "crypto": ["bitcoin", "cryptocurrency", "blockchain", "digital currency", "crypto trading"],
            "debt": ["credit card", "bills pile", "financial stress", "debt payoff", "money worry"],
            "wealth": ["luxury lifestyle", "expensive car", "mansion", "yacht", "wealthy person"],
            "retirement": ["elderly couple", "retirement planning", "pension", "golden years", "senior lifestyle"],
            "default": ["money", "business meeting", "office work", "financial planning"]
        },
        "technology": {
            "ai": ["artificial intelligence", "robot", "machine learning", "neural network", "futuristic"],
            "coding": ["programming", "developer", "code screen", "software development", "computer coding"],
            "cybersecurity": ["hacker", "data security", "encryption", "cyber attack", "firewall"],
            "gadgets": ["smartphone", "laptop", "tech devices", "electronics", "innovation"],
            "default": ["technology", "computer", "digital", "circuit board", "data center"]
        },
        "motivation": {
            "fitness": ["gym workout", "running", "exercise", "training", "athlete"],
            "success": ["celebration", "trophy", "achievement", "winning", "podium"],
            "entrepreneurship": ["startup", "business owner", "hustle", "working late", "entrepreneur"],
            "mindset": ["meditation", "focus", "determination", "mindfulness", "positive thinking"],
            "default": ["success celebration", "goal achievement", "determined person", "sunrise motivation"]
        },
        "health": {
            "fitness": ["workout", "gym", "exercise", "running", "yoga"],
            "nutrition": ["healthy food", "vegetables", "cooking", "meal prep", "diet"],
            "mental_health": ["meditation", "therapy", "relaxation", "stress relief", "self care"],
            "medical": ["doctor", "hospital", "medical equipment", "healthcare", "treatment"],
            "default": ["healthy lifestyle", "wellness", "medical", "fitness"]
        }
    }

    # Keywords/phrases that map to sub-topics for detection
    SUBTOPIC_TRIGGERS = {
        "psychology": {
            "narcissism": ["narciss", "manipulat", "toxic", "gaslight", "emotional abuse", "covert", "supply", "flying monkey"],
            "anxiety": ["anxiet", "anxious", "worry", "panic", "nervous", "stress", "overwhelm"],
            "motivation": ["motivat", "inspire", "achiev", "success", "goal", "discipline", "habit"],
            "depression": ["depress", "sad", "lonely", "hopeless", "empty", "meaningless"],
            "relationships": ["relationship", "partner", "marriage", "dating", "attachment", "love bomb"],
            "trauma": ["trauma", "ptsd", "abuse", "survivor", "healing", "recover"],
            "self_esteem": ["self-esteem", "self esteem", "confidence", "self-worth", "insecur"]
        },
        "storytelling": {
            "disappearance": ["disappear", "missing", "vanish", "gone", "lost person", "search"],
            "crime": ["murder", "killer", "crime", "criminal", "assault", "robbery"],
            "mystery": ["mystery", "unsolved", "strange", "unexplained", "bizarre", "weird"],
            "horror": ["horror", "scary", "terrif", "haunt", "creepy", "paranormal"],
            "true_crime": ["true crime", "serial", "case", "investigation", "suspect", "trial"],
            "historical": ["history", "historical", "ancient", "century", "war", "past"]
        },
        "finance": {
            "investing": ["invest", "stock", "market", "portfolio", "dividend", "etf", "index fund"],
            "budgeting": ["budget", "saving", "frugal", "expense", "spending", "money management"],
            "passive_income": ["passive", "income stream", "side hustle", "affiliate", "royalt"],
            "crypto": ["crypto", "bitcoin", "ethereum", "blockchain", "nft", "defi"],
            "debt": ["debt", "credit card", "loan", "payoff", "interest", "owe"],
            "wealth": ["wealth", "rich", "millionaire", "billionaire", "luxury", "affluent"],
            "retirement": ["retire", "401k", "pension", "ira", "social security", "golden years"]
        },
        "technology": {
            "ai": ["ai", "artificial intelligence", "machine learning", "chatgpt", "neural", "deep learning"],
            "coding": ["cod", "program", "develop", "software", "javascript", "python", "web dev"],
            "cybersecurity": ["security", "hack", "cyber", "breach", "malware", "phishing"],
            "gadgets": ["phone", "laptop", "device", "gadget", "smart", "wearable"]
        },
        "motivation": {
            "fitness": ["fitness", "workout", "gym", "exercise", "body", "muscle", "weight"],
            "success": ["success", "win", "achieve", "goal", "champion", "victory"],
            "entrepreneurship": ["entrepreneur", "business", "startup", "founder", "hustle", "grind"],
            "mindset": ["mindset", "mental", "focus", "meditat", "mindful", "discipline"]
        },
        "health": {
            "fitness": ["fitness", "workout", "exercise", "gym", "training"],
            "nutrition": ["nutrition", "diet", "food", "eat", "meal", "vitamin"],
            "mental_health": ["mental health", "therapy", "anxiety", "depression", "stress"],
            "medical": ["doctor", "hospital", "treatment", "medicine", "diagnosis"]
        }
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Pexels client.

        Args:
            api_key: Pexels API key (or set PEXELS_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("PEXELS_API_KEY")

        if not self.api_key:
            logger.warning(
                "Pexels API key not found. Get one free at https://www.pexels.com/api/\n"
                "Set PEXELS_API_KEY in your .env file"
            )
        else:
            logger.info("StockFootage initialized with Pexels API")

        self.cache_dir = Path("cache/stock")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Track API usage
        self.requests_made = 0

    def _headers(self) -> Dict[str, str]:
        """Get API headers."""
        return {"Authorization": self.api_key}

    def _detect_subtopic(self, topic: str, niche: str) -> Optional[str]:
        """
        Detect the sub-topic from the topic text.

        Args:
            topic: The topic text to analyze
            niche: The content niche

        Returns:
            Detected sub-topic name or None
        """
        topic_lower = topic.lower()
        triggers = self.SUBTOPIC_TRIGGERS.get(niche, {})

        # Score each sub-topic based on trigger matches
        scores = {}
        for subtopic, trigger_words in triggers.items():
            score = 0
            for trigger in trigger_words:
                if trigger.lower() in topic_lower:
                    # Longer matches are more specific, so weight them higher
                    score += len(trigger)
            if score > 0:
                scores[subtopic] = score

        # Return the highest scoring sub-topic
        if scores:
            best_subtopic = max(scores, key=scores.get)
            logger.debug(f"Detected subtopic '{best_subtopic}' for topic '{topic}' (score: {scores[best_subtopic]})")
            return best_subtopic

        return None

    def get_smart_keywords(self, topic: str, niche: str) -> List[str]:
        """
        Get smart keywords based on topic analysis and niche.

        Analyzes the topic text to detect sub-topics and returns
        relevant keywords for stock footage searching.

        Args:
            topic: The topic text to analyze
            niche: The content niche (e.g., "psychology", "finance", "storytelling")

        Returns:
            List of relevant keywords for stock footage search
        """
        keywords = []
        niche_data = self.NICHE_KEYWORDS.get(niche, {})

        # Try to detect a specific sub-topic
        detected_subtopic = self._detect_subtopic(topic, niche)

        if detected_subtopic and detected_subtopic in niche_data:
            # Use sub-topic specific keywords
            subtopic_keywords = niche_data[detected_subtopic]
            keywords.extend(subtopic_keywords)
            logger.info(f"Using '{detected_subtopic}' keywords for niche '{niche}': {subtopic_keywords[:3]}...")
        else:
            # Fall back to default keywords for the niche
            default_keywords = niche_data.get("default", [])
            keywords.extend(default_keywords)
            logger.info(f"Using default keywords for niche '{niche}': {default_keywords[:3]}...")

        # Also add some words extracted from the topic itself
        topic_words = [w for w in topic.lower().split() if len(w) > 3]
        keywords.extend(topic_words[:3])

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)

        return unique_keywords

    def search_videos(
        self,
        query: str,
        count: int = 10,
        min_duration: int = 5,
        max_duration: int = 60,
        orientation: str = "landscape"
    ) -> List[StockVideo]:
        """
        Search for stock videos.

        Args:
            query: Search query
            count: Number of videos to fetch
            min_duration: Minimum video duration in seconds
            max_duration: Maximum video duration
            orientation: landscape, portrait, or square

        Returns:
            List of StockVideo objects
        """
        if not self.api_key:
            logger.warning("No API key - returning empty results")
            return []

        try:
            url = f"{self.BASE_URL}/videos/search"
            params = {
                "query": query,
                "per_page": min(count * 2, 80),  # Fetch extra for filtering
                "orientation": orientation,
                "size": "medium"  # medium quality for faster downloads
            }

            response = requests.get(url, headers=self._headers(), params=params, timeout=30)
            self.requests_made += 1

            if response.status_code != 200:
                logger.error(f"Pexels API error: {response.status_code} - {response.text[:200]}")
                return []

            data = response.json()
            videos = []

            for video in data.get("videos", []):
                duration = video.get("duration", 0)

                # Filter by duration
                if duration < min_duration or duration > max_duration:
                    continue

                # Get best video file (HD preferred)
                video_files = video.get("video_files", [])
                best_file = None

                # Prefer HD (1280x720) or higher
                for vf in sorted(video_files, key=lambda x: x.get("width", 0), reverse=True):
                    if vf.get("width", 0) >= 1280:
                        best_file = vf
                        break

                if not best_file and video_files:
                    best_file = video_files[0]

                if not best_file:
                    continue

                # Get preview image
                preview_pics = video.get("video_pictures", [])
                preview_url = preview_pics[0]["picture"] if preview_pics else ""

                videos.append(StockVideo(
                    id=video["id"],
                    url=video["url"],
                    duration=duration,
                    width=best_file.get("width", 1920),
                    height=best_file.get("height", 1080),
                    preview_url=preview_url,
                    download_url=best_file["link"],
                    photographer=video.get("user", {}).get("name", "Unknown"),
                    tags=query.split()
                ))

                if len(videos) >= count:
                    break

            logger.info(f"Found {len(videos)} videos for: {query}")
            return videos

        except Exception as e:
            logger.error(f"Video search failed: {e}")
            return []

    def search_images(
        self,
        query: str,
        count: int = 10,
        orientation: str = "landscape"
    ) -> List[StockImage]:
        """
        Search for stock images.

        Args:
            query: Search query
            count: Number of images to fetch
            orientation: landscape, portrait, or square

        Returns:
            List of StockImage objects
        """
        if not self.api_key:
            return []

        try:
            url = f"{self.BASE_URL}/v1/search"
            params = {
                "query": query,
                "per_page": count,
                "orientation": orientation
            }

            response = requests.get(url, headers=self._headers(), params=params, timeout=30)
            self.requests_made += 1

            if response.status_code != 200:
                logger.error(f"Pexels API error: {response.status_code}")
                return []

            data = response.json()
            images = []

            for photo in data.get("photos", []):
                images.append(StockImage(
                    id=photo["id"],
                    url=photo["url"],
                    width=photo["width"],
                    height=photo["height"],
                    download_url=photo["src"]["large2x"],  # High quality
                    photographer=photo["photographer"],
                    alt=photo.get("alt", "")
                ))

            logger.info(f"Found {len(images)} images for: {query}")
            return images

        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return []

    def download_video(
        self,
        video: StockVideo,
        output_path: str,
        timeout: int = 120
    ) -> Optional[str]:
        """
        Download a stock video.

        Args:
            video: StockVideo object
            output_path: Where to save the video
            timeout: Download timeout in seconds

        Returns:
            Path to downloaded file or None
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Check cache first
            cache_file = self.cache_dir / f"video_{video.id}.mp4"
            if cache_file.exists():
                # Copy from cache
                import shutil
                shutil.copy(cache_file, output_path)
                logger.debug(f"Using cached video: {video.id}")
                return output_path

            logger.info(f"Downloading video {video.id} ({video.duration}s)...")

            response = requests.get(video.download_url, stream=True, timeout=timeout)

            if response.status_code != 200:
                logger.error(f"Download failed: {response.status_code}")
                return None

            # Save to cache and output
            with open(cache_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            import shutil
            shutil.copy(cache_file, output_path)

            logger.success(f"Downloaded: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None

    def download_image(
        self,
        image: StockImage,
        output_path: str
    ) -> Optional[str]:
        """Download a stock image."""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            response = requests.get(image.download_url, timeout=30)

            if response.status_code != 200:
                return None

            with open(output_path, 'wb') as f:
                f.write(response.content)

            logger.debug(f"Downloaded image: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Image download failed: {e}")
            return None

    def get_videos_for_script(
        self,
        script_sections: List[Dict],
        niche: str = "general",
        topic: str = ""
    ) -> List[Dict]:
        """
        Get relevant videos for each script section.

        Uses smart keyword detection for better stock footage matching.

        Args:
            script_sections: List of script sections with keywords
            niche: Content niche for fallback keywords
            topic: Overall topic for smart keyword detection

        Returns:
            List of dicts with section info and matched videos
        """
        results = []

        # Get smart keywords based on topic and niche
        if topic:
            smart_keywords = self.get_smart_keywords(topic, niche)
        else:
            niche_data = self.NICHE_KEYWORDS.get(niche, {})
            smart_keywords = niche_data.get("default", ["abstract", "background", "nature"])

        for section in script_sections:
            # Extract keywords from section
            keywords = section.get("keywords", [])
            title = section.get("title", "")

            # Build search query using section keywords or smart keywords
            if keywords:
                query = " ".join(keywords[:3])
            elif title:
                # Use smart keywords based on section title
                section_keywords = self.get_smart_keywords(title, niche)
                query = section_keywords[0] if section_keywords else title
            else:
                query = random.choice(smart_keywords) if smart_keywords else "abstract"

            # Search for videos
            videos = self.search_videos(query, count=3, min_duration=5, max_duration=30)

            # Fallback to smart keywords if no results
            if not videos and smart_keywords:
                fallback = random.choice(smart_keywords)
                videos = self.search_videos(fallback, count=3)

            results.append({
                "section": section,
                "query": query,
                "videos": videos
            })

        return results

    def get_b_roll_clips(
        self,
        topic: str,
        niche: str = "general",
        count: int = 10,
        total_duration: int = 300  # 5 minutes
    ) -> List[StockVideo]:
        """
        Get B-roll clips for a topic.

        Uses smart keyword detection for better stock footage matching.
        Ensures we have enough footage to cover the target duration.

        Args:
            topic: Main topic for searching
            niche: Content niche for smart keyword detection
            count: Number of clips to fetch
            total_duration: Target total duration in seconds

        Returns:
            List of StockVideo objects
        """
        all_videos = []
        total_time = 0

        # Use smart keywords based on topic and niche
        smart_keywords = self.get_smart_keywords(topic, niche)

        # Build search terms: start with topic, then smart keywords
        search_terms = [topic] + smart_keywords

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in search_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)

        logger.info(f"Searching B-roll with terms: {unique_terms[:5]}...")

        # Search each term
        for term in unique_terms[:8]:  # Limit API calls
            if total_time >= total_duration:
                break

            videos = self.search_videos(term, count=5, min_duration=5, max_duration=30)

            for video in videos:
                if video.id not in [v.id for v in all_videos]:  # No duplicates
                    all_videos.append(video)
                    total_time += video.duration

                    if len(all_videos) >= count or total_time >= total_duration:
                        break

        # If still not enough, use default keywords
        if total_time < total_duration:
            niche_data = self.NICHE_KEYWORDS.get(niche, {})
            default_kw = niche_data.get("default", ["abstract background", "cinematic"])
            for term in default_kw[:3]:
                if term.lower() not in seen:
                    videos = self.search_videos(term, count=3, min_duration=5, max_duration=30)
                    for video in videos:
                        if video.id not in [v.id for v in all_videos]:
                            all_videos.append(video)
                            total_time += video.duration

        logger.info(f"Collected {len(all_videos)} B-roll clips ({total_time}s total)")
        return all_videos


# Fallback: Generate placeholder clips when no API key
class PlaceholderFootage:
    """Generate placeholder content when Pexels API unavailable."""

    @staticmethod
    def create_gradient_video(
        output_path: str,
        duration: int = 10,
        colors: List[str] = None
    ) -> str:
        """Create a gradient background video using FFmpeg."""
        from .video_fast import FastVideoGenerator

        gen = FastVideoGenerator()
        if not gen.ffmpeg:
            return None

        colors = colors or ["#1a1a2e", "#16213e", "#0f3460"]
        color = random.choice(colors)

        # Create background image
        bg_path = str(Path(output_path).with_suffix('.bg.png'))
        gen.create_background_image(bg_path)

        # Create silent video
        import subprocess
        cmd = [
            gen.ffmpeg, '-y',
            '-loop', '1',
            '-i', bg_path,
            '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
            '-t', str(duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-shortest',
            output_path
        ]

        subprocess.run(cmd, capture_output=True, timeout=60)

        # Cleanup
        if os.path.exists(bg_path):
            os.remove(bg_path)

        return output_path if os.path.exists(output_path) else None


# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  STOCK FOOTAGE TEST")
    print("="*60 + "\n")

    stock = StockFootage()

    # Test smart keyword detection
    print("Testing Smart Keyword Detection:\n")

    test_cases = [
        ("10 Signs You're Dating a Covert Narcissist", "psychology"),
        ("How Gaslighting Destroys Your Self-Esteem", "psychology"),
        ("The Missing Girl: A Cold Case Investigation", "storytelling"),
        ("True Crime: The Serial Killer Next Door", "storytelling"),
        ("How to Build Passive Income with Dividend Stocks", "finance"),
        ("Crypto Market Analysis: Bitcoin Price Prediction", "finance"),
        ("Overcome Anxiety with These Simple Techniques", "psychology"),
    ]

    for topic, niche in test_cases:
        keywords = stock.get_smart_keywords(topic, niche)
        print(f"  Topic: '{topic[:50]}...'")
        print(f"  Niche: {niche}")
        print(f"  Keywords: {keywords[:5]}")
        print()

    if stock.api_key:
        print("\n" + "-"*60)
        print("Testing with API (fetching clips):\n")

        # Test B-roll with smart keywords
        videos = stock.get_b_roll_clips(
            topic="How Narcissists Manipulate Their Victims",
            niche="psychology",
            count=5
        )

        print(f"Found {len(videos)} videos:\n")
        for v in videos:
            print(f"  - {v.id}: {v.duration}s ({v.width}x{v.height}) by {v.photographer}")
    else:
        print("\nNo API key. Set PEXELS_API_KEY in .env file")
        print("Get free key: https://www.pexels.com/api/")
