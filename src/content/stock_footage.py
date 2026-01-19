"""
Stock Footage Module - Multi-Source Integration

Fetches relevant stock videos and images from multiple FREE sources:
- Pexels: 20,000 requests/month, 200/hour
- Pixabay: Unlimited requests (with API key)
- Coverr: Free stock videos (web scraping)

Get API keys:
- Pexels: https://www.pexels.com/api/
- Pixabay: https://pixabay.com/api/docs/

Usage:
    from src.content.stock_footage import StockFootageProvider

    # Multi-source provider (tries Pexels -> Pixabay -> Coverr)
    provider = StockFootageProvider()
    videos = provider.search_videos("passive income", count=5)
    provider.download_video(videos[0], "output/clip1.mp4")

    # Or use a specific source
    from src.content.stock_footage import PexelsProvider, PixabayProvider
    pexels = PexelsProvider()
    pixabay = PixabayProvider()
"""

import os
import json
import time
import hashlib
import requests
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class StockVideo:
    """Stock video metadata."""
    id: Union[int, str]
    url: str
    duration: int  # seconds
    width: int
    height: int
    preview_url: str
    download_url: str  # Best quality
    photographer: str
    tags: List[str]
    source: str = "unknown"  # pexels, pixabay, coverr


@dataclass
class StockImage:
    """Stock image metadata."""
    id: Union[int, str]
    url: str
    width: int
    height: int
    download_url: str
    photographer: str
    alt: str
    source: str = "unknown"


@dataclass
class CacheEntry:
    """Cache entry for search results."""
    data: List
    timestamp: float
    query: str
    source: str


class SearchCache:
    """
    Simple file-based cache for search results.

    Avoids repeated API calls for the same queries.
    Cache expires after configurable TTL (default 1 hour).
    """

    def __init__(self, cache_dir: Path, ttl_seconds: int = 3600):
        """
        Initialize cache.

        Args:
            cache_dir: Directory to store cache files
            ttl_seconds: Time-to-live for cache entries (default 1 hour)
        """
        self.cache_dir = cache_dir / "search_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        self._memory_cache: Dict[str, CacheEntry] = {}

    def _get_cache_key(self, query: str, source: str, params: Dict = None) -> str:
        """Generate a unique cache key."""
        key_data = f"{source}:{query}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, query: str, source: str, params: Dict = None) -> Optional[List]:
        """Get cached results if not expired."""
        key = self._get_cache_key(query, source, params)

        # Check memory cache first
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if time.time() - entry.timestamp < self.ttl:
                logger.debug(f"Cache hit (memory): {query} from {source}")
                return entry.data

        # Check file cache
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                if time.time() - cached['timestamp'] < self.ttl:
                    logger.debug(f"Cache hit (file): {query} from {source}")
                    # Store in memory for faster subsequent access
                    self._memory_cache[key] = CacheEntry(
                        data=cached['data'],
                        timestamp=cached['timestamp'],
                        query=query,
                        source=source
                    )
                    return cached['data']
            except (json.JSONDecodeError, KeyError):
                pass

        return None

    def set(self, query: str, source: str, data: List, params: Dict = None):
        """Store results in cache."""
        key = self._get_cache_key(query, source, params)
        timestamp = time.time()

        # Store in memory
        self._memory_cache[key] = CacheEntry(
            data=data,
            timestamp=timestamp,
            query=query,
            source=source
        )

        # Store in file
        cache_file = self.cache_dir / f"{key}.json"
        try:
            # Convert StockVideo/StockImage objects to dicts for JSON serialization
            serializable_data = []
            for item in data:
                if hasattr(item, '__dict__'):
                    serializable_data.append(item.__dict__)
                else:
                    serializable_data.append(item)

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'data': serializable_data,
                    'timestamp': timestamp,
                    'query': query,
                    'source': source
                }, f)
            logger.debug(f"Cached {len(data)} results for: {query} from {source}")
        except Exception as e:
            logger.warning(f"Failed to write cache: {e}")

    def clear(self, older_than_seconds: int = None):
        """Clear cache entries."""
        self._memory_cache.clear()

        if older_than_seconds:
            cutoff = time.time() - older_than_seconds
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        cached = json.load(f)
                    if cached['timestamp'] < cutoff:
                        cache_file.unlink()
                except:
                    pass
        else:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()


class BaseStockProvider(ABC):
    """Abstract base class for stock footage providers."""

    SOURCE_NAME = "base"

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

    def __init__(self, cache: SearchCache = None):
        """Initialize provider with optional cache."""
        self.cache = cache
        self.requests_made = 0

    @abstractmethod
    def search_videos(
        self,
        query: str,
        count: int = 10,
        min_duration: int = 5,
        max_duration: int = 60,
        orientation: str = "landscape"
    ) -> List[StockVideo]:
        """Search for stock videos."""
        pass

    @abstractmethod
    def search_images(
        self,
        query: str,
        count: int = 10,
        orientation: str = "landscape"
    ) -> List[StockImage]:
        """Search for stock images."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available (API key configured)."""
        pass

    def _detect_subtopic(self, topic: str, niche: str) -> Optional[str]:
        """Detect the sub-topic from the topic text."""
        topic_lower = topic.lower()
        triggers = self.SUBTOPIC_TRIGGERS.get(niche, {})

        scores = {}
        for subtopic, trigger_words in triggers.items():
            score = 0
            for trigger in trigger_words:
                if trigger.lower() in topic_lower:
                    score += len(trigger)
            if score > 0:
                scores[subtopic] = score

        if scores:
            best_subtopic = max(scores, key=scores.get)
            logger.debug(f"Detected subtopic '{best_subtopic}' for topic '{topic}' (score: {scores[best_subtopic]})")
            return best_subtopic

        return None

    def get_smart_keywords(self, topic: str, niche: str) -> List[str]:
        """Get smart keywords based on topic analysis and niche."""
        keywords = []
        niche_data = self.NICHE_KEYWORDS.get(niche, {})

        detected_subtopic = self._detect_subtopic(topic, niche)

        if detected_subtopic and detected_subtopic in niche_data:
            subtopic_keywords = niche_data[detected_subtopic]
            keywords.extend(subtopic_keywords)
            logger.info(f"Using '{detected_subtopic}' keywords for niche '{niche}': {subtopic_keywords[:3]}...")
        else:
            default_keywords = niche_data.get("default", [])
            keywords.extend(default_keywords)
            logger.info(f"Using default keywords for niche '{niche}': {default_keywords[:3]}...")

        topic_words = [w for w in topic.lower().split() if len(w) > 3]
        keywords.extend(topic_words[:3])

        seen = set()
        unique_keywords = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)

        return unique_keywords


class PexelsProvider(BaseStockProvider):
    """
    Pexels API client for stock videos and images.

    Free tier: 200 requests/hour, 20,000/month
    """

    SOURCE_NAME = "pexels"
    BASE_URL = "https://api.pexels.com"

    def __init__(self, api_key: Optional[str] = None, cache: SearchCache = None):
        """
        Initialize Pexels client.

        Args:
            api_key: Pexels API key (or set PEXELS_API_KEY env var)
            cache: Optional SearchCache instance
        """
        super().__init__(cache)
        self.api_key = api_key or os.getenv("PEXELS_API_KEY")

        if not self.api_key:
            logger.warning(
                "Pexels API key not found. Get one free at https://www.pexels.com/api/\n"
                "Set PEXELS_API_KEY in your .env file"
            )
        else:
            logger.info("PexelsProvider initialized")

    def is_available(self) -> bool:
        """Check if Pexels API is available."""
        return bool(self.api_key)

    def _headers(self) -> Dict[str, str]:
        """Get API headers."""
        return {"Authorization": self.api_key}

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
                    tags=query.split(),
                    source="pexels"
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
                    alt=photo.get("alt", ""),
                    source="pexels"
                ))

            logger.info(f"Found {len(images)} images for: {query}")
            return images

        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return []

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


class PixabayProvider(BaseStockProvider):
    """
    Pixabay API client for stock videos and images.

    Free tier: Unlimited requests with API key
    Get key: https://pixabay.com/api/docs/
    """

    SOURCE_NAME = "pixabay"
    BASE_URL = "https://pixabay.com/api"

    def __init__(self, api_key: Optional[str] = None, cache: SearchCache = None):
        """
        Initialize Pixabay client.

        Args:
            api_key: Pixabay API key (or set PIXABAY_API_KEY env var)
            cache: Optional SearchCache instance
        """
        super().__init__(cache)
        self.api_key = api_key or os.getenv("PIXABAY_API_KEY")

        if not self.api_key:
            logger.warning(
                "Pixabay API key not found. Get one free at https://pixabay.com/api/docs/\n"
                "Set PIXABAY_API_KEY in your .env file"
            )
        else:
            logger.info("PixabayProvider initialized")

    def is_available(self) -> bool:
        """Check if Pixabay API is available."""
        return bool(self.api_key)

    def search_videos(
        self,
        query: str,
        count: int = 10,
        min_duration: int = 5,
        max_duration: int = 60,
        orientation: str = "landscape"
    ) -> List[StockVideo]:
        """
        Search for stock videos on Pixabay.

        Args:
            query: Search query
            count: Number of videos to fetch
            min_duration: Minimum video duration in seconds
            max_duration: Maximum video duration
            orientation: horizontal, vertical, or all

        Returns:
            List of StockVideo objects
        """
        if not self.api_key:
            logger.warning("No Pixabay API key - returning empty results")
            return []

        # Check cache first
        if self.cache:
            params_key = {"min_dur": min_duration, "max_dur": max_duration, "orientation": orientation}
            cached = self.cache.get(query, "pixabay_videos", params_key)
            if cached:
                # Reconstruct StockVideo objects from cached dicts
                return [StockVideo(**v) if isinstance(v, dict) else v for v in cached[:count]]

        try:
            # Map orientation names
            orientation_map = {"landscape": "horizontal", "portrait": "vertical", "square": "all"}
            pixabay_orientation = orientation_map.get(orientation, "all")

            url = f"{self.BASE_URL}/videos/"
            params = {
                "key": self.api_key,
                "q": query,
                "per_page": min(count * 2, 200),  # Fetch extra for filtering
                "orientation": pixabay_orientation,
                "safesearch": "true"
            }

            response = requests.get(url, params=params, timeout=30)
            self.requests_made += 1

            if response.status_code != 200:
                logger.error(f"Pixabay API error: {response.status_code} - {response.text[:200]}")
                return []

            data = response.json()
            videos = []

            for video in data.get("hits", []):
                duration = video.get("duration", 0)

                # Filter by duration
                if duration < min_duration or duration > max_duration:
                    continue

                # Get best video file (large preferred)
                video_sizes = video.get("videos", {})
                best_file = None

                # Prefer large > medium > small > tiny
                for size in ["large", "medium", "small", "tiny"]:
                    if size in video_sizes and video_sizes[size].get("url"):
                        best_file = video_sizes[size]
                        break

                if not best_file:
                    continue

                videos.append(StockVideo(
                    id=f"pixabay_{video['id']}",
                    url=video.get("pageURL", ""),
                    duration=duration,
                    width=best_file.get("width", 1920),
                    height=best_file.get("height", 1080),
                    preview_url=video.get("picture_id", ""),
                    download_url=best_file["url"],
                    photographer=video.get("user", "Unknown"),
                    tags=video.get("tags", "").split(", "),
                    source="pixabay"
                ))

                if len(videos) >= count:
                    break

            logger.info(f"[Pixabay] Found {len(videos)} videos for: {query}")

            # Cache the results
            if self.cache and videos:
                params_key = {"min_dur": min_duration, "max_dur": max_duration, "orientation": orientation}
                self.cache.set(query, "pixabay_videos", videos, params_key)

            return videos

        except Exception as e:
            logger.error(f"Pixabay video search failed: {e}")
            return []

    def search_images(
        self,
        query: str,
        count: int = 10,
        orientation: str = "landscape"
    ) -> List[StockImage]:
        """
        Search for stock images on Pixabay.

        Args:
            query: Search query
            count: Number of images to fetch
            orientation: horizontal, vertical, or all

        Returns:
            List of StockImage objects
        """
        if not self.api_key:
            return []

        # Check cache first
        if self.cache:
            params_key = {"orientation": orientation}
            cached = self.cache.get(query, "pixabay_images", params_key)
            if cached:
                return [StockImage(**i) if isinstance(i, dict) else i for i in cached[:count]]

        try:
            # Map orientation names
            orientation_map = {"landscape": "horizontal", "portrait": "vertical", "square": "all"}
            pixabay_orientation = orientation_map.get(orientation, "all")

            url = f"{self.BASE_URL}/"
            params = {
                "key": self.api_key,
                "q": query,
                "per_page": count,
                "orientation": pixabay_orientation,
                "safesearch": "true",
                "image_type": "photo"
            }

            response = requests.get(url, params=params, timeout=30)
            self.requests_made += 1

            if response.status_code != 200:
                logger.error(f"Pixabay API error: {response.status_code}")
                return []

            data = response.json()
            images = []

            for photo in data.get("hits", []):
                images.append(StockImage(
                    id=f"pixabay_{photo['id']}",
                    url=photo.get("pageURL", ""),
                    width=photo.get("imageWidth", 1920),
                    height=photo.get("imageHeight", 1080),
                    download_url=photo.get("largeImageURL", photo.get("webformatURL", "")),
                    photographer=photo.get("user", "Unknown"),
                    alt=photo.get("tags", ""),
                    source="pixabay"
                ))

            logger.info(f"[Pixabay] Found {len(images)} images for: {query}")

            # Cache the results
            if self.cache and images:
                params_key = {"orientation": orientation}
                self.cache.set(query, "pixabay_images", images, params_key)

            return images

        except Exception as e:
            logger.error(f"Pixabay image search failed: {e}")
            return []


class CoverrProvider(BaseStockProvider):
    """
    Coverr provider for free stock videos.

    Uses web scraping to get video URLs from Coverr.co
    No API key required, but be respectful of rate limits.
    """

    SOURCE_NAME = "coverr"
    BASE_URL = "https://coverr.co"
    API_URL = "https://api.coverr.co/videos"

    def __init__(self, cache: SearchCache = None):
        """Initialize Coverr client."""
        super().__init__(cache)
        logger.info("CoverrProvider initialized (no API key required)")

    def is_available(self) -> bool:
        """Coverr is always available (no API key required)."""
        return True

    def search_videos(
        self,
        query: str,
        count: int = 10,
        min_duration: int = 5,
        max_duration: int = 60,
        orientation: str = "landscape"
    ) -> List[StockVideo]:
        """
        Search for stock videos on Coverr.

        Args:
            query: Search query
            count: Number of videos to fetch
            min_duration: Minimum video duration in seconds
            max_duration: Maximum video duration
            orientation: Not used for Coverr (always returns various)

        Returns:
            List of StockVideo objects
        """
        # Check cache first
        if self.cache:
            params_key = {"min_dur": min_duration, "max_dur": max_duration}
            cached = self.cache.get(query, "coverr_videos", params_key)
            if cached:
                return [StockVideo(**v) if isinstance(v, dict) else v for v in cached[:count]]

        try:
            # Try the Coverr API endpoint
            params = {
                "query": query,
                "page": 1,
                "page_size": min(count * 2, 25)
            }

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json"
            }

            response = requests.get(self.API_URL, params=params, headers=headers, timeout=30)
            self.requests_made += 1

            if response.status_code != 200:
                logger.warning(f"Coverr API returned {response.status_code}, trying fallback")
                return self._search_fallback(query, count, min_duration, max_duration)

            data = response.json()
            videos = []

            for video in data.get("videos", data.get("hits", [])):
                # Get duration (Coverr may not always provide this)
                duration = video.get("duration", 15)  # Default to 15 seconds

                # Filter by duration
                if duration < min_duration or duration > max_duration:
                    continue

                # Get video URL
                video_url = video.get("video_url", "")
                if not video_url:
                    # Try to get from urls object
                    urls = video.get("urls", {})
                    video_url = urls.get("mp4_download", urls.get("mp4", ""))

                if not video_url:
                    continue

                videos.append(StockVideo(
                    id=f"coverr_{video.get('id', hash(video_url))}",
                    url=video.get("url", f"{self.BASE_URL}/videos/{video.get('slug', '')}"),
                    duration=duration,
                    width=video.get("width", 1920),
                    height=video.get("height", 1080),
                    preview_url=video.get("thumbnail", video.get("poster", "")),
                    download_url=video_url,
                    photographer=video.get("user", {}).get("name", "Coverr"),
                    tags=video.get("tags", query.split()),
                    source="coverr"
                ))

                if len(videos) >= count:
                    break

            logger.info(f"[Coverr] Found {len(videos)} videos for: {query}")

            # Cache the results
            if self.cache and videos:
                params_key = {"min_dur": min_duration, "max_dur": max_duration}
                self.cache.set(query, "coverr_videos", videos, params_key)

            return videos

        except Exception as e:
            logger.error(f"Coverr video search failed: {e}")
            return []

    def _search_fallback(
        self,
        query: str,
        count: int,
        min_duration: int,
        max_duration: int
    ) -> List[StockVideo]:
        """Fallback search using web scraping if API fails."""
        try:
            # Try to scrape the search page
            search_url = f"{self.BASE_URL}/s/{query.replace(' ', '-')}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(search_url, headers=headers, timeout=30)

            if response.status_code != 200:
                return []

            # Very basic extraction - look for video URLs in the response
            # This is a simplified fallback that may not always work
            videos = []
            content = response.text

            # Look for MP4 URLs
            import re
            mp4_pattern = r'https?://[^\s"\'<>]+\.mp4[^\s"\'<>]*'
            mp4_urls = re.findall(mp4_pattern, content)

            for i, url in enumerate(set(mp4_urls)[:count]):
                videos.append(StockVideo(
                    id=f"coverr_fallback_{i}",
                    url=search_url,
                    duration=15,  # Default duration
                    width=1920,
                    height=1080,
                    preview_url="",
                    download_url=url,
                    photographer="Coverr",
                    tags=query.split(),
                    source="coverr"
                ))

            return videos

        except Exception as e:
            logger.debug(f"Coverr fallback search failed: {e}")
            return []

    def search_images(
        self,
        query: str,
        count: int = 10,
        orientation: str = "landscape"
    ) -> List[StockImage]:
        """Coverr primarily provides videos, not images."""
        logger.debug("Coverr does not provide images, returning empty list")
        return []


class StockFootageProvider(BaseStockProvider):
    """
    Multi-source stock footage provider.

    Tries multiple sources in order (Pexels -> Pixabay -> Coverr)
    with caching to avoid repeated API calls and automatic fallback
    if one source fails or returns no results.

    Usage:
        provider = StockFootageProvider()
        videos = provider.search_videos("nature", count=10)
        provider.download_video(videos[0], "output/clip.mp4")
    """

    SOURCE_NAME = "multi"

    def __init__(
        self,
        pexels_key: Optional[str] = None,
        pixabay_key: Optional[str] = None,
        cache_ttl: int = 3600,
        source_order: List[str] = None
    ):
        """
        Initialize multi-source provider.

        Args:
            pexels_key: Pexels API key (or set PEXELS_API_KEY env var)
            pixabay_key: Pixabay API key (or set PIXABAY_API_KEY env var)
            cache_ttl: Cache time-to-live in seconds (default 1 hour)
            source_order: Order to try sources (default: ["pexels", "pixabay", "coverr"])
        """
        # Initialize shared cache
        self.cache_dir = Path("cache/stock")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = SearchCache(self.cache_dir, cache_ttl)

        super().__init__(self.cache)

        # Initialize providers
        self.providers: Dict[str, BaseStockProvider] = {
            "pexels": PexelsProvider(pexels_key, self.cache),
            "pixabay": PixabayProvider(pixabay_key, self.cache),
            "coverr": CoverrProvider(self.cache)
        }

        # Set source order (customizable)
        self.source_order = source_order or ["pexels", "pixabay", "coverr"]

        # Log available sources
        available = [name for name in self.source_order if self.providers[name].is_available()]
        logger.info(f"StockFootageProvider initialized with sources: {available}")

    def is_available(self) -> bool:
        """Check if at least one provider is available."""
        return any(p.is_available() for p in self.providers.values())

    def get_available_sources(self) -> List[str]:
        """Get list of available source names."""
        return [name for name in self.source_order if self.providers[name].is_available()]

    def search_videos(
        self,
        query: str,
        count: int = 10,
        min_duration: int = 5,
        max_duration: int = 60,
        orientation: str = "landscape"
    ) -> List[StockVideo]:
        """
        Search for stock videos from multiple sources.

        Tries each source in order until we have enough videos.
        Combines results for more variety.

        Args:
            query: Search query
            count: Number of videos to fetch
            min_duration: Minimum video duration in seconds
            max_duration: Maximum video duration
            orientation: landscape, portrait, or square

        Returns:
            List of StockVideo objects from various sources
        """
        all_videos = []
        seen_ids = set()

        for source_name in self.source_order:
            if len(all_videos) >= count:
                break

            provider = self.providers.get(source_name)
            if not provider or not provider.is_available():
                continue

            try:
                needed = count - len(all_videos)
                videos = provider.search_videos(
                    query=query,
                    count=needed + 2,  # Fetch a few extra for filtering duplicates
                    min_duration=min_duration,
                    max_duration=max_duration,
                    orientation=orientation
                )

                # Add unique videos
                for video in videos:
                    video_key = f"{video.source}_{video.id}"
                    if video_key not in seen_ids:
                        seen_ids.add(video_key)
                        all_videos.append(video)

                        if len(all_videos) >= count:
                            break

                if videos:
                    logger.debug(f"Got {len(videos)} videos from {source_name}")

            except Exception as e:
                logger.warning(f"Failed to search {source_name}: {e}")
                continue

        if not all_videos:
            logger.warning(f"No videos found for: {query}")

        return all_videos

    def search_images(
        self,
        query: str,
        count: int = 10,
        orientation: str = "landscape"
    ) -> List[StockImage]:
        """
        Search for stock images from multiple sources.

        Args:
            query: Search query
            count: Number of images to fetch
            orientation: landscape, portrait, or square

        Returns:
            List of StockImage objects from various sources
        """
        all_images = []
        seen_ids = set()

        for source_name in self.source_order:
            if len(all_images) >= count:
                break

            provider = self.providers.get(source_name)
            if not provider or not provider.is_available():
                continue

            try:
                needed = count - len(all_images)
                images = provider.search_images(
                    query=query,
                    count=needed + 2,
                    orientation=orientation
                )

                for image in images:
                    image_key = f"{image.source}_{image.id}"
                    if image_key not in seen_ids:
                        seen_ids.add(image_key)
                        all_images.append(image)

                        if len(all_images) >= count:
                            break

            except Exception as e:
                logger.warning(f"Failed to search images from {source_name}: {e}")
                continue

        return all_images

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

            # Check video cache first
            cache_file = self.cache_dir / f"video_{video.source}_{video.id}.mp4"
            if cache_file.exists():
                import shutil
                shutil.copy(cache_file, output_path)
                logger.debug(f"Using cached video: {video.source}/{video.id}")
                return output_path

            logger.info(f"Downloading video {video.id} from {video.source} ({video.duration}s)...")

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
        Get relevant videos for each script section from multiple sources.

        Args:
            script_sections: List of script sections with keywords
            niche: Content niche for fallback keywords
            topic: Overall topic for smart keyword detection

        Returns:
            List of dicts with section info and matched videos
        """
        results = []

        if topic:
            smart_keywords = self.get_smart_keywords(topic, niche)
        else:
            niche_data = self.NICHE_KEYWORDS.get(niche, {})
            smart_keywords = niche_data.get("default", ["abstract", "background", "nature"])

        for section in script_sections:
            keywords = section.get("keywords", [])
            title = section.get("title", "")

            if keywords:
                query = " ".join(keywords[:3])
            elif title:
                section_keywords = self.get_smart_keywords(title, niche)
                query = section_keywords[0] if section_keywords else title
            else:
                query = random.choice(smart_keywords) if smart_keywords else "abstract"

            videos = self.search_videos(query, count=3, min_duration=5, max_duration=30)

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
        total_duration: int = 300
    ) -> List[StockVideo]:
        """
        Get B-roll clips from multiple sources for variety.

        Args:
            topic: Main topic for searching
            niche: Content niche for smart keyword detection
            count: Number of clips to fetch
            total_duration: Target total duration in seconds

        Returns:
            List of StockVideo objects from various sources
        """
        all_videos = []
        total_time = 0

        smart_keywords = self.get_smart_keywords(topic, niche)
        search_terms = [topic] + smart_keywords

        seen = set()
        unique_terms = []
        for term in search_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)

        logger.info(f"Searching B-roll with terms: {unique_terms[:5]}...")

        for term in unique_terms[:8]:
            if total_time >= total_duration:
                break

            videos = self.search_videos(term, count=5, min_duration=5, max_duration=30)

            for video in videos:
                video_key = f"{video.source}_{video.id}"
                if video_key not in [f"{v.source}_{v.id}" for v in all_videos]:
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
                        video_key = f"{video.source}_{video.id}"
                        if video_key not in [f"{v.source}_{v.id}" for v in all_videos]:
                            all_videos.append(video)
                            total_time += video.duration

        # Log source distribution
        source_counts = {}
        for v in all_videos:
            source_counts[v.source] = source_counts.get(v.source, 0) + 1
        logger.info(f"Collected {len(all_videos)} B-roll clips ({total_time}s total) from: {source_counts}")

        return all_videos

    def clear_cache(self, older_than_hours: int = None):
        """
        Clear the search cache.

        Args:
            older_than_hours: Only clear entries older than this (None = clear all)
        """
        seconds = older_than_hours * 3600 if older_than_hours else None
        self.cache.clear(seconds)
        logger.info(f"Cache cleared{f' (entries older than {older_than_hours}h)' if older_than_hours else ''}")


# Backwards compatibility alias
StockFootage = StockFootageProvider


# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  MULTI-SOURCE STOCK FOOTAGE TEST")
    print("="*60 + "\n")

    # Initialize multi-source provider
    provider = StockFootageProvider()

    # Show available sources
    print(f"Available sources: {provider.get_available_sources()}\n")

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
        keywords = provider.get_smart_keywords(topic, niche)
        print(f"  Topic: '{topic[:50]}...'")
        print(f"  Niche: {niche}")
        print(f"  Keywords: {keywords[:5]}")
        print()

    if provider.is_available():
        print("\n" + "-"*60)
        print("Testing Multi-Source Search (fetching clips):\n")

        # Test B-roll with smart keywords from multiple sources
        videos = provider.get_b_roll_clips(
            topic="How Narcissists Manipulate Their Victims",
            niche="psychology",
            count=10
        )

        print(f"Found {len(videos)} videos from multiple sources:\n")
        for v in videos:
            print(f"  - [{v.source}] {v.id}: {v.duration}s ({v.width}x{v.height}) by {v.photographer}")

        # Show source distribution
        source_counts = {}
        for v in videos:
            source_counts[v.source] = source_counts.get(v.source, 0) + 1
        print(f"\nSource distribution: {source_counts}")
    else:
        print("\nNo API keys configured. Set at least one of:")
        print("  - PEXELS_API_KEY (https://www.pexels.com/api/)")
        print("  - PIXABAY_API_KEY (https://pixabay.com/api/docs/)")
        print("\nCoverr will be used as fallback (no API key required)")
