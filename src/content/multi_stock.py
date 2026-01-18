"""
Multi-Source Stock Footage Module

Combines Pexels + Pixabay APIs for maximum variety.
Both are FREE with generous limits.

Pexels: 200 req/hour, 20,000/month
Pixabay: 100 req/minute (requires approval)

Usage:
    from src.content.multi_stock import MultiStockProvider

    stock = MultiStockProvider()
    clips = stock.get_clips_for_topic("passive income", count=10)
"""

import os
import random
import hashlib
import requests
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
from loguru import logger
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables from relative paths (portable)
_env_paths = [
    Path(__file__).parent.parent.parent / "config" / ".env",  # src/content -> root/config
    Path.cwd() / "config" / ".env",  # Current working directory
]
for _env_path in _env_paths:
    if _env_path.exists():
        load_dotenv(_env_path)
        logger.debug(f"Loaded .env from: {_env_path}")
        break


@dataclass
class StockClip:
    """Unified stock clip from any source."""
    id: str
    source: str  # "pexels" or "pixabay"
    url: str
    download_url: str
    preview_url: str
    duration: int  # seconds
    width: int
    height: int
    tags: List[str] = field(default_factory=list)
    photographer: str = ""


class PexelsClient:
    """Pexels API client."""

    BASE_URL = "https://api.pexels.com"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"Authorization": api_key}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError))
    )
    def _search_videos_request(self, query: str, count: int) -> requests.Response:
        """Make the API request for video search with retry logic."""
        response = requests.get(
            f"{self.BASE_URL}/videos/search",
            headers=self.headers,
            params={
                "query": query,
                "per_page": min(count * 2, 80),
                "orientation": "landscape",
                "size": "medium"
            },
            timeout=30
        )
        return response

    def search_videos(self, query: str, count: int = 10) -> List[StockClip]:
        """Search Pexels for videos."""
        try:
            response = self._search_videos_request(query, count)

            if response.status_code != 200:
                logger.warning(f"Pexels API error: {response.status_code}")
                return []

            clips = []
            for video in response.json().get("videos", []):
                duration = video.get("duration", 0)
                if duration < 3 or duration > 60:
                    continue

                # Get best video file
                video_files = video.get("video_files", [])
                best_file = None
                for vf in sorted(video_files, key=lambda x: x.get("width", 0), reverse=True):
                    if vf.get("width", 0) >= 1280:
                        best_file = vf
                        break

                if not best_file and video_files:
                    best_file = video_files[0]

                if not best_file:
                    continue

                preview_pics = video.get("video_pictures", [])

                clips.append(StockClip(
                    id=f"pexels_{video['id']}",
                    source="pexels",
                    url=video["url"],
                    download_url=best_file["link"],
                    preview_url=preview_pics[0]["picture"] if preview_pics else "",
                    duration=duration,
                    width=best_file.get("width", 1920),
                    height=best_file.get("height", 1080),
                    tags=query.split(),
                    photographer=video.get("user", {}).get("name", "")
                ))

                if len(clips) >= count:
                    break

            return clips

        except (requests.RequestException, ConnectionError, TimeoutError, KeyError, ValueError) as e:
            logger.error(f"Pexels search failed: {e}")
            return []

    def search_images(self, query: str, count: int = 10) -> List[Dict]:
        """Search Pexels for images."""
        try:
            response = requests.get(
                f"{self.BASE_URL}/v1/search",
                headers=self.headers,
                params={
                    "query": query,
                    "per_page": count,
                    "orientation": "landscape"
                },
                timeout=30
            )

            if response.status_code != 200:
                return []

            images = []
            for photo in response.json().get("photos", []):
                images.append({
                    "id": f"pexels_img_{photo['id']}",
                    "source": "pexels",
                    "url": photo["src"]["large2x"],
                    "width": photo["width"],
                    "height": photo["height"]
                })

            return images

        except (requests.RequestException, ConnectionError, TimeoutError, KeyError, ValueError) as e:
            logger.error(f"Pexels image search failed: {e}")
            return []


class PixabayClient:
    """Pixabay API client."""

    BASE_URL = "https://pixabay.com/api"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError))
    )
    def _search_videos_request(self, query: str, count: int) -> requests.Response:
        """Make the API request for video search with retry logic."""
        response = requests.get(
            f"{self.BASE_URL}/videos/",
            params={
                "key": self.api_key,
                "q": query,
                "per_page": min(count * 2, 200),
                "video_type": "film",
                "min_width": 1280
            },
            timeout=30
        )
        return response

    def search_videos(self, query: str, count: int = 10) -> List[StockClip]:
        """Search Pixabay for videos."""
        try:
            response = self._search_videos_request(query, count)

            if response.status_code != 200:
                logger.warning(f"Pixabay API error: {response.status_code}")
                return []

            clips = []
            for video in response.json().get("hits", []):
                duration = video.get("duration", 0)
                if duration < 3 or duration > 60:
                    continue

                # Get video URLs
                videos = video.get("videos", {})
                large = videos.get("large", {})
                medium = videos.get("medium", {})

                best = large if large.get("url") else medium
                if not best.get("url"):
                    continue

                clips.append(StockClip(
                    id=f"pixabay_{video['id']}",
                    source="pixabay",
                    url=video.get("pageURL", ""),
                    download_url=best["url"],
                    preview_url=video.get("userImageURL", ""),
                    duration=duration,
                    width=best.get("width", 1920),
                    height=best.get("height", 1080),
                    tags=video.get("tags", "").split(", "),
                    photographer=video.get("user", "")
                ))

                if len(clips) >= count:
                    break

            return clips

        except (requests.RequestException, ConnectionError, TimeoutError, KeyError, ValueError) as e:
            logger.error(f"Pixabay search failed: {e}")
            return []

    def search_images(self, query: str, count: int = 10) -> List[Dict]:
        """Search Pixabay for images."""
        try:
            response = requests.get(
                f"{self.BASE_URL}/",
                params={
                    "key": self.api_key,
                    "q": query,
                    "per_page": count,
                    "image_type": "photo",
                    "orientation": "horizontal",
                    "min_width": 1920
                },
                timeout=30
            )

            if response.status_code != 200:
                return []

            images = []
            for img in response.json().get("hits", []):
                images.append({
                    "id": f"pixabay_img_{img['id']}",
                    "source": "pixabay",
                    "url": img.get("largeImageURL", img.get("webformatURL")),
                    "width": img.get("imageWidth", 1920),
                    "height": img.get("imageHeight", 1080)
                })

            return images

        except (requests.RequestException, ConnectionError, TimeoutError, KeyError, ValueError) as e:
            logger.error(f"Pixabay image search failed: {e}")
            return []


class MultiStockProvider:
    """
    Multi-source stock footage provider.
    Combines Pexels and Pixabay for maximum variety.
    """

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

    def __init__(self):
        """Initialize with available API clients."""
        self.clients = []
        self.cache_dir = Path("cache/stock_multi")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._api_verified = False
        self._last_api_error = None

        # Initialize Pexels
        pexels_key = os.getenv("PEXELS_API_KEY")
        if pexels_key:
            if pexels_key.startswith("#") or pexels_key.lower() in ("your_key_here", "your_pexels_key_here", ""):
                logger.warning("PEXELS_API_KEY appears to be a placeholder - please set a real API key")
            else:
                self.clients.append(("pexels", PexelsClient(pexels_key)))
                logger.info("Pexels API initialized")
        else:
            logger.warning("PEXELS_API_KEY not found in environment. Get free key at: https://www.pexels.com/api/")

        # Initialize Pixabay
        pixabay_key = os.getenv("PIXABAY_API_KEY")
        if pixabay_key:
            if pixabay_key.startswith("#") or pixabay_key.lower() in ("your_key_here", "your_pixabay_key_here", ""):
                logger.warning("PIXABAY_API_KEY appears to be a placeholder - please set a real API key")
            else:
                self.clients.append(("pixabay", PixabayClient(pixabay_key)))
                logger.info("Pixabay API initialized")
        else:
            logger.debug("PIXABAY_API_KEY not found - Pixabay source disabled (optional)")

        if not self.clients:
            logger.error(
                "NO STOCK FOOTAGE APIs CONFIGURED!\n"
                "  Videos will only use gradient backgrounds (very small file sizes).\n"
                "  To fix: Add PEXELS_API_KEY and/or PIXABAY_API_KEY to config/.env\n"
                "  Get free Pexels key: https://www.pexels.com/api/\n"
                "  Get free Pixabay key: https://pixabay.com/api/docs/"
            )
        else:
            logger.info(f"MultiStockProvider initialized with {len(self.clients)} source(s): {[c[0] for c in self.clients]}")

    def verify_api_connectivity(self) -> Dict[str, bool]:
        """
        Test API connectivity for all configured providers.

        Returns:
            Dict mapping provider name to connectivity status (True/False)
        """
        results = {}

        if not self.clients:
            logger.error("No API clients configured - cannot verify connectivity")
            return {"error": "No API clients configured"}

        for name, client in self.clients:
            try:
                logger.info(f"Testing {name} API connectivity...")
                # Try a simple search to verify the API works
                if hasattr(client, 'search_videos'):
                    test_clips = client.search_videos("nature", count=1)
                    if test_clips:
                        results[name] = True
                        logger.success(f"{name} API: Connected and working (found {len(test_clips)} test clip)")
                    else:
                        results[name] = False
                        logger.warning(f"{name} API: Connected but returned no results (may be rate limited or invalid key)")
                else:
                    results[name] = False
                    logger.warning(f"{name} API: Client missing search_videos method")
            except requests.exceptions.HTTPError as e:
                results[name] = False
                if e.response and e.response.status_code == 401:
                    logger.error(f"{name} API: Authentication failed - invalid API key")
                elif e.response and e.response.status_code == 429:
                    logger.warning(f"{name} API: Rate limited - try again later")
                else:
                    logger.error(f"{name} API: HTTP error {e.response.status_code if e.response else 'unknown'}")
            except requests.exceptions.ConnectionError:
                results[name] = False
                logger.error(f"{name} API: Connection failed - check internet connection")
            except requests.exceptions.Timeout:
                results[name] = False
                logger.error(f"{name} API: Request timed out")
            except Exception as e:
                results[name] = False
                logger.error(f"{name} API: Unexpected error - {type(e).__name__}: {e}")

        self._api_verified = any(results.values())
        if not self._api_verified:
            logger.error("All stock footage APIs failed! Videos will use gradient fallbacks.")

        return results

    def is_available(self) -> bool:
        """
        Check if at least one stock provider is available.

        Returns:
            True if at least one provider is configured
        """
        return len(self.clients) > 0

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

    def search_videos(self, query: str, count: int = 10) -> List[StockClip]:
        """Search all sources for videos."""
        all_clips = []
        per_source = (count // len(self.clients)) + 1 if self.clients else 0

        for name, client in self.clients:
            clips = client.search_videos(query, per_source)
            all_clips.extend(clips)
            logger.debug(f"{name}: Found {len(clips)} clips for '{query}'")

        # Shuffle to mix sources
        random.shuffle(all_clips)
        return all_clips[:count]

    def search_images(self, query: str, count: int = 10) -> List[Dict]:
        """Search all sources for images."""
        all_images = []
        per_source = (count // len(self.clients)) + 1 if self.clients else 0

        for name, client in self.clients:
            images = client.search_images(query, per_source)
            all_images.extend(images)

        random.shuffle(all_images)
        return all_images[:count]

    def get_clips_for_topic(
        self,
        topic: str,
        niche: str = "default",
        count: int = 15,
        min_total_duration: int = 300
    ) -> List[StockClip]:
        """
        Get diverse clips for a topic with fallbacks.

        Uses smart keyword detection to find topic-specific stock footage
        instead of generic niche keywords.

        Args:
            topic: Main topic to search
            niche: Content niche for keyword expansion
            count: Number of clips to fetch
            min_total_duration: Minimum total duration needed

        Returns:
            List of StockClip objects
        """
        all_clips = []
        seen_ids = set()
        total_duration = 0

        # Use smart keywords based on topic analysis
        smart_keywords = self.get_smart_keywords(topic, niche)

        # Build search queries: start with the full topic, then smart keywords
        queries = [topic] + smart_keywords

        # Remove duplicates while preserving order
        seen_queries = set()
        unique_queries = []
        for q in queries:
            if q.lower() not in seen_queries:
                seen_queries.add(q.lower())
                unique_queries.append(q)

        logger.info(f"Searching for clips with queries: {unique_queries[:5]}...")

        # Search each query
        for query in unique_queries[:10]:
            if len(all_clips) >= count and total_duration >= min_total_duration:
                break

            clips = self.search_videos(query, count=5)

            for clip in clips:
                if clip.id not in seen_ids:
                    seen_ids.add(clip.id)
                    all_clips.append(clip)
                    total_duration += clip.duration

                    if len(all_clips) >= count and total_duration >= min_total_duration:
                        break

        # If still not enough, try default keywords for the niche
        if total_duration < min_total_duration:
            niche_data = self.NICHE_KEYWORDS.get(niche, {})
            default_kw = niche_data.get("default", ["abstract background", "cinematic"])
            for query in default_kw[:3]:
                if query.lower() not in seen_queries:
                    clips = self.search_videos(query, count=3)
                    for clip in clips:
                        if clip.id not in seen_ids:
                            seen_ids.add(clip.id)
                            all_clips.append(clip)
                            total_duration += clip.duration

        logger.info(f"Collected {len(all_clips)} clips ({total_duration}s total) for topic: {topic}")
        return all_clips

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError))
    )
    def _download_clip_request(self, download_url: str) -> requests.Response:
        """Make the download request with retry logic."""
        response = requests.get(download_url, stream=True, timeout=120)
        response.raise_for_status()
        return response

    def download_clip(self, clip: StockClip, output_dir: str = None) -> Optional[str]:
        """
        Download a stock clip to local file.

        Includes validation to ensure the file was actually downloaded
        and is a valid video file (not empty or corrupted).

        Args:
            clip: StockClip object to download
            output_dir: Optional output directory

        Returns:
            Path to downloaded file or None if download failed
        """
        output_dir = output_dir or str(self.cache_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Check cache
        cache_file = self.cache_dir / f"{clip.id}.mp4"
        if cache_file.exists():
            # Validate cached file is not empty/corrupted
            file_size = cache_file.stat().st_size
            if file_size > 10000:  # At least 10KB for a valid video
                logger.debug(f"Using cached: {clip.id} ({file_size / 1024:.1f} KB)")
                return str(cache_file)
            else:
                logger.warning(f"Cached file {clip.id} is too small ({file_size} bytes) - re-downloading")
                cache_file.unlink()  # Delete corrupt cache

        try:
            logger.info(f"Downloading {clip.id} ({clip.duration}s) from {clip.source}...")
            response = self._download_clip_request(clip.download_url)

            with open(cache_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Validate download
            if not cache_file.exists():
                logger.error(f"Download failed: file not created for {clip.id}")
                return None

            file_size = cache_file.stat().st_size
            min_valid_size = 10000  # 10KB minimum for a valid video

            if file_size < min_valid_size:
                logger.error(
                    f"Download failed: {clip.id} is only {file_size} bytes "
                    f"(expected at least {min_valid_size} bytes). "
                    f"API may be returning error response instead of video."
                )
                # Try to read and log error response
                try:
                    with open(cache_file, 'r', errors='ignore') as f:
                        content = f.read(500)
                        if content.strip().startswith(('<', '{', 'error', 'Error')):
                            logger.error(f"API error response: {content[:200]}")
                except:
                    pass
                cache_file.unlink()  # Delete invalid file
                return None

            logger.success(f"Downloaded: {clip.id} ({file_size / 1024:.1f} KB)")
            return str(cache_file)

        except (requests.RequestException, ConnectionError, TimeoutError, OSError, IOError) as e:
            logger.error(f"Download failed for {clip.id}: {type(e).__name__}: {e}")
            # Clean up partial download
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except:
                    pass
            return None

    def download_image(self, image: Dict, output_dir: str = None) -> Optional[str]:
        """Download a stock image."""
        output_dir = output_dir or str(self.cache_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        img_id = image.get("id", hashlib.md5(image["url"].encode()).hexdigest())
        cache_file = self.cache_dir / f"{img_id}.jpg"

        if cache_file.exists():
            return str(cache_file)

        try:
            response = requests.get(image["url"], timeout=30)
            if response.status_code == 200:
                with open(cache_file, 'wb') as f:
                    f.write(response.content)
                return str(cache_file)
        except (requests.RequestException, ConnectionError, TimeoutError, OSError, IOError, KeyError) as e:
            logger.error(f"Image download failed: {e}")

        return None


def diagnose_stock_api() -> Dict[str, any]:
    """
    Diagnostic function to test stock API connectivity and configuration.

    Run this to troubleshoot why videos might only be 0.10 MB (gradient fallback).

    Returns:
        Dict with diagnostic results
    """
    print("\n" + "=" * 70)
    print("  STOCK FOOTAGE API DIAGNOSTIC")
    print("=" * 70 + "\n")

    results = {
        "env_loaded": False,
        "pexels_key_found": False,
        "pixabay_key_found": False,
        "pexels_key_valid": False,
        "pixabay_key_valid": False,
        "apis_working": {},
        "test_download": False,
        "recommendations": []
    }

    # Check environment loading
    from dotenv import load_dotenv
    env_paths = [
        Path(__file__).parent.parent.parent / "config" / ".env",
        Path.cwd() / "config" / ".env",
    ]

    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            results["env_loaded"] = True
            print(f"[OK] Environment loaded from: {env_path}")
            break

    if not results["env_loaded"]:
        print("[ERROR] No .env file found!")
        results["recommendations"].append("Create config/.env file with API keys")

    # Check API keys
    pexels_key = os.getenv("PEXELS_API_KEY", "")
    pixabay_key = os.getenv("PIXABAY_API_KEY", "")

    # Pexels check
    if pexels_key:
        if pexels_key.startswith("#") or pexels_key.lower() in ("your_key_here", ""):
            print("[WARNING] PEXELS_API_KEY is a placeholder value")
            results["recommendations"].append("Set a real PEXELS_API_KEY in config/.env")
        else:
            results["pexels_key_found"] = True
            print(f"[OK] PEXELS_API_KEY found ({len(pexels_key)} chars)")
    else:
        print("[WARNING] PEXELS_API_KEY not set")
        results["recommendations"].append("Add PEXELS_API_KEY to config/.env (get free key at https://www.pexels.com/api/)")

    # Pixabay check
    if pixabay_key:
        if pixabay_key.startswith("#") or pixabay_key.lower() in ("your_key_here", ""):
            print("[INFO] PIXABAY_API_KEY is a placeholder value")
        else:
            results["pixabay_key_found"] = True
            print(f"[OK] PIXABAY_API_KEY found ({len(pixabay_key)} chars)")
    else:
        print("[INFO] PIXABAY_API_KEY not set (optional)")

    print()

    # Test API connectivity
    print("Testing API connectivity...\n")

    provider = MultiStockProvider()
    if provider.clients:
        api_results = provider.verify_api_connectivity()
        results["apis_working"] = api_results

        for api_name, is_working in api_results.items():
            if api_name == "error":
                print(f"[ERROR] {is_working}")
            elif is_working:
                print(f"[OK] {api_name.upper()} API is working")
                if api_name == "pexels":
                    results["pexels_key_valid"] = True
                elif api_name == "pixabay":
                    results["pixabay_key_valid"] = True
            else:
                print(f"[ERROR] {api_name.upper()} API is NOT working")
                results["recommendations"].append(f"Check your {api_name.upper()}_API_KEY - it may be invalid or expired")
    else:
        print("[ERROR] No API clients configured!")
        results["recommendations"].append("At least one API key (PEXELS or PIXABAY) must be configured")

    print()

    # Test download if API is working
    if any(results["apis_working"].values()):
        print("Testing clip search and download...\n")
        try:
            clips = provider.search_videos("nature", count=1)
            if clips:
                print(f"[OK] Found {len(clips)} test clip(s)")
                test_path = provider.download_clip(clips[0])
                if test_path:
                    file_size = Path(test_path).stat().st_size
                    print(f"[OK] Successfully downloaded test clip ({file_size / 1024:.1f} KB)")
                    results["test_download"] = True

                    if file_size < 10000:
                        print("[WARNING] Downloaded file is suspiciously small - may be an error response")
                        results["test_download"] = False
                        results["recommendations"].append("API key may be invalid or rate limited")
                else:
                    print("[ERROR] Download failed")
                    results["recommendations"].append("Check internet connection and API key validity")
            else:
                print("[WARNING] No clips found in test search")
                results["recommendations"].append("API may be rate limited or key invalid")
        except Exception as e:
            print(f"[ERROR] Test failed: {e}")
            results["recommendations"].append(f"Fix error: {e}")

    print()

    # Summary
    print("=" * 70)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 70 + "\n")

    if results["test_download"]:
        print("[SUCCESS] Stock footage API is working correctly!")
        print("  Videos should download real stock footage, not just gradients.")
    else:
        print("[PROBLEM] Stock footage is NOT working properly.")
        print("  Videos will fall back to gradient backgrounds (very small file size).")

    if results["recommendations"]:
        print("\nRecommendations:")
        for i, rec in enumerate(results["recommendations"], 1):
            print(f"  {i}. {rec}")

    print()
    return results


# Quick test
if __name__ == "__main__":
    import sys

    # If --diagnose flag is passed, run diagnostic
    if len(sys.argv) > 1 and sys.argv[1] in ("--diagnose", "-d", "diagnose"):
        diagnose_stock_api()
        sys.exit(0)

    print("\n" + "="*60)
    print("MULTI-SOURCE STOCK FOOTAGE TEST")
    print("="*60 + "\n")
    print("TIP: Run with --diagnose flag for full API diagnostic\n")

    stock = MultiStockProvider()

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

    if stock.clients:
        print("\n" + "-"*60)
        print("Testing with API (fetching clips):\n")

        clips = stock.get_clips_for_topic(
            topic="How Narcissists Manipulate Their Victims",
            niche="psychology",
            count=10
        )

        print(f"\nFound {len(clips)} clips:\n")
        for clip in clips:
            print(f"  [{clip.source}] {clip.id}: {clip.duration}s - {clip.tags[:3]}")
    else:
        print("\nNo API keys configured!")
        print("Add PEXELS_API_KEY and/or PIXABAY_API_KEY to config/.env")
        print("\nRun with --diagnose flag for detailed troubleshooting.")
