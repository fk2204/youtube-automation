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

    def search_videos(self, query: str, count: int = 10) -> List[StockClip]:
        """Search Pexels for videos."""
        try:
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

        except Exception as e:
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

        except Exception as e:
            logger.error(f"Pexels image search failed: {e}")
            return []


class PixabayClient:
    """Pixabay API client."""

    BASE_URL = "https://pixabay.com/api"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search_videos(self, query: str, count: int = 10) -> List[StockClip]:
        """Search Pixabay for videos."""
        try:
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

        except Exception as e:
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

        except Exception as e:
            logger.error(f"Pixabay image search failed: {e}")
            return []


class MultiStockProvider:
    """
    Multi-source stock footage provider.
    Combines Pexels and Pixabay for maximum variety.
    """

    # Enhanced niche keywords with more variety
    NICHE_KEYWORDS = {
        "finance": {
            "primary": ["money", "business", "stock market", "investment", "finance"],
            "secondary": ["laptop work", "office", "success", "growth", "wealthy lifestyle"],
            "abstract": ["gold coins", "dollar bills", "trading charts", "calculator", "bank"]
        },
        "psychology": {
            "primary": ["brain", "mind", "thinking person", "meditation", "psychology"],
            "secondary": ["abstract thoughts", "contemplation", "mental health", "therapy", "emotions"],
            "abstract": ["neurons", "silhouette thinking", "mirror reflection", "dark mood", "introspection"]
        },
        "storytelling": {
            "primary": ["mystery", "dark cinematic", "suspense", "investigation", "documentary"],
            "secondary": ["abandoned building", "fog night", "crime scene", "detective", "evidence"],
            "abstract": ["old photographs", "newspaper", "shadows", "flashlight dark", "rain window"]
        },
        "technology": {
            "primary": ["coding", "computer", "technology", "programming", "software"],
            "secondary": ["data center", "circuit board", "futuristic", "digital", "artificial intelligence"],
            "abstract": ["binary code", "network", "cyber", "hologram", "innovation"]
        },
        "motivation": {
            "primary": ["success", "achievement", "winning", "champion", "determination"],
            "secondary": ["mountain climbing", "sunrise", "running athlete", "gym workout", "celebration"],
            "abstract": ["road ahead", "eagle flying", "lion", "fire", "storm"]
        }
    }

    def __init__(self):
        """Initialize with available API clients."""
        self.clients = []
        self.cache_dir = Path("cache/stock_multi")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Pexels
        pexels_key = os.getenv("PEXELS_API_KEY")
        if pexels_key:
            self.clients.append(("pexels", PexelsClient(pexels_key)))
            logger.info("Pexels API initialized")

        # Initialize Pixabay
        pixabay_key = os.getenv("PIXABAY_API_KEY")
        if pixabay_key:
            self.clients.append(("pixabay", PixabayClient(pixabay_key)))
            logger.info("Pixabay API initialized")

        if not self.clients:
            logger.warning("No stock footage APIs configured. Add PEXELS_API_KEY or PIXABAY_API_KEY to .env")
        else:
            logger.info(f"MultiStockProvider initialized with {len(self.clients)} sources")

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

        # Build search queries
        queries = [topic]

        # Add words from topic
        words = topic.lower().split()[:3]
        queries.extend(words)

        # Add niche keywords
        niche_kw = self.NICHE_KEYWORDS.get(niche, self.NICHE_KEYWORDS.get("motivation", {}))
        queries.extend(niche_kw.get("primary", [])[:3])
        queries.extend(niche_kw.get("secondary", [])[:2])

        # Remove duplicates while preserving order
        seen_queries = set()
        unique_queries = []
        for q in queries:
            if q.lower() not in seen_queries:
                seen_queries.add(q.lower())
                unique_queries.append(q)

        # Search each query
        for query in unique_queries[:8]:
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

        # If still not enough, try abstract keywords
        if total_duration < min_total_duration:
            abstract_kw = niche_kw.get("abstract", ["abstract background", "cinematic"])
            for query in abstract_kw[:3]:
                clips = self.search_videos(query, count=3)
                for clip in clips:
                    if clip.id not in seen_ids:
                        seen_ids.add(clip.id)
                        all_clips.append(clip)
                        total_duration += clip.duration

        logger.info(f"Collected {len(all_clips)} clips ({total_duration}s total) for topic: {topic}")
        return all_clips

    def download_clip(self, clip: StockClip, output_dir: str = None) -> Optional[str]:
        """Download a stock clip to local file."""
        output_dir = output_dir or str(self.cache_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Check cache
        cache_file = self.cache_dir / f"{clip.id}.mp4"
        if cache_file.exists():
            logger.debug(f"Using cached: {clip.id}")
            return str(cache_file)

        try:
            logger.info(f"Downloading {clip.id} ({clip.duration}s)...")
            response = requests.get(clip.download_url, stream=True, timeout=120)

            if response.status_code != 200:
                logger.error(f"Download failed: {response.status_code}")
                return None

            with open(cache_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.success(f"Downloaded: {clip.id}")
            return str(cache_file)

        except Exception as e:
            logger.error(f"Download failed: {e}")
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
        except Exception as e:
            logger.error(f"Image download failed: {e}")

        return None


# Quick test
if __name__ == "__main__":
    print("\n" + "="*60)
    print("MULTI-SOURCE STOCK FOOTAGE TEST")
    print("="*60 + "\n")

    stock = MultiStockProvider()

    if stock.clients:
        clips = stock.get_clips_for_topic(
            topic="passive income money",
            niche="finance",
            count=10
        )

        print(f"\nFound {len(clips)} clips:\n")
        for clip in clips:
            print(f"  [{clip.source}] {clip.id}: {clip.duration}s - {clip.tags[:3]}")
    else:
        print("No API keys configured!")
        print("Add PEXELS_API_KEY and/or PIXABAY_API_KEY to config/.env")
