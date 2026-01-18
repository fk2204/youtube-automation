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

    # Fallback keywords for different niches
    NICHE_KEYWORDS = {
        "finance": ["money", "business", "office", "success", "growth", "investment", "laptop work", "stock market"],
        "psychology": ["brain", "mind", "thinking", "meditation", "abstract", "mystery", "dark", "contemplation"],
        "storytelling": ["mystery", "dark", "night", "abandoned", "fog", "suspense", "detective", "crime scene"],
        "technology": ["computer", "coding", "technology", "digital", "futuristic", "data", "network"],
        "motivation": ["success", "mountain top", "sunrise", "running", "fitness", "achievement", "victory"]
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
        niche: str = "general"
    ) -> List[Dict]:
        """
        Get relevant videos for each script section.

        Args:
            script_sections: List of script sections with keywords
            niche: Content niche for fallback keywords

        Returns:
            List of dicts with section info and matched videos
        """
        results = []

        # Get niche-specific keywords for fallback
        fallback_keywords = self.NICHE_KEYWORDS.get(niche, ["abstract", "background", "nature"])

        for section in script_sections:
            # Extract keywords from section
            keywords = section.get("keywords", [])
            title = section.get("title", "")

            # Build search query
            if keywords:
                query = " ".join(keywords[:3])
            elif title:
                query = title
            else:
                query = random.choice(fallback_keywords)

            # Search for videos
            videos = self.search_videos(query, count=3, min_duration=5, max_duration=30)

            # Fallback to niche keywords if no results
            if not videos:
                fallback = random.choice(fallback_keywords)
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
        count: int = 10,
        total_duration: int = 300  # 5 minutes
    ) -> List[StockVideo]:
        """
        Get B-roll clips for a topic.

        Ensures we have enough footage to cover the target duration.

        Args:
            topic: Main topic for searching
            count: Number of clips to fetch
            total_duration: Target total duration in seconds

        Returns:
            List of StockVideo objects
        """
        all_videos = []
        total_time = 0

        # Generate search variations
        search_terms = [topic]
        words = topic.lower().split()

        # Add individual words
        search_terms.extend(words[:3])

        # Add related terms based on common patterns
        related = {
            "money": ["cash", "wealth", "finance", "business"],
            "income": ["earnings", "profit", "money"],
            "psychology": ["brain", "mind", "thinking"],
            "dark": ["mystery", "shadow", "night"],
            "story": ["documentary", "narrative"],
            "crime": ["detective", "investigation", "mystery"]
        }

        for word in words:
            if word in related:
                search_terms.extend(related[word][:2])

        # Search each term
        for term in search_terms[:5]:  # Limit API calls
            if total_time >= total_duration:
                break

            videos = self.search_videos(term, count=5, min_duration=5, max_duration=30)

            for video in videos:
                if video.id not in [v.id for v in all_videos]:  # No duplicates
                    all_videos.append(video)
                    total_time += video.duration

                    if len(all_videos) >= count or total_time >= total_duration:
                        break

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

    if stock.api_key:
        # Search videos
        videos = stock.search_videos("passive income business", count=5)

        print(f"Found {len(videos)} videos:\n")
        for v in videos:
            print(f"  - {v.id}: {v.duration}s ({v.width}x{v.height}) by {v.photographer}")

        # Download first video
        if videos:
            print("\nDownloading first video...")
            stock.download_video(videos[0], "output/test_stock.mp4")
    else:
        print("No API key. Set PEXELS_API_KEY in .env file")
        print("Get free key: https://www.pexels.com/api/")
