"""
Performance benchmarks for video generation.

Run with: pytest tests/benchmarks/ -v --benchmark-only
Or: python tests/benchmarks/bench_video_generation.py
"""

import time
import tempfile
import os
import sys
import subprocess
import asyncio
import psutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from loguru import logger


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    duration_seconds: float
    memory_mb: float
    file_size_mb: float
    iterations: int
    avg_duration: float
    min_duration: float
    max_duration: float
    timestamp: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return asdict(self)


class VideoBenchmark:
    """Benchmark video generation performance."""

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize video benchmark.

        Args:
            output_dir: Directory for benchmark outputs. Uses temp dir if not specified.
        """
        self.results: List[BenchmarkResult] = []
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"VideoBenchmark initialized, output dir: {self.output_dir}")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def _get_file_size_mb(self, filepath: str) -> float:
        """Get file size in MB."""
        if os.path.exists(filepath):
            return os.path.getsize(filepath) / 1024 / 1024
        return 0.0

    def _create_test_audio(self, duration: int = 10) -> str:
        """Create a test audio file using FFmpeg."""
        output_file = str(self.output_dir / f"test_audio_{duration}s.mp3")
        if os.path.exists(output_file):
            return output_file

        # Generate silence with a tone
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={duration}",
            "-c:a", "libmp3lame",
            "-b:a", "128k",
            output_file
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=30)
            logger.debug(f"Created test audio: {output_file}")
        except Exception as e:
            logger.warning(f"Could not create test audio: {e}")
            # Create a minimal MP3 file as fallback
            with open(output_file, 'wb') as f:
                f.write(b'\xff\xfb\x90\x00' * 1000)  # Minimal MP3 header
        return output_file

    def _create_test_image(self, width: int = 1920, height: int = 1080) -> str:
        """Create a test image for video generation."""
        output_file = str(self.output_dir / f"test_image_{width}x{height}.png")
        if os.path.exists(output_file):
            return output_file

        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (width, height), color=(30, 30, 60))
            draw = ImageDraw.Draw(img)
            # Add some text
            draw.text((width//2 - 100, height//2), "Benchmark Test", fill="white")
            img.save(output_file)
            logger.debug(f"Created test image: {output_file}")
        except ImportError:
            # Create minimal PNG if PIL not available
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c=blue:size={width}x{height}:d=1",
                "-frames:v", "1",
                output_file
            ]
            subprocess.run(cmd, capture_output=True, timeout=10)
        return output_file

    def benchmark_ffmpeg_direct(self, iterations: int = 3) -> BenchmarkResult:
        """
        Benchmark FFmpeg-direct video creation.

        Creates a video directly with FFmpeg, bypassing MoviePy.
        This is typically the fastest method.
        """
        logger.info(f"Running FFmpeg direct benchmark ({iterations} iterations)")
        durations = []
        memory_samples = []
        output_file = ""

        audio_file = self._create_test_audio(10)
        image_file = self._create_test_image()

        for i in range(iterations):
            output_file = str(self.output_dir / f"ffmpeg_direct_{i}.mp4")
            memory_before = self._get_memory_usage()

            start_time = time.perf_counter()

            # FFmpeg direct command - creates video from image + audio
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", image_file,
                "-i", audio_file,
                "-c:v", "libx264",
                "-preset", "fast",
                "-tune", "stillimage",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-pix_fmt", "yuv420p",
                output_file
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, timeout=120)
                duration = time.perf_counter() - start_time
                durations.append(duration)

                memory_after = self._get_memory_usage()
                memory_samples.append(memory_after - memory_before)

                logger.debug(f"  Iteration {i+1}: {duration:.2f}s")
            except subprocess.TimeoutExpired:
                logger.warning(f"FFmpeg timeout on iteration {i+1}")
                durations.append(120.0)
            except Exception as e:
                logger.error(f"FFmpeg error: {e}")
                durations.append(0.0)

        result = BenchmarkResult(
            name="ffmpeg_direct",
            duration_seconds=sum(durations),
            memory_mb=max(memory_samples) if memory_samples else 0,
            file_size_mb=self._get_file_size_mb(output_file),
            iterations=iterations,
            avg_duration=sum(durations) / len(durations) if durations else 0,
            min_duration=min(durations) if durations else 0,
            max_duration=max(durations) if durations else 0,
            metadata={"preset": "fast", "codec": "libx264"}
        )

        self.results.append(result)
        logger.info(f"FFmpeg direct: avg={result.avg_duration:.2f}s, file={result.file_size_mb:.1f}MB")
        return result

    def benchmark_moviepy(self, iterations: int = 3) -> BenchmarkResult:
        """
        Benchmark MoviePy video creation.

        Uses MoviePy's CompositeVideoClip for video assembly.
        """
        logger.info(f"Running MoviePy benchmark ({iterations} iterations)")
        durations = []
        memory_samples = []
        output_file = ""

        try:
            from moviepy.editor import (
                ColorClip, AudioFileClip, CompositeVideoClip
            )
        except ImportError:
            logger.error("MoviePy not installed, skipping benchmark")
            return BenchmarkResult(
                name="moviepy",
                duration_seconds=0, memory_mb=0, file_size_mb=0,
                iterations=0, avg_duration=0, min_duration=0, max_duration=0,
                metadata={"error": "MoviePy not installed"}
            )

        audio_file = self._create_test_audio(10)

        for i in range(iterations):
            output_file = str(self.output_dir / f"moviepy_{i}.mp4")
            memory_before = self._get_memory_usage()

            start_time = time.perf_counter()

            try:
                # Load audio
                audio = AudioFileClip(audio_file)
                duration_sec = audio.duration

                # Create video with colored background
                video = ColorClip(
                    size=(1920, 1080),
                    color=(30, 30, 60),
                    duration=duration_sec
                )
                video = video.set_audio(audio)

                # Write video
                video.write_videofile(
                    output_file,
                    fps=30,
                    codec="libx264",
                    audio_codec="aac",
                    preset="fast",
                    logger=None
                )

                # Cleanup
                video.close()
                audio.close()

                duration = time.perf_counter() - start_time
                durations.append(duration)

                memory_after = self._get_memory_usage()
                memory_samples.append(memory_after - memory_before)

                logger.debug(f"  Iteration {i+1}: {duration:.2f}s")
            except Exception as e:
                logger.error(f"MoviePy error: {e}")
                durations.append(0.0)

        result = BenchmarkResult(
            name="moviepy",
            duration_seconds=sum(durations),
            memory_mb=max(memory_samples) if memory_samples else 0,
            file_size_mb=self._get_file_size_mb(output_file),
            iterations=iterations,
            avg_duration=sum(durations) / len(durations) if durations else 0,
            min_duration=min(durations) if durations else 0,
            max_duration=max(durations) if durations else 0,
            metadata={"method": "CompositeVideoClip"}
        )

        self.results.append(result)
        logger.info(f"MoviePy: avg={result.avg_duration:.2f}s, file={result.file_size_mb:.1f}MB")
        return result

    def benchmark_with_effects(self, iterations: int = 3) -> BenchmarkResult:
        """
        Benchmark video with Ken Burns effects.

        Tests video generation with pan/zoom effects applied.
        """
        logger.info(f"Running video with effects benchmark ({iterations} iterations)")
        durations = []
        memory_samples = []
        output_file = ""

        try:
            from moviepy.editor import ImageClip, AudioFileClip
        except ImportError:
            logger.error("MoviePy not installed, skipping benchmark")
            return BenchmarkResult(
                name="video_with_effects",
                duration_seconds=0, memory_mb=0, file_size_mb=0,
                iterations=0, avg_duration=0, min_duration=0, max_duration=0,
                metadata={"error": "MoviePy not installed"}
            )

        audio_file = self._create_test_audio(10)
        image_file = self._create_test_image()

        for i in range(iterations):
            output_file = str(self.output_dir / f"effects_{i}.mp4")
            memory_before = self._get_memory_usage()

            start_time = time.perf_counter()

            try:
                audio = AudioFileClip(audio_file)
                duration_sec = audio.duration

                # Load image and apply Ken Burns effect (zoom)
                clip = ImageClip(image_file).set_duration(duration_sec)

                # Simple zoom effect using resize
                def zoom_effect(get_frame, t):
                    """Apply zoom effect over time."""
                    zoom_factor = 1 + 0.1 * (t / duration_sec)  # 1.0 to 1.1
                    return clip.resize(zoom_factor).get_frame(t)

                # Apply the effect using fl
                clip = clip.fl(zoom_effect, apply_to=['mask', 'video'])
                clip = clip.set_audio(audio)

                clip.write_videofile(
                    output_file,
                    fps=30,
                    codec="libx264",
                    audio_codec="aac",
                    preset="fast",
                    logger=None
                )

                clip.close()
                audio.close()

                duration = time.perf_counter() - start_time
                durations.append(duration)

                memory_after = self._get_memory_usage()
                memory_samples.append(memory_after - memory_before)

                logger.debug(f"  Iteration {i+1}: {duration:.2f}s")
            except Exception as e:
                logger.error(f"Effects benchmark error: {e}")
                # Fallback to simple video without effects
                durations.append(0.0)

        result = BenchmarkResult(
            name="video_with_effects",
            duration_seconds=sum(durations),
            memory_mb=max(memory_samples) if memory_samples else 0,
            file_size_mb=self._get_file_size_mb(output_file),
            iterations=iterations,
            avg_duration=sum(durations) / len(durations) if durations else 0,
            min_duration=min(durations) if durations else 0,
            max_duration=max(durations) if durations else 0,
            metadata={"effects": ["ken_burns_zoom"]}
        )

        self.results.append(result)
        logger.info(f"Video with effects: avg={result.avg_duration:.2f}s")
        return result

    def benchmark_shorts_generation(self, iterations: int = 3) -> BenchmarkResult:
        """
        Benchmark YouTube Shorts generation.

        Tests 9:16 vertical video creation (1080x1920).
        """
        logger.info(f"Running Shorts generation benchmark ({iterations} iterations)")
        durations = []
        memory_samples = []
        output_file = ""

        audio_file = self._create_test_audio(30)  # 30 second short
        image_file = self._create_test_image(1080, 1920)  # Vertical

        for i in range(iterations):
            output_file = str(self.output_dir / f"short_{i}.mp4")
            memory_before = self._get_memory_usage()

            start_time = time.perf_counter()

            # FFmpeg command for shorts (vertical video)
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", image_file,
                "-i", audio_file,
                "-c:v", "libx264",
                "-preset", "faster",  # Faster for shorts
                "-tune", "stillimage",
                "-c:a", "aac",
                "-b:a", "192k",
                "-vf", "scale=1080:1920",
                "-shortest",
                "-pix_fmt", "yuv420p",
                output_file
            ]

            try:
                subprocess.run(cmd, capture_output=True, timeout=120)
                duration = time.perf_counter() - start_time
                durations.append(duration)

                memory_after = self._get_memory_usage()
                memory_samples.append(memory_after - memory_before)

                logger.debug(f"  Iteration {i+1}: {duration:.2f}s")
            except Exception as e:
                logger.error(f"Shorts benchmark error: {e}")
                durations.append(0.0)

        result = BenchmarkResult(
            name="shorts_generation",
            duration_seconds=sum(durations),
            memory_mb=max(memory_samples) if memory_samples else 0,
            file_size_mb=self._get_file_size_mb(output_file),
            iterations=iterations,
            avg_duration=sum(durations) / len(durations) if durations else 0,
            min_duration=min(durations) if durations else 0,
            max_duration=max(durations) if durations else 0,
            metadata={"resolution": "1080x1920", "target_duration": 30}
        )

        self.results.append(result)
        logger.info(f"Shorts generation: avg={result.avg_duration:.2f}s")
        return result

    def compare_methods(self) -> Dict[str, Any]:
        """
        Compare all video generation methods and return summary.

        Returns:
            Dictionary with comparison results.
        """
        logger.info("Comparing all video generation methods...")

        # Run all benchmarks if not already run
        if not self.results:
            self.benchmark_ffmpeg_direct()
            self.benchmark_moviepy()
            self.benchmark_with_effects()
            self.benchmark_shorts_generation()

        # Build comparison
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "total_benchmarks": len(self.results),
            "results": [r.to_dict() for r in self.results],
            "fastest_method": None,
            "slowest_method": None,
            "recommendations": []
        }

        # Find fastest/slowest
        valid_results = [r for r in self.results if r.avg_duration > 0]
        if valid_results:
            fastest = min(valid_results, key=lambda x: x.avg_duration)
            slowest = max(valid_results, key=lambda x: x.avg_duration)
            comparison["fastest_method"] = fastest.name
            comparison["slowest_method"] = slowest.name

            # Generate recommendations
            if fastest.name == "ffmpeg_direct":
                comparison["recommendations"].append(
                    "FFmpeg direct is fastest - use for simple videos without effects"
                )
            if any(r.name == "moviepy" and r.memory_mb > 500 for r in valid_results):
                comparison["recommendations"].append(
                    "MoviePy uses significant memory - consider FFmpeg for large batches"
                )

        logger.info(f"Comparison complete: fastest={comparison['fastest_method']}")
        return comparison

    def generate_report(self) -> str:
        """
        Generate markdown benchmark report.

        Returns:
            Markdown-formatted report string.
        """
        lines = [
            "# Video Generation Benchmark Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"Total benchmarks run: {len(self.results)}",
            "",
            "## Results",
            "",
            "| Method | Avg Duration | Min | Max | Memory | File Size |",
            "|--------|-------------|-----|-----|--------|-----------|",
        ]

        for r in self.results:
            lines.append(
                f"| {r.name} | {r.avg_duration:.2f}s | {r.min_duration:.2f}s | "
                f"{r.max_duration:.2f}s | {r.memory_mb:.1f}MB | {r.file_size_mb:.1f}MB |"
            )

        # Add comparison
        comparison = self.compare_methods()
        lines.extend([
            "",
            "## Analysis",
            "",
            f"**Fastest method:** {comparison.get('fastest_method', 'N/A')}",
            f"**Slowest method:** {comparison.get('slowest_method', 'N/A')}",
            "",
            "### Recommendations",
            "",
        ])

        for rec in comparison.get("recommendations", []):
            lines.append(f"- {rec}")

        return "\n".join(lines)

    def cleanup(self):
        """Remove benchmark output files."""
        import shutil
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir, ignore_errors=True)
            logger.info(f"Cleaned up benchmark output: {self.output_dir}")


class TTSBenchmark:
    """Benchmark TTS generation performance."""

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize TTS benchmark."""
        self.results: List[BenchmarkResult] = []
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"TTSBenchmark initialized, output dir: {self.output_dir}")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def _get_file_size_mb(self, filepath: str) -> float:
        """Get file size in MB."""
        if os.path.exists(filepath):
            return os.path.getsize(filepath) / 1024 / 1024
        return 0.0

    def benchmark_edge_tts(self, text_lengths: List[int] = None) -> List[BenchmarkResult]:
        """
        Benchmark Edge-TTS at various text lengths.

        Args:
            text_lengths: List of text character counts to test.
                         Default: [100, 500, 1000, 5000]

        Returns:
            List of BenchmarkResults for each text length.
        """
        if text_lengths is None:
            text_lengths = [100, 500, 1000, 5000]

        logger.info(f"Running Edge-TTS benchmark for lengths: {text_lengths}")
        results = []

        try:
            import edge_tts
        except ImportError:
            logger.error("edge-tts not installed, skipping benchmark")
            return []

        # Sample text for benchmarking
        base_text = "This is a sample text for benchmarking text to speech generation. " * 20

        async def generate_speech(text: str, output_file: str):
            """Generate speech using Edge-TTS."""
            communicate = edge_tts.Communicate(
                text=text,
                voice="en-US-GuyNeural"
            )
            await communicate.save(output_file)

        for length in text_lengths:
            test_text = base_text[:length]
            output_file = str(self.output_dir / f"edge_tts_{length}.mp3")
            durations = []
            memory_samples = []

            for i in range(3):  # 3 iterations
                memory_before = self._get_memory_usage()
                start_time = time.perf_counter()

                try:
                    asyncio.run(generate_speech(test_text, output_file))
                    duration = time.perf_counter() - start_time
                    durations.append(duration)

                    memory_after = self._get_memory_usage()
                    memory_samples.append(memory_after - memory_before)
                except Exception as e:
                    logger.error(f"Edge-TTS error for length {length}: {e}")
                    durations.append(0.0)

            result = BenchmarkResult(
                name=f"edge_tts_{length}chars",
                duration_seconds=sum(durations),
                memory_mb=max(memory_samples) if memory_samples else 0,
                file_size_mb=self._get_file_size_mb(output_file),
                iterations=3,
                avg_duration=sum(durations) / len(durations) if durations else 0,
                min_duration=min(durations) if durations else 0,
                max_duration=max(durations) if durations else 0,
                metadata={"text_length": length, "voice": "en-US-GuyNeural"}
            )
            results.append(result)
            self.results.append(result)
            logger.info(f"Edge-TTS ({length} chars): avg={result.avg_duration:.2f}s")

        return results

    def benchmark_fish_audio(self, text_lengths: List[int] = None) -> List[BenchmarkResult]:
        """
        Benchmark Fish Audio TTS.

        Args:
            text_lengths: List of text character counts to test.

        Returns:
            List of BenchmarkResults for each text length.
        """
        if text_lengths is None:
            text_lengths = [100, 500, 1000]

        logger.info(f"Running Fish Audio benchmark for lengths: {text_lengths}")
        results = []

        try:
            from src.content.tts_fish import FishAudioTTS
        except ImportError:
            logger.warning("Fish Audio TTS not available, skipping benchmark")
            return []

        # Check for API key
        api_key = os.getenv("FISH_AUDIO_API_KEY")
        if not api_key:
            logger.warning("FISH_AUDIO_API_KEY not set, skipping Fish Audio benchmark")
            return []

        base_text = "This is a sample text for benchmarking text to speech generation. " * 20

        for length in text_lengths:
            test_text = base_text[:length]
            output_file = str(self.output_dir / f"fish_audio_{length}.mp3")
            durations = []
            memory_samples = []

            tts = FishAudioTTS(api_key=api_key)

            for i in range(3):
                memory_before = self._get_memory_usage()
                start_time = time.perf_counter()

                try:
                    asyncio.run(tts.generate(test_text, output_file))
                    duration = time.perf_counter() - start_time
                    durations.append(duration)

                    memory_after = self._get_memory_usage()
                    memory_samples.append(memory_after - memory_before)
                except Exception as e:
                    logger.error(f"Fish Audio error for length {length}: {e}")
                    durations.append(0.0)

            result = BenchmarkResult(
                name=f"fish_audio_{length}chars",
                duration_seconds=sum(durations),
                memory_mb=max(memory_samples) if memory_samples else 0,
                file_size_mb=self._get_file_size_mb(output_file),
                iterations=3,
                avg_duration=sum(durations) / len(durations) if durations else 0,
                min_duration=min(durations) if durations else 0,
                max_duration=max(durations) if durations else 0,
                metadata={"text_length": length, "provider": "fish_audio"}
            )
            results.append(result)
            self.results.append(result)
            logger.info(f"Fish Audio ({length} chars): avg={result.avg_duration:.2f}s")

        return results

    def compare_providers(self) -> Dict[str, Any]:
        """Compare TTS providers."""
        edge_results = [r for r in self.results if r.name.startswith("edge_tts")]
        fish_results = [r for r in self.results if r.name.startswith("fish_audio")]

        comparison = {
            "edge_tts": {
                "avg_duration": sum(r.avg_duration for r in edge_results) / len(edge_results) if edge_results else 0,
                "count": len(edge_results),
            },
            "fish_audio": {
                "avg_duration": sum(r.avg_duration for r in fish_results) / len(fish_results) if fish_results else 0,
                "count": len(fish_results),
            },
            "recommendation": "edge_tts" if not fish_results or
                (edge_results and sum(r.avg_duration for r in edge_results) < sum(r.avg_duration for r in fish_results))
                else "fish_audio"
        }
        return comparison

    def cleanup(self):
        """Remove benchmark output files."""
        import shutil
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir, ignore_errors=True)


class PipelineBenchmark:
    """Benchmark full pipeline performance."""

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize pipeline benchmark."""
        self.results: List[BenchmarkResult] = []
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PipelineBenchmark initialized, output dir: {self.output_dir}")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def _get_file_size_mb(self, filepath: str) -> float:
        """Get file size in MB."""
        if os.path.exists(filepath):
            return os.path.getsize(filepath) / 1024 / 1024
        return 0.0

    def benchmark_full_video_pipeline(self) -> BenchmarkResult:
        """
        Benchmark complete video creation pipeline.

        Tests: TTS generation -> Audio processing -> Video assembly
        """
        logger.info("Running full video pipeline benchmark")

        durations = []
        memory_samples = []
        output_file = str(self.output_dir / "pipeline_video.mp4")

        for i in range(2):  # 2 iterations due to longer duration
            memory_before = self._get_memory_usage()
            start_time = time.perf_counter()

            try:
                # Step 1: Generate TTS
                tts_file = str(self.output_dir / f"pipeline_tts_{i}.mp3")
                test_text = "Welcome to this tutorial. Today we will learn about Python programming. " * 5

                try:
                    import edge_tts

                    async def gen_tts():
                        comm = edge_tts.Communicate(test_text, "en-US-GuyNeural")
                        await comm.save(tts_file)

                    asyncio.run(gen_tts())
                except ImportError:
                    # Create dummy audio
                    subprocess.run([
                        "ffmpeg", "-y", "-f", "lavfi",
                        "-i", "sine=frequency=440:duration=10",
                        "-c:a", "libmp3lame", tts_file
                    ], capture_output=True)

                # Step 2: Create video
                try:
                    from src.content.video_assembler import VideoAssembler
                    assembler = VideoAssembler(resolution=(1920, 1080))
                    assembler.create_video_from_audio(
                        audio_file=tts_file,
                        output_file=output_file,
                        title="Benchmark Test"
                    )
                except ImportError:
                    # Fallback to FFmpeg direct
                    subprocess.run([
                        "ffmpeg", "-y",
                        "-f", "lavfi", "-i", "color=c=blue:s=1920x1080:d=10",
                        "-i", tts_file,
                        "-c:v", "libx264", "-preset", "fast",
                        "-c:a", "aac", "-shortest",
                        output_file
                    ], capture_output=True)

                duration = time.perf_counter() - start_time
                durations.append(duration)

                memory_after = self._get_memory_usage()
                memory_samples.append(memory_after - memory_before)

                logger.debug(f"  Pipeline iteration {i+1}: {duration:.2f}s")

            except Exception as e:
                logger.error(f"Pipeline benchmark error: {e}")
                durations.append(0.0)

        result = BenchmarkResult(
            name="full_video_pipeline",
            duration_seconds=sum(durations),
            memory_mb=max(memory_samples) if memory_samples else 0,
            file_size_mb=self._get_file_size_mb(output_file),
            iterations=2,
            avg_duration=sum(durations) / len(durations) if durations else 0,
            min_duration=min(durations) if durations else 0,
            max_duration=max(durations) if durations else 0,
            metadata={"steps": ["tts", "video_assembly"]}
        )

        self.results.append(result)
        logger.info(f"Full pipeline: avg={result.avg_duration:.2f}s")
        return result

    def benchmark_shorts_pipeline(self) -> BenchmarkResult:
        """
        Benchmark complete shorts pipeline.

        Tests vertical video generation for YouTube Shorts.
        """
        logger.info("Running shorts pipeline benchmark")

        durations = []
        memory_samples = []
        output_file = str(self.output_dir / "pipeline_short.mp4")

        for i in range(2):
            memory_before = self._get_memory_usage()
            start_time = time.perf_counter()

            try:
                # Generate short TTS (30 seconds max)
                tts_file = str(self.output_dir / f"shorts_tts_{i}.mp3")
                test_text = "Did you know that Python is one of the most popular programming languages? Let me show you why in just 30 seconds!"

                try:
                    import edge_tts

                    async def gen_tts():
                        comm = edge_tts.Communicate(test_text, "en-US-GuyNeural", rate="+10%")
                        await comm.save(tts_file)

                    asyncio.run(gen_tts())
                except ImportError:
                    subprocess.run([
                        "ffmpeg", "-y", "-f", "lavfi",
                        "-i", "sine=frequency=440:duration=30",
                        "-c:a", "libmp3lame", tts_file
                    ], capture_output=True)

                # Create vertical video
                subprocess.run([
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "color=c=darkblue:s=1080x1920:d=30",
                    "-i", tts_file,
                    "-c:v", "libx264", "-preset", "faster",
                    "-c:a", "aac", "-shortest",
                    output_file
                ], capture_output=True, timeout=120)

                duration = time.perf_counter() - start_time
                durations.append(duration)

                memory_after = self._get_memory_usage()
                memory_samples.append(memory_after - memory_before)

            except Exception as e:
                logger.error(f"Shorts pipeline error: {e}")
                durations.append(0.0)

        result = BenchmarkResult(
            name="shorts_pipeline",
            duration_seconds=sum(durations),
            memory_mb=max(memory_samples) if memory_samples else 0,
            file_size_mb=self._get_file_size_mb(output_file),
            iterations=2,
            avg_duration=sum(durations) / len(durations) if durations else 0,
            min_duration=min(durations) if durations else 0,
            max_duration=max(durations) if durations else 0,
            metadata={"resolution": "1080x1920", "target_duration": 30}
        )

        self.results.append(result)
        logger.info(f"Shorts pipeline: avg={result.avg_duration:.2f}s")
        return result

    def cleanup(self):
        """Remove benchmark output files."""
        import shutil
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir, ignore_errors=True)


def run_all_benchmarks():
    """Run all benchmarks and save report."""
    logger.info("=" * 60)
    logger.info("  RUNNING ALL PERFORMANCE BENCHMARKS")
    logger.info("=" * 60)

    output_dir = Path("output/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Video benchmarks
    logger.info("\n--- Video Generation Benchmarks ---")
    video_bench = VideoBenchmark(str(output_dir / "video"))
    try:
        video_bench.benchmark_ffmpeg_direct(iterations=2)
        video_bench.benchmark_moviepy(iterations=2)
        video_bench.benchmark_shorts_generation(iterations=2)
        all_results.extend(video_bench.results)
    except Exception as e:
        logger.error(f"Video benchmark failed: {e}")

    # TTS benchmarks
    logger.info("\n--- TTS Benchmarks ---")
    tts_bench = TTSBenchmark(str(output_dir / "tts"))
    try:
        tts_bench.benchmark_edge_tts([100, 500, 1000])
        all_results.extend(tts_bench.results)
    except Exception as e:
        logger.error(f"TTS benchmark failed: {e}")

    # Pipeline benchmarks
    logger.info("\n--- Pipeline Benchmarks ---")
    pipeline_bench = PipelineBenchmark(str(output_dir / "pipeline"))
    try:
        pipeline_bench.benchmark_full_video_pipeline()
        pipeline_bench.benchmark_shorts_pipeline()
        all_results.extend(pipeline_bench.results)
    except Exception as e:
        logger.error(f"Pipeline benchmark failed: {e}")

    # Generate and save report
    report = video_bench.generate_report()

    # Add TTS section to report
    report += "\n\n## TTS Benchmarks\n\n"
    report += "| Provider | Text Length | Avg Duration | File Size |\n"
    report += "|----------|-------------|--------------|----------|\n"
    for r in tts_bench.results:
        report += f"| {r.name} | {r.metadata.get('text_length', 'N/A')} | {r.avg_duration:.2f}s | {r.file_size_mb:.2f}MB |\n"

    # Add pipeline section
    report += "\n\n## Pipeline Benchmarks\n\n"
    report += "| Pipeline | Avg Duration | Memory | File Size |\n"
    report += "|----------|--------------|--------|----------|\n"
    for r in pipeline_bench.results:
        report += f"| {r.name} | {r.avg_duration:.2f}s | {r.memory_mb:.1f}MB | {r.file_size_mb:.1f}MB |\n"

    # Save report
    report_file = output_dir / "benchmark_report.md"
    with open(report_file, "w") as f:
        f.write(report)

    logger.success(f"Benchmark report saved to: {report_file}")
    print(report)

    # Cleanup
    logger.info("\nCleaning up benchmark files...")
    video_bench.cleanup()
    tts_bench.cleanup()
    pipeline_bench.cleanup()

    logger.info("=" * 60)
    logger.info("  ALL BENCHMARKS COMPLETE")
    logger.info("=" * 60)

    return all_results


if __name__ == "__main__":
    run_all_benchmarks()
