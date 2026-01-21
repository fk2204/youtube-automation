"""
GPU Acceleration Utilities

Detects and configures GPU acceleration for FFmpeg video encoding.
Supports NVIDIA (NVENC), AMD (AMF), and Intel (Quick Sync) hardware acceleration.

Usage:
    from src.utils.gpu_utils import GPUAccelerator

    accelerator = GPUAccelerator()
    if accelerator.is_available():
        encoder = accelerator.get_encoder()
        ffmpeg_args = accelerator.get_ffmpeg_args(preset='fast')
"""

import os
import subprocess
import platform
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class GPUType(Enum):
    """Supported GPU types for hardware acceleration."""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    CPU = "cpu"  # Fallback


@dataclass
class GPUInfo:
    """GPU information and capabilities."""
    gpu_type: GPUType
    name: str
    available: bool
    encoder: str  # FFmpeg encoder name (h264_nvenc, h264_amf, h264_qsv, libx264)
    decoder: str  # FFmpeg decoder (h264_cuvid, etc.)
    scale_filter: str  # Scaling filter (scale_cuda, scale_vaapi, scale)
    supports_hevc: bool = False
    max_encode_resolution: Tuple[int, int] = (3840, 2160)  # 4K default


class GPUAccelerator:
    """
    Detect and configure GPU acceleration for video encoding.

    Provides 2-3x faster encoding with NVIDIA/AMD/Intel GPUs.
    Automatically falls back to CPU if no GPU detected.
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg", prefer_quality: bool = False):
        """
        Initialize GPU accelerator.

        Args:
            ffmpeg_path: Path to FFmpeg executable
            prefer_quality: If True, use quality-optimized settings (slower but better)
        """
        self.ffmpeg_path = ffmpeg_path
        self.prefer_quality = prefer_quality
        self._gpu_info: Optional[GPUInfo] = None
        self._detect_gpu()

    def _run_command(self, cmd: List[str], timeout: int = 10) -> Tuple[bool, str]:
        """Run a command and return success status and output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)

    def _detect_gpu(self) -> None:
        """Detect available GPU and set capabilities."""
        logger.info("Detecting GPU capabilities...")

        # Check FFmpeg encoders
        success, output = self._run_command([self.ffmpeg_path, "-encoders"])
        if not success:
            logger.warning("Could not detect FFmpeg encoders")
            self._gpu_info = self._cpu_fallback()
            return

        encoders_available = output.lower()

        # NVIDIA Check (NVENC)
        if "h264_nvenc" in encoders_available or "nvenc" in encoders_available:
            gpu_name = self._get_nvidia_gpu_name()
            if gpu_name:
                self._gpu_info = GPUInfo(
                    gpu_type=GPUType.NVIDIA,
                    name=gpu_name,
                    available=True,
                    encoder="h264_nvenc",
                    decoder="h264_cuvid",
                    scale_filter="scale_cuda",
                    supports_hevc="hevc_nvenc" in encoders_available,
                    max_encode_resolution=(7680, 4320)  # 8K on modern NVIDIA
                )
                logger.success(f"NVIDIA GPU detected: {gpu_name}")
                return

        # AMD Check (AMF)
        if "h264_amf" in encoders_available or "amf" in encoders_available:
            gpu_name = self._get_amd_gpu_name()
            if gpu_name:
                self._gpu_info = GPUInfo(
                    gpu_type=GPUType.AMD,
                    name=gpu_name,
                    available=True,
                    encoder="h264_amf",
                    decoder="h264",  # AMD doesn't have dedicated decoder
                    scale_filter="scale",
                    supports_hevc="hevc_amf" in encoders_available,
                    max_encode_resolution=(3840, 2160)  # 4K
                )
                logger.success(f"AMD GPU detected: {gpu_name}")
                return

        # Intel Check (Quick Sync)
        if "h264_qsv" in encoders_available or "qsv" in encoders_available:
            gpu_name = self._get_intel_gpu_name()
            if gpu_name:
                self._gpu_info = GPUInfo(
                    gpu_type=GPUType.INTEL,
                    name=gpu_name,
                    available=True,
                    encoder="h264_qsv",
                    decoder="h264_qsv",
                    scale_filter="scale_qsv",
                    supports_hevc="hevc_qsv" in encoders_available,
                    max_encode_resolution=(4096, 2304)  # 4K+
                )
                logger.success(f"Intel GPU detected: {gpu_name}")
                return

        # Fallback to CPU
        self._gpu_info = self._cpu_fallback()
        logger.info("No GPU acceleration available, using CPU encoding")

    def _get_nvidia_gpu_name(self) -> Optional[str]:
        """Get NVIDIA GPU name using nvidia-smi."""
        try:
            success, output = self._run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
            if success and output.strip():
                return output.strip().split('\n')[0]
        except:
            pass

        # Fallback: check Windows registry or Linux sysfs
        if platform.system() == "Windows":
            return self._get_gpu_from_wmic("NVIDIA")

        return "NVIDIA GPU"

    def _get_amd_gpu_name(self) -> Optional[str]:
        """Get AMD GPU name."""
        if platform.system() == "Windows":
            return self._get_gpu_from_wmic("AMD")
        return "AMD GPU"

    def _get_intel_gpu_name(self) -> Optional[str]:
        """Get Intel GPU name."""
        if platform.system() == "Windows":
            return self._get_gpu_from_wmic("Intel")
        return "Intel GPU"

    def _get_gpu_from_wmic(self, vendor: str) -> Optional[str]:
        """Get GPU name from Windows WMIC."""
        try:
            success, output = self._run_command(
                ["wmic", "path", "win32_VideoController", "get", "name"]
            )
            if success:
                for line in output.split('\n'):
                    line = line.strip()
                    if vendor.lower() in line.lower() and line != "Name":
                        return line
        except:
            pass
        return None

    def _cpu_fallback(self) -> GPUInfo:
        """Return CPU encoding configuration."""
        return GPUInfo(
            gpu_type=GPUType.CPU,
            name="CPU (Software Encoding)",
            available=True,
            encoder="libx264",
            decoder="h264",
            scale_filter="scale",
            supports_hevc=False,
            max_encode_resolution=(7680, 4320)
        )

    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self._gpu_info is not None and self._gpu_info.gpu_type != GPUType.CPU

    def get_gpu_type(self) -> GPUType:
        """Get detected GPU type."""
        return self._gpu_info.gpu_type if self._gpu_info else GPUType.CPU

    def get_encoder(self, codec: str = "h264") -> str:
        """
        Get the best encoder for the detected GPU.

        Args:
            codec: Codec to use (h264 or hevc)

        Returns:
            FFmpeg encoder name
        """
        if not self._gpu_info:
            return "libx264"

        if codec == "hevc" and self._gpu_info.supports_hevc:
            encoder_map = {
                GPUType.NVIDIA: "hevc_nvenc",
                GPUType.AMD: "hevc_amf",
                GPUType.INTEL: "hevc_qsv",
                GPUType.CPU: "libx265"
            }
            return encoder_map[self._gpu_info.gpu_type]

        return self._gpu_info.encoder

    def get_decoder(self) -> str:
        """Get hardware decoder if available."""
        return self._gpu_info.decoder if self._gpu_info else "h264"

    def get_scale_filter(self) -> str:
        """Get GPU-accelerated scaling filter."""
        return self._gpu_info.scale_filter if self._gpu_info else "scale"

    def get_ffmpeg_args(
        self,
        preset: str = "fast",
        bitrate: str = "8M",
        quality: Optional[int] = None,
        width: int = 1920,
        height: int = 1080
    ) -> List[str]:
        """
        Get optimized FFmpeg arguments for the detected GPU.

        Args:
            preset: Encoding preset (ultrafast, fast, medium, slow)
            bitrate: Target bitrate (e.g., "8M", "5M")
            quality: Quality level (CRF for CPU, CQ for GPU). None = use bitrate
            width: Output width
            height: Output height

        Returns:
            List of FFmpeg arguments
        """
        if not self._gpu_info:
            return self._get_cpu_args(preset, bitrate, quality)

        gpu_type = self._gpu_info.gpu_type

        if gpu_type == GPUType.NVIDIA:
            return self._get_nvidia_args(preset, bitrate, quality)
        elif gpu_type == GPUType.AMD:
            return self._get_amd_args(preset, bitrate, quality)
        elif gpu_type == GPUType.INTEL:
            return self._get_intel_args(preset, bitrate, quality)
        else:
            return self._get_cpu_args(preset, bitrate, quality)

    def _get_nvidia_args(self, preset: str, bitrate: str, quality: Optional[int]) -> List[str]:
        """Get NVIDIA NVENC encoding arguments."""
        args = [
            "-c:v", "h264_nvenc",
            "-preset", preset,  # p1-p7 presets, or fast/medium/slow
            "-b:v", bitrate,
        ]

        # GPU has different quality scale: 0 (best) to 51 (worst)
        if quality is not None:
            args.extend(["-cq", str(quality)])

        # Quality optimizations
        if self.prefer_quality:
            args.extend([
                "-rc", "vbr_hq",  # High-quality VBR
                "-spatial_aq", "1",  # Spatial AQ
                "-temporal_aq", "1",  # Temporal AQ
                "-rc-lookahead", "20"  # Lookahead frames
            ])
        else:
            args.extend(["-rc", "vbr"])  # Standard VBR

        args.extend(["-pix_fmt", "yuv420p"])

        return args

    def _get_amd_args(self, preset: str, bitrate: str, quality: Optional[int]) -> List[str]:
        """Get AMD AMF encoding arguments."""
        args = [
            "-c:v", "h264_amf",
            "-quality", preset if preset in ["balanced", "speed", "quality"] else "balanced",
            "-b:v", bitrate,
        ]

        if quality is not None:
            args.extend(["-qp_i", str(quality), "-qp_p", str(quality)])

        args.extend(["-pix_fmt", "yuv420p"])

        return args

    def _get_intel_args(self, preset: str, bitrate: str, quality: Optional[int]) -> List[str]:
        """Get Intel Quick Sync encoding arguments."""
        # Map preset to Intel preset names
        preset_map = {
            "ultrafast": "veryfast",
            "fast": "fast",
            "medium": "medium",
            "slow": "slow"
        }

        args = [
            "-c:v", "h264_qsv",
            "-preset", preset_map.get(preset, "medium"),
            "-b:v", bitrate,
        ]

        if quality is not None:
            args.extend(["-global_quality", str(quality)])

        args.extend(["-pix_fmt", "yuv420p"])

        return args

    def _get_cpu_args(self, preset: str, bitrate: str, quality: Optional[int]) -> List[str]:
        """Get CPU (libx264) encoding arguments."""
        args = [
            "-c:v", "libx264",
            "-preset", preset,
            "-b:v", bitrate,
        ]

        if quality is not None:
            args.extend(["-crf", str(quality)])
        else:
            args.extend(["-crf", "23"])  # Default quality

        args.extend(["-pix_fmt", "yuv420p"])

        return args

    def get_input_args(self, use_hwaccel: bool = True) -> List[str]:
        """
        Get FFmpeg input arguments for hardware-accelerated decoding.

        Args:
            use_hwaccel: Whether to use hardware decoding

        Returns:
            List of FFmpeg input arguments (place before -i)
        """
        if not use_hwaccel or not self._gpu_info or self._gpu_info.gpu_type == GPUType.CPU:
            return []

        gpu_type = self._gpu_info.gpu_type

        if gpu_type == GPUType.NVIDIA:
            return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        elif gpu_type == GPUType.AMD:
            return []  # AMD AMF doesn't require special input args
        elif gpu_type == GPUType.INTEL:
            return ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]

        return []

    def get_status(self) -> Dict:
        """Get GPU status information."""
        if not self._gpu_info:
            return {
                "available": False,
                "type": "cpu",
                "name": "CPU (Software Encoding)",
                "encoder": "libx264",
                "speedup": "1x (baseline)"
            }

        speedup_map = {
            GPUType.NVIDIA: "2-3x faster",
            GPUType.AMD: "2-3x faster",
            GPUType.INTEL: "1.5-2x faster",
            GPUType.CPU: "1x (baseline)"
        }

        return {
            "available": self._gpu_info.gpu_type != GPUType.CPU,
            "type": self._gpu_info.gpu_type.value,
            "name": self._gpu_info.name,
            "encoder": self._gpu_info.encoder,
            "decoder": self._gpu_info.decoder,
            "scale_filter": self._gpu_info.scale_filter,
            "supports_hevc": self._gpu_info.supports_hevc,
            "max_resolution": f"{self._gpu_info.max_encode_resolution[0]}x{self._gpu_info.max_encode_resolution[1]}",
            "speedup": speedup_map[self._gpu_info.gpu_type]
        }

    def print_status(self) -> None:
        """Print GPU status to console."""
        status = self.get_status()

        print("\n" + "="*60)
        print("  GPU ACCELERATION STATUS")
        print("="*60)
        print(f"GPU Available:     {'YES' if status['available'] else 'NO'}")
        print(f"Type:              {status['type'].upper()}")
        print(f"Name:              {status['name']}")
        print(f"Encoder:           {status['encoder']}")
        if status['available']:
            print(f"Decoder:           {status['decoder']}")
            print(f"Scale Filter:      {status['scale_filter']}")
            print(f"HEVC Support:      {'YES' if status['supports_hevc'] else 'NO'}")
            print(f"Max Resolution:    {status['max_resolution']}")
        print(f"Expected Speedup:  {status['speedup']}")
        print("="*60 + "\n")


# Singleton instance
_gpu_accelerator: Optional[GPUAccelerator] = None


def get_gpu_accelerator(ffmpeg_path: str = "ffmpeg", prefer_quality: bool = False) -> GPUAccelerator:
    """Get or create GPU accelerator singleton."""
    global _gpu_accelerator
    if _gpu_accelerator is None:
        _gpu_accelerator = GPUAccelerator(ffmpeg_path, prefer_quality)
    return _gpu_accelerator


# Example usage
if __name__ == "__main__":
    accelerator = GPUAccelerator()
    accelerator.print_status()

    if accelerator.is_available():
        print("\nOptimized FFmpeg encoding arguments:")
        print(" ".join(accelerator.get_ffmpeg_args(preset="fast", bitrate="8M")))
        print("\nInput arguments for hardware decoding:")
        print(" ".join(accelerator.get_input_args()))
