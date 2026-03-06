"""
GPU Configuration Module for Data Analysis Toolkit

Detects GPU compute capability vs TensorFlow's supported capabilities.
Automatically falls back to CPU when the GPU architecture is unsupported
(e.g. NVIDIA RTX 5070 Ti / Blackwell with compute capability 12.0 on
TensorFlow builds that only support up to compute 9.0).

Blackwell (RTX 50-series) GPU support status
---------------------------------------------
- TensorFlow 2.17.0 inside NVIDIA Optimized TensorFlow containers
  (nvcr.io/nvidia/tensorflow:25.01-tf2-py3 or 25.02) fully supports
  Blackwell, including FP4/FP6/FP8 via 5th-gen Tensor Cores.
- Pre-built pip packages for TensorFlow 2.19 / 2.20 do NOT include
  compute_120 kernels and will fail with CUDA_ERROR_INVALID_HANDLE.
- A custom source build of TF 2.19+ with CUDA 12.8.1 and Clang can
  enable Blackwell support (see GitHub forks).
- Future TF releases (2.20+) are expected to add native Blackwell
  support in the official pip packages.

Environment variable overrides
-------------------------------
DATA_TOOLKIT_FORCE_GPU=1   — skip the compute-capability check and
                             attempt to use the GPU anyway (useful if
                             you have a patched / container TF build).
DATA_TOOLKIT_FORCE_CPU=1   — always use CPU, even if a compatible GPU
                             is detected.

Usage:
    # Call once before any TF model operations:
    from data_toolkit.gpu_config import configure_gpu
    configure_gpu()

    # Or import at module level (auto-configures on first call):
    from data_toolkit.gpu_config import ensure_gpu_ready
    ensure_gpu_ready()
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Optional

# Suppress TensorFlow v1 deprecation warnings from Keras internals
# (e.g., tf.reset_default_graph being deprecated in keras/backend/global_state.py)
warnings.filterwarnings('ignore', message='.*reset_default_graph.*', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*reset_default_graph.*')

logger = logging.getLogger(__name__)

# Module-level state
_configured = False
_gpu_usable = False
_fallback_reason: Optional[str] = None


def configure_gpu(force_cpu: bool = False, force_gpu: bool = False,
                  verbose: bool = True, reconfigure: bool = False) -> dict:
    """
    Configure TensorFlow GPU usage with automatic compatibility detection.

    This function:
    1. Checks if a GPU is available
    2. Verifies that TF was built with kernels compatible with the GPU's
       compute capability
    3. Runs a small smoke test (matmul + cast) on the GPU
    4. Falls back to CPU if any step fails

    The behaviour can be overridden via environment variables:
        DATA_TOOLKIT_FORCE_GPU=1  — skip compute-capability check, try GPU
        DATA_TOOLKIT_FORCE_CPU=1  — always use CPU

    Args:
        force_cpu: If True, skip GPU detection and use CPU only.
        force_gpu: If True, skip the compute-capability check and attempt
                   to use the GPU (runs a smoke test first).
        verbose: If True, print status messages.
        reconfigure: If True, run even if already configured.

    Returns:
        dict with keys:
            - 'device': 'GPU' or 'CPU'
            - 'gpu_name': GPU name or None
            - 'compute_capability': tuple or None
            - 'supported_capabilities': list of str
            - 'fallback_reason': str or None
    """
    global _configured, _gpu_usable, _fallback_reason

    # Skip if already configured (unless explicitly asked to reconfigure)
    if _configured and not reconfigure:
        return {
            'device': 'GPU' if _gpu_usable else 'CPU',
            'gpu_name': None,
            'compute_capability': None,
            'supported_capabilities': [],
            'fallback_reason': _fallback_reason,
        }

    # --- Environment variable overrides ---
    if os.environ.get('DATA_TOOLKIT_FORCE_CPU', '').strip() == '1':
        force_cpu = True
    if os.environ.get('DATA_TOOLKIT_FORCE_GPU', '').strip() == '1':
        force_gpu = True

    try:
        import tensorflow as tf
    except ImportError:
        _configured = True
        _gpu_usable = False
        _fallback_reason = 'TensorFlow not installed'
        return {
            'device': 'CPU',
            'gpu_name': None,
            'compute_capability': None,
            'supported_capabilities': [],
            'fallback_reason': _fallback_reason,
        }

    result = {
        'device': 'CPU',
        'gpu_name': None,
        'compute_capability': None,
        'supported_capabilities': [],
        'fallback_reason': None,
    }

    if force_cpu:
        _force_cpu(tf, verbose)
        result['fallback_reason'] = 'Forced CPU mode'
        _configured = True
        _gpu_usable = False
        _fallback_reason = result['fallback_reason']
        return result

    # --- Step 1: Check for physical GPUs ---
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        if verbose:
            print("ℹ️  No GPU detected — using CPU.")
        result['fallback_reason'] = 'No GPU detected'
        _configured = True
        _gpu_usable = False
        _fallback_reason = result['fallback_reason']
        return result

    gpu = gpus[0]
    result['gpu_name'] = gpu.name

    # --- Step 2: Get GPU compute capability ---
    try:
        details = tf.config.experimental.get_device_details(gpu)
        cc = details.get('compute_capability')
        if cc:
            result['compute_capability'] = cc
    except Exception:
        pass

    # --- Step 3: Check TF build info for supported compute capabilities ---
    try:
        build_info = tf.sysconfig.get_build_info()
        supported = build_info.get('cuda_compute_capabilities', [])
        result['supported_capabilities'] = supported
    except Exception:
        supported = []

    # --- Step 4: Check compatibility ---
    cc = result['compute_capability']
    incompatible = False
    is_blackwell = cc and cc[0] >= 12  # Blackwell = compute 12.0+

    if cc and supported:
        # TF lists capabilities like 'sm_60', 'sm_70', 'compute_90'
        # GPU cc is a tuple like (12, 0) meaning compute capability 12.0
        gpu_major = cc[0]
        max_supported_major = 0
        for cap in supported:
            cap_str = cap.replace('sm_', '').replace('compute_', '')
            try:
                major = int(cap_str[0]) if len(cap_str) <= 2 else int(cap_str[:2])
                max_supported_major = max(max_supported_major, major)
            except (ValueError, IndexError):
                pass

        if gpu_major > max_supported_major:
            if force_gpu:
                # User explicitly wants to try the GPU (e.g. NVIDIA container
                # or custom TF build).  Skip the capability check but still
                # run the smoke test below.
                if verbose:
                    print(
                        f"ℹ️  GPU compute capability {cc[0]}.{cc[1]} exceeds "
                        f"TF build max ({max_supported_major}.x), but "
                        f"DATA_TOOLKIT_FORCE_GPU is set — attempting GPU anyway."
                    )
            else:
                incompatible = True
                reason = (
                    f"GPU compute capability {cc[0]}.{cc[1]} "
                    f"({'Blackwell/RTX 50xx' if is_blackwell else 'newer arch'}) "
                    f"is not natively supported by this TensorFlow "
                    f"{tf.__version__} build "
                    f"(max supported: {max_supported_major}.x). "
                    f"JIT compilation from PTX may fail with "
                    f"CUDA_ERROR_INVALID_HANDLE."
                )
                if verbose:
                    print(f"⚠️  {reason}")
                    print("⚠️  Falling back to CPU to avoid CUDA errors.")
                    if is_blackwell:
                        print(
                            "\n"
                            "💡 Blackwell GPU workarounds:\n"
                            "   1. Use NVIDIA Optimized TF containers "
                            "(nvcr.io/nvidia/tensorflow:25.02-tf2-py3)\n"
                            "      which include native Blackwell support "
                            "(FP4/FP6/FP8).\n"
                            "   2. Build TF from source with CUDA ≥12.8.1 "
                            "and Clang.\n"
                            "   3. Set DATA_TOOLKIT_FORCE_GPU=1 to skip this "
                            "check if you\n"
                            "      have a compatible TF build.\n"
                            "   4. Wait for official TF pip packages with "
                            "Blackwell support.\n"
                        )
                result['fallback_reason'] = reason

    # --- Step 5: Enable memory growth (prevents OOM) and smoke test ---
    # Run the smoke test when: (a) the GPU looks compatible, OR
    # (b) force_gpu is set (user has a patched/container build).
    if not incompatible:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass  # Already set or GPUs already initialized

        # Smoke test: run a small operation on the GPU
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
                c = tf.matmul(a, b)
                # Also test Cast (the op that fails with Blackwell)
                d = tf.cast(c, tf.int32)
                _ = d.numpy()
            result['device'] = 'GPU'
            if verbose:
                cc_str = f"{cc[0]}.{cc[1]}" if cc else 'unknown'
                print(f"✅ GPU smoke test passed — using GPU (CC {cc_str}).")
        except Exception as e:
            reason = f"GPU smoke test failed: {e}"
            if verbose:
                print(f"⚠️  {reason}")
                print("⚠️  Falling back to CPU.")
                if is_blackwell:
                    print(
                        "💡 The GPU smoke test failed even with force_gpu. "
                        "Your TF build likely lacks Blackwell kernel support.\n"
                        "   Try the NVIDIA Optimized TF container "
                        "(nvcr.io/nvidia/tensorflow:25.02-tf2-py3)."
                    )
            result['fallback_reason'] = reason
            incompatible = True

    # --- Step 6: Fall back to CPU if needed ---
    if incompatible:
        _force_cpu(tf, verbose)
        result['device'] = 'CPU'

    _configured = True
    _gpu_usable = (result['device'] == 'GPU')
    _fallback_reason = result['fallback_reason']
    return result


def _force_cpu(tf, verbose: bool = True):
    """Hide all GPUs from TensorFlow so it only uses CPU."""
    try:
        tf.config.set_visible_devices([], 'GPU')
        if verbose:
            print("ℹ️  TensorFlow configured to use CPU only.")
    except RuntimeError as e:
        # If devices are already initialized, we can't change visibility.
        # Set the environment variable as a fallback for child processes.
        if verbose:
            print(f"ℹ️  Could not hide GPU devices (already initialized): {e}")
            print("ℹ️  Setting CUDA_VISIBLE_DEVICES='' for this process.")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''


def ensure_gpu_ready() -> dict:
    """
    Ensure GPU configuration has been run. Idempotent — only runs once.

    Returns the same dict as configure_gpu().
    """
    global _configured
    if not _configured:
        return configure_gpu()
    return {
        'device': 'GPU' if _gpu_usable else 'CPU',
        'gpu_name': None,
        'compute_capability': None,
        'supported_capabilities': [],
        'fallback_reason': _fallback_reason,
    }


def is_gpu_usable() -> bool:
    """Return True if GPU passed compatibility checks."""
    ensure_gpu_ready()
    return _gpu_usable


def get_device_summary() -> str:
    """Return a human-readable summary of the GPU/CPU configuration."""
    info = ensure_gpu_ready()
    lines = [f"Device: {info['device']}"]
    if info['gpu_name']:
        lines.append(f"GPU: {info['gpu_name']}")
    if info['compute_capability']:
        cc = info['compute_capability']
        lines.append(f"Compute capability: {cc[0]}.{cc[1]}")
    if info['fallback_reason']:
        lines.append(f"Fallback reason: {info['fallback_reason']}")
    return ' | '.join(lines)
