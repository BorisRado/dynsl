import threading

## ONLY USE ON JNs !!!

_GPU_MAXFREQ = "/sys/devices/gpu.0/devfreq/57000000.gpu/max_freq"
_GPU_FULL = 921600000   # MAXN ceiling, from your jetson_clocks --show
_GPU_CAP  = 230400000   # ~1/4 of full; tune to the slowdown magnitude you want


def _write_gpu_maxfreq(hz):
    """Set the GPU DVFS ceiling. Requires privileged pod + root."""
    with open(_GPU_MAXFREQ, "w") as f:
        f.write(str(hz))


def throttle_gpu(delay_s=5.0):
    """
    Cap the GPU clock `delay_s` seconds from now, so the slowdown
    lands mid-training (mirrors the tc trigger in the bandwidth exp).
    Returns the Timer so the caller can cancel it if needed.
    """
    t = threading.Timer(delay_s, lambda: _write_gpu_maxfreq(_GPU_CAP))
    t.daemon = True
    t.start()
    return t


def restore_gpu():
    """Best-effort restore to full clock. Safe to call unconditionally."""
    try:
        _write_gpu_maxfreq(_GPU_FULL)
    except Exception as e:
        print(f"[WARN] failed to restore GPU clock: {e}")
