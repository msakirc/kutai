from nerd_herd.load import LoadManager


class _FakeGPU:
    def __init__(self, frac): self._frac = frac
    def gpu_state(self):
        from nerd_herd.types import GPUState
        return GPUState(available=True, vram_total_mb=8000, vram_free_mb=8000)
    def detect_external_gpu_usage(self):
        from nerd_herd.types import ExternalGPUUsage
        return ExternalGPUUsage(detected=True, external_vram_mb=int(8000*self._frac),
                                total_vram_mb=8000)


def test_external_fraction_defaults_zero_before_first_detect():
    lm = LoadManager(gpu_collector=_FakeGPU(0.4))
    assert lm.get_external_gpu_fraction() == 0.0


def test_record_external_updates_cache():
    lm = LoadManager(gpu_collector=_FakeGPU(0.4))
    lm._record_external(_FakeGPU(0.4).detect_external_gpu_usage())
    assert abs(lm.get_external_gpu_fraction() - 0.4) < 1e-6
