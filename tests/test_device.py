import bettertranscribe.device as device


def test_explicit_cpu():
    b = device.resolve_backend("cpu")
    assert (b.name, b.device, b.compute_type) == ("faster", "cpu", "int8")


def test_explicit_cuda():
    b = device.resolve_backend("cuda")
    assert (b.name, b.device) == ("faster", "cuda")


def test_auto_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(device, "_cuda_available", lambda: True)
    b = device.resolve_backend("auto")
    assert b.device == "cuda" and b.name == "faster"


def test_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(device, "_cuda_available", lambda: False)
    monkeypatch.setattr(device, "_is_apple_silicon", lambda: False)
    b = device.resolve_backend("auto")
    assert (b.name, b.device) == ("faster", "cpu")


def test_auto_uses_mlx_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(device, "_cuda_available", lambda: False)
    monkeypatch.setattr(device, "_is_apple_silicon", lambda: True)
    monkeypatch.setattr(device, "_mlx_available", lambda: True)
    b = device.resolve_backend("auto")
    assert (b.name, b.device) == ("mlx", "mps")


def test_mps_without_mlx_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(device, "_mlx_available", lambda: False)
    b = device.resolve_backend("mps")
    assert (b.name, b.device) == ("faster", "cpu")
