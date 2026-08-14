"""Navier-Stokes 数据准备脚本的离线单元测试。"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "prepare_ns_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_ns_data", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
prepare_ns_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_ns_data)


def test_extract_endpoints_uses_final_time(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prepare_ns_data, "EXPECTED_INPUT_SHAPE", (3, 4, 4))
    monkeypatch.setattr(
        prepare_ns_data,
        "EXPECTED_TRAJECTORY_SHAPE",
        (3, 4, 4, 2)
    )
    inputs = np.zeros((3, 4, 4), dtype=np.float32)
    trajectory = np.zeros((3, 4, 4, 2), dtype=np.float32)
    trajectory[..., -1] = 7.0
    actual_inputs, targets = prepare_ns_data.extract_endpoints(
        {"a": inputs, "u": trajectory}
    )
    assert actual_inputs.shape == (3, 4, 4)
    assert targets.shape == (3, 4, 4)
    assert targets.dtype == np.float32
    assert np.all(targets == 7.0)


def test_extract_endpoints_rejects_wrong_resolution(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prepare_ns_data, "EXPECTED_INPUT_SHAPE", (3, 4, 4))
    monkeypatch.setattr(
        prepare_ns_data,
        "EXPECTED_TRAJECTORY_SHAPE",
        (3, 4, 4, 2)
    )
    inputs = np.zeros((3, 2, 2), dtype=np.float32)
    trajectory = np.zeros((3, 2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="a shape"):
        prepare_ns_data.extract_endpoints({"a": inputs, "u": trajectory})


def test_metadata_records_official_split(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mat"
    source_path.write_bytes(b"source")
    output_path = tmp_path / "dataset.npz"
    np.savez(
        output_path,
        a=np.zeros((2, 2, 2), dtype=np.float32),
        u=np.ones((2, 2, 2), dtype=np.float32)
    )
    metadata_path = tmp_path / "metadata.json"
    prepare_ns_data.write_metadata(
        metadata_path,
        source_path,
        output_path,
        prepare_ns_data.sha256_file(source_path)
    )
    text = metadata_path.read_text(encoding="utf-8")
    assert '"train": [' in text
    assert '"n_train": 1000' in text
    assert '"n_test": 200' in text
