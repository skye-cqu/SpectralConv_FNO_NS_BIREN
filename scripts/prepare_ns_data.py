"""下载并转换 FNO 论文使用的 64×64 Navier-Stokes 数据。"""

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any
from typing import Mapping

import numpy as np


OFFICIAL_SOURCE_PAGE = (
    "https://drive.google.com/drive/folders/"
    "1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing"
)
MIRROR_URL = (
    "https://huggingface.co/datasets/kmario23/standard-pde-benchmark/"
    "resolve/main/ns/NavierStokes_V1e-5_N1200_T20.mat"
)
SOURCE_FILENAME = "NavierStokes_V1e-5_N1200_T20.mat"
SOURCE_SHA256 = "b4995d0180ec15f06878e17527fd3e08a8f504bbc059ee025caffad11a9a82c0"
EXPECTED_INPUT_SHAPE = (1200, 64, 64)
EXPECTED_TRAJECTORY_SHAPE = (1200, 64, 64, 20)


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    """把 URL 下载到临时文件，成功后原子替换目标。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SpectralConv-FNO-data-preparer/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_mat_arrays(path: Path) -> Mapping[str, Any]:
    """读取 MATLAB v5/v7 或 v7.3 文件。"""
    try:
        import scipy.io
    except ImportError as error:
        raise ImportError("转换 .mat 数据需要 scipy") from error
    try:
        return scipy.io.loadmat(path)
    except (NotImplementedError, ValueError):
        try:
            import h5py
        except ImportError as error:
            raise ImportError("读取 MATLAB v7.3 数据还需要 h5py") from error
        arrays: dict[str, Any] = {}
        with h5py.File(path, "r") as archive:
            for key, value in archive.items():
                if not isinstance(value, h5py.Dataset):
                    continue
                array = np.asarray(value)
                arrays[key] = array.transpose(tuple(reversed(range(array.ndim))))
        return arrays


def extract_endpoints(raw: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """提取 ω₀ 与第 20 个时间点，严格校验官方数据形状。"""
    if "a" not in raw or "u" not in raw:
        available = ", ".join(sorted(str(key) for key in raw))
        raise KeyError(f"数据必须包含 a 和 u，实际 keys: {available}")
    inputs = np.asarray(raw["a"])
    trajectory = np.asarray(raw["u"])
    if inputs.shape != EXPECTED_INPUT_SHAPE:
        raise ValueError(
            f"a shape 应为 {EXPECTED_INPUT_SHAPE}，实际为 {inputs.shape}"
        )
    if trajectory.shape != EXPECTED_TRAJECTORY_SHAPE:
        raise ValueError(
            "u shape 应为 "
            f"{EXPECTED_TRAJECTORY_SHAPE}，实际为 {trajectory.shape}"
        )
    targets = trajectory[..., -1]
    if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise ValueError("数据包含 NaN 或 Inf")
    return inputs.astype(np.float32), targets.astype(np.float32)


def save_dataset(
    output_path: Path,
    inputs: np.ndarray,
    targets: np.ndarray
) -> None:
    """保存训练代码可直接读取的紧凑 NPZ。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, a=inputs, u=targets)


def write_metadata(
    metadata_path: Path,
    source_path: Path,
    output_path: Path,
    source_sha256: str
) -> None:
    """记录来源、哈希、时间切片和确定性划分。"""
    metadata = {
        "dataset": SOURCE_FILENAME,
        "official_source_page": OFFICIAL_SOURCE_PAGE,
        "download_mirror": MIRROR_URL,
        "source_file": str(source_path),
        "source_sha256": source_sha256,
        "converted_file": str(output_path),
        "converted_sha256": sha256_file(output_path),
        "input": {
            "key": "a",
            "meaning": "initial vorticity omega_0",
            "shape": list(EXPECTED_INPUT_SHAPE)
        },
        "target": {
            "source_key": "u",
            "source_time_index": 19,
            "meaning": "vorticity omega_T",
            "shape": list(EXPECTED_INPUT_SHAPE)
        },
        "resolution": [64, 64],
        "split": {
            "policy": "official file order, no shuffle",
            "train": [0, 1000],
            "test": [1000, 1200],
            "n_train": 1000,
            "n_test": 200
        },
        "dtype": "float32"
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    """解析下载、校验和转换参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        help="已有官方 .mat 文件；省略时从公开镜像下载"
    )
    parser.add_argument("--url", default=MIRROR_URL)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/raw") / SOURCE_FILENAME
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/navier_stokes_64x64_n1200.npz")
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/navier_stokes_64x64_n1200.metadata.json")
    )
    parser.add_argument("--expected-sha256", default=SOURCE_SHA256)
    parser.add_argument(
        "--allow-hash-mismatch",
        action="store_true",
        help="仅用于已确认的等价重打包文件；元数据仍记录实际哈希"
    )
    return parser.parse_args()


def main() -> int:
    """准备数据并输出可审计摘要。"""
    args = parse_args()
    source_path = args.source or args.cache
    if args.source is None and not source_path.is_file():
        print(f"从公开镜像下载: {args.url}")
        download_file(args.url, source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"源数据不存在: {source_path}")

    actual_sha256 = sha256_file(source_path)
    if actual_sha256.lower() != args.expected_sha256.lower():
        message = (
            f"源文件 SHA256 不匹配：expected={args.expected_sha256}, "
            f"actual={actual_sha256}"
        )
        if not args.allow_hash_mismatch:
            raise ValueError(message)
        print(f"[WARN] {message}")

    inputs, targets = extract_endpoints(load_mat_arrays(source_path))
    save_dataset(args.output, inputs, targets)
    write_metadata(
        args.metadata,
        source_path,
        args.output,
        actual_sha256
    )
    print(f"[PASS] 已生成: {args.output}")
    print(f"[PASS] 元数据: {args.metadata}")
    print("划分: train=[0,1000), test=[1000,1200)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
