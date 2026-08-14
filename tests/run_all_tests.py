"""全部测试运行入口"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_test(label: str, command: list[str]) -> bool:
    """运行单个测试脚本，返回是否成功"""
    print(f"\n{'=' * 60}")
    print(f"运行: {label}")
    print(f"{'=' * 60}")
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=False,
        text=True
    )
    return result.returncode == 0


def main() -> int:
    tests: list[tuple[str, list[str]]] = [
        (
            "SpectralConv 正确性",
            [sys.executable, "tests/test_correctness.py"]
        ),
        (
            "SpectralConv 反向传播",
            [sys.executable, "tests/test_backward.py"]
        ),
        (
            "SUPA backend 接口",
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_supa_operator.py",
                "-q"
            ]
        ),
        (
            "FNO-NS 冒烟",
            [sys.executable, "fno_ns/test_smoke.py"]
        ),
        (
            "SpectralConv 性能",
            [sys.executable, "tests/test_performance.py"]
        )
    ]

    passed = 0
    failed = 0

    for label, command in tests:
        if run_test(label, command):
            passed += 1
        else:
            failed += 1
            print(f"  [失败] {label}")

    print(f"\n{'=' * 60}")
    print(f"总计: {passed}/{len(tests)} 通过, {failed} 失败")
    print(f"{'=' * 60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
