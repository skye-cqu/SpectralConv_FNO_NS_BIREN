"""采集 BIREN 环境信息并验证 SUPA 上的关键 PyTorch 能力。"""

import argparse
import importlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_command(command: list[str]) -> dict[str, Any]:
    """运行只读诊断命令，并限制输出长度。"""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15,
            check=False
        )
        output = (result.stdout + result.stderr).strip()
        return {
            "command": command,
            "returncode": result.returncode,
            "output": output[:12000]
        }
    except Exception as error:
        return {
            "command": command,
            "error": f"{type(error).__name__}: {error}"
        }


def command_probe() -> dict[str, Any]:
    """检查设备管理器与常用编译器是否存在。"""
    candidates = ["brsmi", "supacc", "supa", "c++", "g++"]
    result: dict[str, Any] = {}
    for name in candidates:
        path = shutil.which(name)
        result[name] = {"path": path}
    brsmi_path = result["brsmi"]["path"]
    if brsmi_path:
        result["brsmi"]["diagnostic"] = run_command([brsmi_path])
    return result


def import_optional(module_name: str) -> dict[str, Any]:
    """尝试导入平台模块并记录公开版本号。"""
    try:
        module = importlib.import_module(module_name)
        return {
            "available": True,
            "version": getattr(module, "__version__", "unknown"),
            "module_file": getattr(module, "__file__", "unknown")
        }
    except Exception as error:
        return {
            "available": False,
            "error": f"{type(error).__name__}: {error}"
        }


def call_if_available(owner: Any, name: str) -> Any:
    """调用无参数查询函数；不存在时返回 None。"""
    value = getattr(owner, name, None)
    if callable(value):
        try:
            return value()
        except Exception as error:
            return f"{type(error).__name__}: {error}"
    return None


def relative_l2_cpu(
    actual: Any,
    expected: Any
) -> float:
    """Compare two tensors on CPU without relying on SUPA reductions."""
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    difference = float(
        (actual_cpu - expected_cpu).abs().square().sum().sqrt().item()
    )
    denominator = float(
        expected_cpu.abs().square().sum().sqrt().item()
    )
    return difference / max(denominator, 1e-12)


def torch_probe() -> tuple[dict[str, Any], bool]:
    """验证 BIREN PyTorch、FFT、复数张量及反向传播。"""
    result: dict[str, Any] = {
        "platform_modules": {
            "torch_br": import_optional("torch_br")
        }
    }
    try:
        import torch
    except Exception as error:
        result["torch_import"] = {
            "passed": False,
            "error": f"{type(error).__name__}: {error}"
        }
        return result, False

    result["torch_import"] = {
        "passed": True,
        "version": torch.__version__,
        "file": torch.__file__
    }
    supa = getattr(torch, "supa", None)
    result["supa_namespace"] = {
        "available": supa is not None
    }
    if supa is None:
        result["critical_tests"] = {
            "passed": False,
            "error": "torch.supa 命名空间不存在"
        }
        return result, False

    is_available = call_if_available(supa, "is_available")
    device_count = call_if_available(supa, "device_count")
    current_device = call_if_available(supa, "current_device")
    device_name = call_if_available(supa, "get_device_name")
    device_properties: Any = None
    get_device_properties = getattr(supa, "get_device_properties", None)
    if callable(get_device_properties):
        try:
            device_properties = str(get_device_properties(0))
        except Exception as error:
            device_properties = f"{type(error).__name__}: {error}"
    result["supa_namespace"].update({
        "is_available": is_available,
        "device_count": device_count,
        "current_device": current_device,
        "device_name": device_name,
        "device_properties": device_properties
    })
    if is_available is not True:
        result["critical_tests"] = {
            "passed": False,
            "error": f"torch.supa.is_available()={is_available}"
        }
        return result, False

    try:
        device = torch.device("supa")
        cpu_x = torch.randn(2, 2, 8, 8)
        x = cpu_x.to(device).requires_grad_(True)
        expected_x_ft = torch.fft.rfft2(cpu_x)
        native_x_ft = torch.fft.rfft2(x)
        sequential_x_ft = torch.fft.fft(
            torch.fft.rfft(x, dim=-1),
            dim=-2
        )
        native_fft_error = relative_l2_cpu(native_x_ft, expected_x_ft)
        sequential_fft_error = relative_l2_cpu(
            sequential_x_ft,
            expected_x_ft
        )
        native_vs_sequential_error = relative_l2_cpu(
            native_x_ft,
            sequential_x_ft
        )
        fft_tolerance = 1e-4
        native_height_axis_valid = native_fft_error <= fft_tolerance
        sequential_valid = sequential_fft_error <= fft_tolerance

        x_ft = sequential_x_ft
        complex_value = x_ft * torch.complex(
            torch.ones_like(x_ft.real),
            torch.ones_like(x_ft.real)
        )
        height_spatial = torch.fft.ifft(complex_value, dim=-2)
        y = torch.fft.irfft(height_spatial, n=8, dim=-1)
        loss = y.square().mean()
        loss.backward()
        synchronize = getattr(supa, "synchronize", None)
        if callable(synchronize):
            synchronize()
        result["critical_tests"] = {
            "passed": True,
            "device": str(x.device),
            "input_dtype": str(x.dtype),
            "fft_dtype": str(x_ft.dtype),
            "output_shape": list(y.shape),
            "gradient_finite": bool(torch.isfinite(x.grad).all().item()),
            "fft_tolerance": fft_tolerance,
            "native_rfft2_relative_l2_vs_cpu": native_fft_error,
            "sequential_rfft2_relative_l2_vs_cpu": sequential_fft_error,
            "native_vs_sequential_relative_l2": native_vs_sequential_error,
            "native_rfft2_height_axis_valid": native_height_axis_valid,
            "selected_fft_strategy": "sequential_rfft_w_then_fft_h"
        }
        passed = bool(
            result["critical_tests"]["gradient_finite"]
            and sequential_valid
        )
        result["critical_tests"]["passed"] = passed
        return result, passed
    except Exception as error:
        result["critical_tests"] = {
            "passed": False,
            "error": f"{type(error).__name__}: {error}"
        }
        return result, False


def collect_report() -> tuple[dict[str, Any], bool]:
    """收集不含密钥的环境摘要。"""
    torch_result, passed = torch_probe()
    try:
        from src.supa import probe_extension

        extension_status = probe_extension()
        project_extension = {
            "available": extension_status.available,
            "module_name": extension_status.module_name,
            "detail": extension_status.detail
        }
    except Exception as error:
        project_extension = {
            "available": False,
            "detail": f"{type(error).__name__}: {error}"
        }
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "executable": sys.executable,
            "working_directory": str(Path.cwd())
        },
        "selected_environment": {
            "SUPA_VISIBLE_DEVICES": os.environ.get("SUPA_VISIBLE_DEVICES", "not set")
        },
        "commands": command_probe(),
        "pytorch": torch_result,
        "project_extension": project_extension
    }
    return report, passed


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="检查 BIREN/SUPA 硬件与 PyTorch 关键能力"
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="可选：把完整结果写入指定 JSON 文件"
    )
    parser.add_argument(
        "--require-extension",
        action="store_true",
        help="项目 SUPA 扩展未构建或不可加载时返回失败"
    )
    return parser.parse_args()


def main() -> int:
    """输出探针结果；关键能力失败时返回 1。"""
    args = parse_args()
    report, passed = collect_report()
    if args.require_extension:
        passed = passed and bool(report["project_extension"]["available"])
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(serialized + "\n", encoding="utf-8")
        print(f"\n结果已写入: {args.json}")
    if passed:
        print("\n[PASS] SUPA 张量、complex64、FFT 和 backward 冒烟测试通过")
        return 0
    print("\n[FAIL] BIREN 关键能力未全部通过，请查看上述诊断")
    return 1


if __name__ == "__main__":
    sys.exit(main())
