#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SUPA_BASE="${SUPA_BASE:-/usr/local/birensupa/sdk/1.11.0.0.rc2}"
SUPA_GPU_ARCH="${SUPA_GPU_ARCH:-br100}"
BRCC="${BRCC:-${SUPA_BASE}/brcc/bin/brcc}"
SUPA_PATH="${SUPA_PATH:-${SUPA_BASE}/supa}"
MODULE_NAME="spectralconv_supa_ext"

TORCH_BR_BASE="$("${PYTHON_BIN}" -c \
    'import importlib.util; from pathlib import Path; spec = importlib.util.find_spec("torch_br"); assert spec and spec.origin; print(Path(spec.origin).resolve().parent)')"
export LD_LIBRARY_PATH="${TORCH_BR_BASE}/lib:${SUPA_PATH}/lib:${LD_LIBRARY_PATH:-}"

TORCH_INCLUDE_FLAGS="$("${PYTHON_BIN}" -c \
    'from torch.utils.cpp_extension import include_paths; print(" ".join("-I" + path for path in include_paths()))')"
TORCH_LIBRARY_FLAGS="$("${PYTHON_BIN}" -c \
    'from torch.utils.cpp_extension import library_paths; print(" ".join("-L" + path for path in library_paths()))')"
TORCH_LIBRARY_PATHS="$("${PYTHON_BIN}" -c \
    'from torch.utils.cpp_extension import library_paths; print(":".join(library_paths()))')"
PYTHON_INCLUDE="$("${PYTHON_BIN}" -c \
    'import sysconfig; print(sysconfig.get_path("include"))')"
EXTENSION_SUFFIX="$("${PYTHON_BIN}" -c \
    'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
OUTPUT="${SCRIPT_DIR}/../${MODULE_NAME}${EXTENSION_SUFFIX}"

mkdir -p "${BUILD_DIR}"

"${BRCC}" \
    -fPIC \
    -O2 \
    --supa-gpu-arch="${SUPA_GPU_ARCH}" \
    --supa-path="${SUPA_PATH}" \
    -I"${SUPA_PATH}/include" \
    -c "${SCRIPT_DIR}/complex_mul_kernel.su" \
    -o "${BUILD_DIR}/complex_mul_kernel.o"

g++ \
    -std=c++17 \
    -fPIC \
    -O2 \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -DTORCH_EXTENSION_NAME="${MODULE_NAME}" \
    ${TORCH_INCLUDE_FLAGS} \
    -I"${PYTHON_INCLUDE}" \
    -I"${TORCH_BR_BASE}/include" \
    -I"${SUPA_PATH}/include" \
    -c "${SCRIPT_DIR}/complex_mul_bindings.cpp" \
    -o "${BUILD_DIR}/complex_mul_bindings.o"

"${BRCC}" \
    --supa-link \
    -shared \
    -fPIC \
    --supa-gpu-arch="${SUPA_GPU_ARCH}" \
    --supa-path="${SUPA_PATH}" \
    "${BUILD_DIR}/complex_mul_bindings.o" \
    "${BUILD_DIR}/complex_mul_kernel.o" \
    ${TORCH_LIBRARY_FLAGS} \
    -L"${TORCH_BR_BASE}/lib" \
    -L"${SUPA_PATH}/lib" \
    -lc10 \
    -ltorch \
    -ltorch_cpu \
    -ltorch_python \
    -ltorch_br \
    -lsupa-runtime \
    -o "${OUTPUT}"

echo "Built ${OUTPUT}"
echo "Runtime library order:"
echo "export LD_LIBRARY_PATH=\"${TORCH_BR_BASE}/lib:${TORCH_LIBRARY_PATHS}:${SUPA_PATH}/lib:\${LD_LIBRARY_PATH:-}\""
