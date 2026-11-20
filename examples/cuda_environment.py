"""在导入 JAX 前使用当前 Python 环境中已经安装的 CUDA 组件。"""

import os
from pathlib import Path
import sys
import sysconfig


def configure_cuda_environment():
    # 与 RAM 启动脚本使用相同的组件，路径随当前 Conda/Python 环境解析。
    if sys.platform != "linux":
        return
    nvidia = Path(sysconfig.get_path("purelib")) / "nvidia"
    compiler = nvidia / "cuda_nvcc" / "bin"
    if (compiler / "ptxas").is_file():
        directories = os.environ.get("PATH", "").split(os.pathsep)
        directories = [entry for entry in directories if entry != str(compiler)]
        os.environ["PATH"] = os.pathsep.join([str(compiler), *directories])

    libraries = [nvidia / "cudnn" / "lib", nvidia / "cuda_runtime" / "lib"]
    available = [str(directory) for directory in libraries if directory.is_dir()]
    if available:
        directories = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        directories = [entry for entry in directories if entry and entry not in available]
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([*available, *directories])

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
