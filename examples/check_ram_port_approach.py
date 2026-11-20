"""在真机上验证母口粗定位和精对准，完成后停在接近高度。"""
import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="标定工具生成的视觉配置 JSON")
    args = parser.parse_args()
    os.environ["RAM_PORT_VISION_CONFIG"] = str(Path(args.config).resolve())

    # 分离参数解析，--help 无需安装学习框架或打开相机。
    from experiments.ram_insertion.config import EnvConfig
    from experiments.ram_insertion.wrapper import RAMEnv

    config = EnvConfig()
    env = RAMEnv(hz=config.CONTROL_HZ, config=config)
    try:
        _, info = env.reset()
        print(info)
    finally:
        env.close()


if __name__ == "__main__":
    main()
