"""
监控指定目录下的新增文件，自动评估模型性能
"""
import argparse

from hoidet.metrics import auto_eval

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='./predict', type=str,
                    help="待监控的目录")
args = parser.parse_args()
print(args)


if __name__ == '__main__':
    auto_eval(
        monitored_dir="./predict"
    )
