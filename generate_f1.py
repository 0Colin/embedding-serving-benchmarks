import argparse
from time import time

from utils.classifier import IntentClassifier


parser = argparse.ArgumentParser()
parser.add_argument("--df_path", required=True)
parser.add_argument("--is_cpu", required=True)
parser.add_argument("--is_onnx", required=True)
parser.add_argument("--is_fp16", required=True)
args = parser.parse_args()


if __name__ == "__main__":
    total_time = 0

    df_path = args.df_path
    cpu = True if args.is_cpu == "t" else False
    onnx = True if args.is_onnx == "t" else False
    fp16 = True if args.is_fp16 == "t" else False
    print(f"config: _is_cpu_: {cpu} | _is_onnx_: {onnx} | _is_fp16_: {fp16}")

    clf = IntentClassifier(
        df_path=df_path, is_cpu=cpu, is_onnx=onnx, is_fp16=fp16
    )
    st = time()
    clf._encode()
    # only considering time to encode(includes extra time from LabelEncoder as well :()
    total_time += time() - st
    st = time()
    clf._scale()
    clf.train()
    clf.report()
    print(f"total time: {total_time: .2f}s")
