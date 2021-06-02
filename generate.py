import argparse
from time import time

import numpy as np

from serving.huggingface.sentence_encoder import Model


parser = argparse.ArgumentParser()
parser.add_argument("--is_cpu", required=True)
parser.add_argument("--is_onnx", required=True)
parser.add_argument("--is_fp16", required=True)
args = parser.parse_args()


if __name__ == "__main__":
    m = Model(
        cpu=True if args.is_cpu == "t" else False,
        onnx=True if args.is_onnx == "t" else False,
        fp16=True if args.is_fp16 == "t" else False,
    )
    times = []
    text = "hello how are you"
    for i in range(100):
        st = time()
        m.get_embeddings([text] * 8)
        times.append(time() - st)
    print(np.mean(times), np.median(times), np.std(times))
