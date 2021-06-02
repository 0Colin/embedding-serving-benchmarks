import argparse
from collections import defaultdict
import json
from time import time

import pandas as pd

from serving.huggingface.sentence_encoder import Model


parser = argparse.ArgumentParser()
parser.add_argument("--is_cpu", required=True)
parser.add_argument("--is_onnx", required=True)
parser.add_argument("--is_fp16", required=True)
args = parser.parse_args()


def get_prediciton_timings(m: Model, data: pd.DataFrame):
    """Generate embeddings from the sentence encoder and caputre timings.

    Args:
        m (Model): Sentence encoder.
    """
    timings = defaultdict(list)
    texts = data.text.values.tolist()
    for i in [1, 8, 64, 128]:
        print(f"generating predictions for batch_size: {i} ...")
        for j in range(len(texts) // i):
            batch_texts = texts[i * j : i * (j + 1)]
            st = time()
            _ = m.get_embeddings(batch_texts)
            timings[i].append(time() - st)
    with open(f"report/cpu_{m.cpu}_onnx_{m.onnx}_fp16_{m.fp16}.json", "w") as f:
        json.dump(timings, f)


if __name__ == "__main__":
    cpu = True if args.is_cpu == "t" else False
    onnx = True if args.is_onnx == "t" else False
    fp16 = True if args.is_fp16 == "t" else False
    data = pd.read_csv("data/snips.csv")
    m = Model(cpu, onnx, fp16)
    print(
        f"config: _is_cpu_: {m.cpu} | _is_onnx_: {m.onnx} | _is_fp16_: {m.fp16}"
    )
    get_prediciton_timings(m, data)
