import os

from loguru import logger

MODEL = "sentence-transformers/LaBSE"
ONNX_MODEL = "labse"


if __name__ == "__main__":
    # to onnx
    logger.info(f"converting model {MODEL} to onnx ...")
    os.system(
        f"python utils/convert_graph_to_onnx.py --framework pt --model {MODEL} onnx/{ONNX_MODEL}.onnx"
    )

    # optimize
    logger.info("optimizing the onnx model ...")
    os.system(
        f"python -m onnxruntime_tools.optimizer_cli --input onnx/{ONNX_MODEL}.onnx --output onnx/{ONNX_MODEL}.onnx --model_type bert"
    )

    # to fp16
    # skip for older cards
    logger.info("converting to fp16 ...")
    os.system(
        f"python -m onnxruntime_tools.optimizer_cli --input onnx/{ONNX_MODEL}.onnx --output onnx/{ONNX_MODEL}_fp16.onnx --model_type bert --float16"
    )

    logger.info("done!")
