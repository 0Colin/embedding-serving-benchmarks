from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from utils.setup import MODEL, ONNX_MODEL
from onnxruntime import ExecutionMode, InferenceSession, SessionOptions


class Model:
    def __init__(
        self, cpu: bool = True, onnx: bool = False, fp16: bool = False
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.cpu = cpu
        self.onnx = onnx
        self.fp16 = fp16
        self.return_tensors = "pt"
        if self.onnx:
            options = SessionOptions()
            options.intra_op_num_threads = 1
            options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
            self.model = InferenceSession(
                f"onnx/{ONNX_MODEL}_fp16.onnx"
                if self.fp16
                else f"onnx/{ONNX_MODEL}.onnx",
                options,
                providers=[
                    "CPUExecutionProvider"
                    if self.cpu
                    else "CUDAExecutionProvider"
                ],
            )
        else:
            self.model = AutoModel.from_pretrained(MODEL)
            if not self.cpu:
                self.model.to("cuda")
            self.model.eval()

    def encode(self, texts: List[str]):
        """Encodes text using the appropriate tokenizer for ``MODEL_NAME``.

        Args:
            texts (List[str]): Input text for encoding.

        Returns:
            BatchEncoding: Encoded text.
        """
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors=self.return_tensors,
        )
        if self.cpu:
            return tokens
        else:
            return tokens.to("cuda")

    def _pt_output(self, encoded_input) -> List[List[float]]:
        """Model output from the pytorch model.

        Args:
            encoded_input ([type]): Tokenized input.

        Returns:
            List[List[float]]: Embeddings.
        """
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = model_output.pooler_output
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings.tolist()

    def _onnx_output(self, encoded_input) -> List[List[float]]:
        """Model output from the onnx model.

        Args:
            encoded_input ([type]): Tokenized input.

        Returns:
            List[List[float]]: Embeddings.
        """
        if not self.cpu:
            encoded_input = {
                k: v.cpu().numpy() for k, v in encoded_input.items()
            }
        else:
            encoded_input = {k: v.numpy() for k, v in encoded_input.items()}
        _, pooled = self.model.run(None, encoded_input)
        embeddings = pooled / np.linalg.norm(pooled)
        return embeddings.tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embeddings from the sentence encoder.

        Args:
            texts (List[str]): Input texts.

        Returns:
            List[List[float]]: Encoded texts.
        """
        encoded_input = self.encode(texts)
        if self.onnx:
            return self._onnx_output(encoded_input)
        else:
            return self._pt_output(encoded_input)
