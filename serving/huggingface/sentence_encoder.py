from typing import List

import torch
from transformers import AutoTokenizer, AutoModel
from transformers.tokenization_utils_base import BatchEncoding


class Model:
    def __init__(self, model_name: str, max_len: int):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.max_len = max_len

    def encode(self, texts: List[str]):
        """Encodes text using the appropriate tokenizer for ``MODEL_NAME``.

        Args:
            texts (List[str]): Input text for encoding.

        Returns:
            BatchEncoding: Encoded text.
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

    def output(self, encoded_input: BatchEncoding) -> List[List[float]]:
        """Pooled and normalized output from the model.

        Args:
            encoded_input (BatchEncoding): Input after tokenization.

        Returns:
            List[List[float]]: Embeddings.
        """
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = model_output.pooler_output
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings.numpy().tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embeddings from the sentence encoder.

        Args:
            texts (List[str]): Input texts.

        Returns:
            List[List[float]]: Encoded texts.
        """
        return self.output(self.encode(texts))
