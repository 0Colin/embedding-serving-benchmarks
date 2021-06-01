from typing import List

from pydantic import BaseModel, validator


class EmbeddingRequest(BaseModel):
    texts: List[str]

    @validator("texts")
    def texts_length(cls, texts):
        if len(texts) < 1:
            raise ValueError("texts cannot be an empty list")
        return texts


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
