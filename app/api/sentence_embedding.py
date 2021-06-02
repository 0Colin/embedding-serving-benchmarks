from fastapi import FastAPI, Body

from app.models.pydantic.sentence_embedding import (
    EmbeddingRequest,
    EmbeddingResponse,
)
from serving.huggingface.sentence_encoder import Model


model = Model(cpu=False, onnx=True, fp16=True)
app = FastAPI()


@app.post("/sentence-embedding/", response_model=EmbeddingResponse)
def get_sentence_embeddings(
    embedding_request: EmbeddingRequest = Body(
        ..., example={"texts": ["hello world!"]}
    )
) -> EmbeddingResponse:
    """Endpoint to generate sentece embeddings using huggingface models.

    Args:
        embedding_request (EmbeddingRequest, optional): Input texts.

    Returns:
        EmbeddingResponse: Encoded texts.
    """
    embeddings = model.get_embeddings(embedding_request.texts)
    embedding_response = EmbeddingResponse(embeddings=embeddings)
    return embedding_response
