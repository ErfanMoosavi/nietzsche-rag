from fastapi import APIRouter, Depends, HTTPException, status
from openai import OpenAI
from qdrant_client import QdrantClient

from app.dependencies import get_openai, get_qdrant, get_rag
from app.schemas import RagReq, RagRes, RetrieveReq, RetrieveRes
from app.services import Rag

router = APIRouter(prefix="/rag", tags=["Rag"])


@router.post("/retrieval", response_model=RetrieveRes)
def retrieve(
    req: RetrieveReq,
    qdrant_client: QdrantClient = Depends(get_qdrant),
    rag: Rag = Depends(get_rag),
) -> RetrieveRes:
    try:
        points = rag.retrieve(qdrant_client, req.text, req.limit, req.book)
        return RetrieveRes(points=points)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/ask", response_model=RagRes)
def ask_question(
    req: RagReq,
    openai_client: OpenAI = Depends(get_openai),
    qdrant_client: QdrantClient = Depends(get_qdrant),
    rag: Rag = Depends(get_rag),
) -> RagRes:
    try:
        response = rag.generate_response(
            openai_client,
            qdrant_client,
            req.question,
            limit=req.retrieval_limit,
            book=req.based_on,
        )
        return RagRes(answer=response)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
