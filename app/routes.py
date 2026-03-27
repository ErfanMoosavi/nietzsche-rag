from fastapi import APIRouter, Depends, HTTPException, status
from qdrant_client import QdrantClient

from app.dependencies import get_qdrant, get_rag
from app.schemas import RetrieveReq, RetrieveRes
from app.services import Rag

router = APIRouter(prefix="/rag", tags=["Rag"])


@router.post("/retrieval", response_model=RetrieveRes)
def retrieve(
    req: RetrieveReq,
    qdrant: QdrantClient = Depends(get_qdrant),
    rag: Rag = Depends(get_rag),
) -> RetrieveRes:
    try:
        points = rag.retrieve(qdrant, req.text, req.limit, req.book, req.language)
        return RetrieveRes(points=points)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
