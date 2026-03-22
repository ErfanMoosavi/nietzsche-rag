from fastapi import APIRouter, Depends
from qdrant_client import QdrantClient

from app.dependencies import get_qdrant, get_rag
from app.schemas import Point, RagReq, RetreiveRes, RetrieveReq
from app.services import Rag

router = APIRouter()


@router.post("/retrieval")
def retrieve(
    req: RetrieveReq,
    qdrant: QdrantClient = Depends(get_qdrant),
    rag: Rag = Depends(get_rag),
):
    response = rag.retrieve(qdrant, req.text, req.limit)
    points: list[Point] = []
    for res in response:
        points.append(Point(text=res.payload["text"], score=res.score))
    return RetreiveRes(points=points)


@router.post("/rag")
def generate_answer(req: RagReq, qdrant: QdrantClient = Depends(get_qdrant)):
    pass
