from fastapi import APIRouter, Depends
from qdrant_client import QdrantClient

from app.dependencies import get_qdrant, get_rag
from app.schemas import Point, RetreiveRes, RetrieveReq
from app.services import Rag

router = APIRouter(prefix="rag", tags=["rag"])


@router.post("/retrieval", response_model=RetreiveRes)
def retrieve(
    req: RetrieveReq,
    qdrant: QdrantClient = Depends(get_qdrant),
    rag: Rag = Depends(get_rag),
) -> RetreiveRes:
    response = rag.retrieve(qdrant, req.text, req.limit)
    points: list[Point] = []
    for res in response:
        points.append(
            Point(
                text=res.payload["text"],
                book_name=res.payload["book_name"],
                score=res.score,
            )
        )
    return RetreiveRes(points=points)
