from fastapi import APIRouter, Depends, HTTPException, status
from qdrant_client import QdrantClient

from app.dependencies import get_qdrant, get_rag
from app.schemas import Point, RetreiveRes, RetrieveReq
from app.services import Rag

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/retrieval", response_model=RetreiveRes)
def retrieve(
    req: RetrieveReq,
    qdrant: QdrantClient = Depends(get_qdrant),
    rag: Rag = Depends(get_rag),
) -> RetreiveRes:
    try:
        response = rag.retrieve(qdrant, req.text, req.limit, req.book)
        points: list[Point] = []
        for res in response:
            points.append(
                Point(
                    text=res.payload["text"],
                    book=res.payload["book"],
                    score=res.score,
                )
            )
        return RetreiveRes(points=points)
    except Exception as e:
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
