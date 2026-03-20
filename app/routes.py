from fastapi import APIRouter

from .schemas import RagReq, RetrieveReq

router = APIRouter()


@router.get("/retrieval")
def retrieve(req: RetrieveReq):
    pass


@router.get("/rag")
def generate_answer(req: RagReq):
    pass
