from pydantic import BaseModel


class RetrieveReq(BaseModel):
    text: str


class RagReq(BaseModel):
    text: str
