from openai import OpenAI
from texttools import TheTool

from app.config import settings


def translate(text: str, target_lang: str) -> str:
    client = OpenAI(base_url=settings.base_url, api_key=settings.openai_api_key)
    tool = TheTool(client, settings.llm_model)
    output = tool.translate(text, target_lang)
    return output.result
