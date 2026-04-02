from app.schemas import Point


def format_chat(role: str, message: str) -> list[dict[str, str]]:
    return [{"role": role, "content": message}]


def format_points(points: list[Point]) -> str:
    formatted_points = ""
    for i, point in enumerate(points):
        formatted_points += f"Source {i}: {point.text}\n\n"
    return formatted_points.strip()
