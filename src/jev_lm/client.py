import os

from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient


def build_client(timeout: float = 60.0) -> TypeSafeClient:
    _ = load_dotenv()
    return TypeSafeClient(
        api_key=os.environ["TYPESAFE_API_KEY"],
        base_url=os.environ.get("TYPESAFE_ENDPOINT"),
        timeout=timeout,
    )
