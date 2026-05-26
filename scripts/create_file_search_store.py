from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai


def get_store_by_display_name(client: genai.Client, display_name: str):
    for store in client.file_search_stores.list():
        if store.display_name == display_name:
            return store
    return None


def get_or_create_store(display_name: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    existing = get_store_by_display_name(client, display_name)
    if existing:
        return existing.name

    created = client.file_search_stores.create(
        config={"display_name": display_name}
    )
    return created.name


def main() -> None:
    load_dotenv()
    store_name = get_or_create_store("luminate_store")
    print(f"File Search store: {store_name}")


if __name__ == "__main__":
    main()
