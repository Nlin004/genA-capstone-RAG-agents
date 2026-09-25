"""One-off diagnostic: lists the Gemini models your API key can actually
use for generateContent. Not part of the app; run manually if the
configured GEMINI_MODEL gets a 404.

Usage:
    python -m scripts.list_gemini_models
"""

from __future__ import annotations

import google.generativeai as genai

from config import get_settings


def main() -> None:
    settings = get_settings()
    genai.configure(api_key=settings.gemini_api_key)

    print("Models that support generateContent:")
    for model in genai.list_models():
        if "generateContent" in model.supported_generation_methods:
            print(f"  {model.name}")


if __name__ == "__main__":
    main()
