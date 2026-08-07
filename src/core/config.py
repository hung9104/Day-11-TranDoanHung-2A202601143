"""
Lab 11 — Configuration & API Key Setup
"""
import os


DEFAULT_LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
DEFAULT_GEMINI_MODEL = DEFAULT_LLM_MODEL


def setup_api_key():
    """Load API key for the active model provider."""
    if DEFAULT_LLM_MODEL.startswith(("gpt-", "o1-", "o3-", "openai/")):
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = input("Enter OpenAI API Key: ").strip()
        print(f"OpenAI API key loaded. model={DEFAULT_LLM_MODEL}")
        return

    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = input("Enter Google API Key: ").strip()
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "0"
    print(f"Google API key loaded. model={DEFAULT_LLM_MODEL}")


# Allowed banking topics (used by topic_filter)
ALLOWED_TOPICS = [
    "banking", "account", "transaction", "transfer",
    "loan", "interest", "savings", "credit",
    "deposit", "withdrawal", "balance", "payment",
    "tai khoan", "giao dich", "tiet kiem", "lai suat",
    "chuyen tien", "the tin dung", "so du", "vay",
    "ngan hang", "atm",
]

# Blocked topics (immediate reject)
BLOCKED_TOPICS = [
    "hack", "exploit", "weapon", "drug", "illegal",
    "violence", "gambling", "bomb", "kill", "steal",
]
