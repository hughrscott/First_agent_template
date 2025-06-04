import os
from pathlib import Path
from dotenv import load_dotenv

# Define project root (assuming check_env.py is in the root)
PROJECT_ROOT = Path(__file__).resolve().parent

print(f"Attempting to load .env from: {PROJECT_ROOT / '.env'}")
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print(f"OPENAI_API_KEY successfully loaded: {api_key[:5]}...{api_key[-5:]} (last 5 chars)")
else:
    print("OPENAI_API_KEY not found or empty after loading .env.")

print(f"Raw os.environ content (first 500 chars): {str(os.environ)[:500]}")