import os
from pathlib import Path
import shutil # For cleaning up directories

# --- IMPORTANT: Load .env file at the absolute beginning ---
# These imports and calls must happen before any other project-specific imports
# that might rely on environment variables (like agent.agent importing agent.nodes, etc.)
try:
    from dotenv import load_dotenv
    # Define PROJECT_ROOT very early so load_dotenv can use it
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("\nWARNING: OPENAI_API_KEY environment variable not set or empty after .env load.")
        print("Please ensure your .env file is in the project root and contains OPENAI_API_KEY=\"your_key_here\".")
        # You might want to raise an error here to stop execution if the key is critical
        # raise ValueError("OPENAI_API_KEY is not set. Cannot proceed without it.")
    else:
        print("OPENAI_API_KEY loaded successfully from .env file (via test_agent.py).")
except ImportError:
    print("Python 'dotenv' package not found. Install with: pip install python-dotenv")
    print("Please set OPENAI_API_KEY manually in your terminal (e.g., export OPENAI_API_KEY=\"your_key\").")
# --- End .env loading block ---


# Now, import other standard libraries
import json
import requests

# Import necessary components from your agent package
# These imports will now execute *after* the dotenv loading is complete
from agent.agent import app # This imports your compiled LangGraph workflow
from agent.nodes import AgentState # The TypedDict for your agent's state
from agent.config import DEFAULT_API_URL, USER_AGENT, ATTACHMENTS, ATTACHMENT_BASE_URL
from agent.utils import get_file_type, download_file


# --- Test Harness Configuration ---
# PROJECT_ROOT is defined above in the dotenv block
TEST_DATA_DIR = PROJECT_ROOT / "data"
QUESTIONS_FILE = TEST_DATA_DIR / "questions.json"
ATTACHMENTS_DIR = TEST_DATA_DIR / "attachments"


def setup_test_environment():
    """Ensures necessary directories exist and cleans up old attachments."""
    print("Setting up test environment...")
    TEST_DATA_DIR.mkdir(exist_ok=True)
    ATTACHMENTS_DIR.mkdir(exist_ok=True)

    # Clear existing attachments to ensure a fresh run
    for item in ATTACHMENTS_DIR.iterdir():
        if item.is_file():
            item.unlink() # Delete files
        elif item.is_dir():
            shutil.rmtree(item) # Delete directories recursively
    print(f"Cleaned attachments directory: {ATTACHMENTS_DIR}")

    # Also clear the in-memory ATTACHMENTS global for a fresh test run
    ATTACHMENTS.clear()


def download_gaia_questions():
    """Downloads GAIA questions from the scoring server and saves them locally."""
    print(f"Downloading GAIA questions from {DEFAULT_API_URL}/questions ...")
    try:
        response = requests.get(
            f"{DEFAULT_API_URL}/questions",
            headers={"User-Agent": USER_AGENT},
            timeout=20 # Increased timeout for robustness
        )
        response.raise_for_status()
        questions = response.json()
        with open(QUESTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=4)
        print(f"Downloaded {len(questions)} questions to {QUESTIONS_FILE}")
        return questions
    except requests.exceptions.RequestException as e:
        print(f"Error downloading questions: {e}")
        return []


def download_gaia_attachment_local(task_id: str):
    """
    Downloads an attachment for a given task_id and saves it locally.
    Also populates the global ATTACHMENTS dictionary with the content for the agent.
    """
    print(f"  Attempting to download attachment for task {task_id}...")
    try:
        response = requests.get(
            f"{ATTACHMENT_BASE_URL}{task_id}", headers={"User-Agent": USER_AGENT}, timeout=20
        )
        response.raise_for_status()

        content_disposition = response.headers.get("content-disposition", "")
        filename_matches = re.findall(r'filename\*?=(?:UTF-8'')?([^;]+)', content_disposition)
        filename = (
            filename_matches[0].strip().strip('"') if filename_matches
            else f"{task_id}_attachment"
        )
        filename = requests.utils.unquote(filename) # Decode URL-encoded characters

        local_path = ATTACHMENTS_DIR / filename
        with open(local_path, "wb") as f:
            f.write(response.content)

        file_type = get_file_type(filename)

        # Populate the global ATTACHMENTS dictionary, as the agent nodes expect it
        ATTACHMENTS[task_id] = {
            "name": filename,
            "content": response.content,
            "type": file_type,
        }
        print(f"  Successfully downloaded '{filename}' to {local_path} (Type: {file_type}).")
        return task_id # Return the task_id which is used as the key in ATTACHMENTS
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"  No attachment found for task {task_id} (404 Not Found).")
        else:
            print(f"  HTTP error fetching attachment for task {task_id}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  Network error fetching attachment for task {task_id}: {e}")
        return None
    except Exception as e:
        print(f"  An unexpected error occurred fetching attachment for task {task_id}: {e}")
        return None


def run_local_agent_test():
    """
    Runs the agent on downloaded GAIA questions and prints the results.
    Does NOT submit answers to the scoring server.
    """
    setup_test_environment()

    # Try to load questions from local file, if not present, download them
    questions = []
    if QUESTIONS_FILE.exists():
        with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} questions from {QUESTIONS_FILE}")
    if not questions:
        questions = download_gaia_questions()

    if not questions:
        print("No questions to process. Exiting local test.")
        return

    print("\n" + "="*50)
    print("Starting Local Agent Test Run")
    print("="*50 + "\n")

    results = []
    for i, q in enumerate(questions):
        print(f"\n--- Processing Question {i+1}/{len(questions)} (Task ID: {q['task_id']}) ---")
        
        # Reset ATTACHMENTS for each question to prevent cross-contamination
        ATTACHMENTS.clear() 

        attachment_id_for_state = None
        if q.get("has_attachment", False): # Assuming questions JSON might indicate this, otherwise always try download
             attachment_id_for_state = download_gaia_attachment_local(q["task_id"])
        else:
            print(f"  Question {q['task_id']} does not indicate an attachment.")
        
        initial_state = AgentState(
            question=q["question"],
            answer="",
            extracted_data="",
            media_type="",
            attachment_id=attachment_id_for_state, # Use the task_id as the key into the ATTACHMENTS global
            task_id=q["task_id"],
        )

        try:
            # Invoke the LangGraph agent
            final_state = app.invoke(initial_state)

            predicted_answer = final_state["answer"]
            results.append({
                "task_id": q["task_id"],
                "question": q["question"],
                "predicted_answer": predicted_answer,
            })
            print(f"\n  Question: {q['question']}")
            print(f"  Agent's Predicted Answer: {predicted_answer}")

        except Exception as e:
            error_msg = f"ERROR: Agent failed to process question {q['task_id']}: {e}"
            print(f"\n  {error_msg}")
            results.append({
                "task_id": q["task_id"],
                "question": q["question"],
                "predicted_answer": error_msg,
            })

    print("\n" + "="*50)
    print("Local Agent Test Run Summary")
    print("="*50 + "\n")
    if results:
        for res in results:
            print(f"Task ID: {res['task_id']}")
            print(f"  Question: {res['question']}")
            print(f"  Answer: {res['predicted_answer']}\n")
    else:
        print("No results to display.")

    print("\n--- Local Test Complete. No answers were submitted. ---")


if __name__ == "__main__":
    # The load_dotenv call is now at the very top of the file
    # So this block simply calls the main test function.
    run_local_agent_test()