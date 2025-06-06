import os, re, base64, tempfile, requests, uuid, json, sys
import pandas as pd
import numpy as np
from typing import TypedDict
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from langgraph.graph import StateGraph, END
import wikipedia, chardet, whisper
from io import BytesIO, StringIO
import gradio as gr
from huggingface_hub import HfApi, get_token
from agent.config import DEFAULT_API_URL, USER_AGENT, ATTACHMENTS, ATTACHMENT_BASE_URL
# Import utility functions from your utils.py
from agent.utils import extract_final_answer, download_file, get_file_type, fetch_task_attachment, get_youtube_transcript
from pathlib import Path
from agent.agent import app
# Get the absolute path of the directory containing app.py (i.e., /home/user/app/agent/)
current_dir = Path(__file__).parent.resolve()
# Get the parent directory (i.e., /home/user/app/)
project_root = current_dir.parent
# Add the project root to sys.path so Python can find 'agent' as a package
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add this debug section right after your imports:
print("=== DEBUG: Environment Check ===")
print(f"OPENAI_API_KEY exists: {bool(os.getenv('OPENAI_API_KEY'))}")
print(f"API key starts with sk-: {os.getenv('OPENAI_API_KEY', '').startswith('sk-')}")
print(f"API key length: {len(os.getenv('OPENAI_API_KEY', ''))}")
print("=== DETAILED DEBUG ===")
key = os.getenv('OPENAI_API_KEY', '')
print(f"Key length: {len(key)}")
print(f"Key repr: {repr(key[:20])}...{repr(key[-10:])}")  # Shows hidden chars
print(f"Has newlines: {'\\n' in key}")
print(f"Has spaces at end: {key != key.strip()}")
print("=== END DETAILED DEBUG ===")
if os.getenv('OPENAI_API_KEY'):
    key = os.getenv('OPENAI_API_KEY')
    print(f"API key preview: {key[:10]}...{key[-4:] if len(key) > 10 else key}")
print("=== END DEBUG ===")

# Gradio integration
def run_and_submit_all():
    ATTACHMENTS.clear()
    results = []
    answers_payload = []
    submit_url = f"{DEFAULT_API_URL}/submit"

    # Get user info from Hugging Face login
    token = get_token()
    if not token:
        return "Please log in with Hugging Face first", pd.DataFrame()

    try:
        username = HfApi().whoami(token)["name"]
    except Exception as e:
        return f"Error getting user info: {str(e)}", pd.DataFrame()

    # Get agent code from environment
    agent_code = os.getenv("SPACE_ID", "unknown_agent")

    print("Fetching questions...")
    try:
        response = requests.get(
            "https://agents-course-unit4-scoring.hf.space/questions",
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        response.raise_for_status()
        questions = response.json()
        print(f"Found {len(questions)} questions to process")
    except Exception as e:
        return f"Error fetching questions: {str(e)}", pd.DataFrame()

    for i, q in enumerate(questions):
        print(f"\nProcessing question {i+1}/{len(questions)} (Task ID: {q['task_id']})")
        attachment_id = fetch_task_attachment(q["task_id"])
        has_attachment = "Yes" if attachment_id else "No"
        print(f"  - Attachment: {has_attachment}")

        initial_state = {
            "question": q["question"],
            "answer": "",
            "extracted_data": "",
            "media_type": "",
            "attachment_id": attachment_id,
            "task_id": q["task_id"],
        }

        final_state = app.invoke(initial_state)

        # Create results for display
        results.append(
            {
                "task_id": q["task_id"],
                "question": q["question"],
                "answer": final_state["answer"],
            }
        )

        # Create payload for submission (correct format)
        answers_payload.append(
            {"task_id": q["task_id"], "submitted_answer": final_state["answer"]}
        )

    # Prepare submission data in required format
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,  # This is the correctly formatted list
    }

    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # Submit results
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        return final_status, pd.DataFrame(results)
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except:
            error_detail += f" Response: {e.response.text[:500]}"
        return f"Submission Failed: {error_detail}", pd.DataFrame(results)
    except requests.exceptions.Timeout:
        return "Submission Failed: The request timed out.", pd.DataFrame(results)
    except requests.exceptions.RequestException as e:
        return f"Submission Failed: Network error - {e}", pd.DataFrame(results)
    except Exception as e:
        return f"An unexpected error occurred during submission: {e}", pd.DataFrame(
            results
        )


# Build Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Agent Evaluation System")
    gr.Markdown(
        "**Instructions:**\n"
        "1. Log in with your Hugging Face account\n"
        "2. Click 'Run Evaluation & Submit All Answers'\n"
        "3. Wait for processing to complete (may take several minutes)"
    )

    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Status", interactive=False)
    results_table = gr.DataFrame(
        label="Results", headers=["Task ID", "Question", "Answer"]
    )

    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])

# Launch the app
if __name__ == "__main__":
    print("Starting AI Agent Evaluation System...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
