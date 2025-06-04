import os
import re
import requests
from youtube_transcript_api import YouTubeTranscriptApi
import wikipedia
# from io import BytesIO, StringIO # These are for data processing, will go with nodes.py
# import pandas as pd # These are for data processing, will go with nodes.py
# import numpy as np # These are for data processing, will go with nodes.py
# import chardet # These are for data processing, will go with nodes.py
# import whisper # These are for audio processing, will go with nodes.py
# import base64 # These are for image processing, will go with nodes.py
# import tempfile # These are for audio processing, will go with nodes.py

# Import configuration variables from your config.py
from .config import USER_AGENT, ATTACHMENTS, ATTACHMENT_BASE_URL


# Initialize wikipedia with the user agent from config
wikipedia.set_user_agent(USER_AGENT)


# Utility Functions
def extract_final_answer(response_text: str) -> str:
    """
    Extracts the final answer from a response string based on a specific template.
    """
    match = re.search(r"FINAL ANSWER: ?([^\n\.]+)", response_text)
    if match:
        return match.group(1).strip()
    return "unknown"


def download_file(url: str) -> bytes:
    """
    Downloads a file from a given URL.
    """
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        response.raise_for_status() # Raise an exception for HTTP errors
        return response.content
    except Exception as e:
        print(f"Error downloading file from {url}: {str(e)}")
        return None


def get_file_type(filename: str) -> str:
    """
    Determines the type of file based on its extension.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".mp3", ".wav", ".m4a"): # Added m4a for common audio
        return "audio"
    if ext in (".xls", ".xlsx", ".csv", ".json", ".py", ".txt"): # Added .txt for general text files
        return "data"
    if ext in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"): # Added more image types
        return "image"
    if ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"): # Added more video types
        return "video"
    return "unknown"


def fetch_task_attachment(task_id: str) -> str:
    """
    Fetches an attachment for a given task ID from the scoring server.
    The content is stored in the global ATTACHMENTS dictionary.
    Returns the task_id if successful, None otherwise.
    """
    try:
        response = requests.get(
            f"{ATTACHMENT_BASE_URL}{task_id}", headers={"User-Agent": USER_AGENT}, timeout=15
        )
        response.raise_for_status()

        content_disposition = response.headers.get("content-disposition", "")
        # Use re.findall to handle cases where filename might be quoted or not
        filename_matches = re.findall(r'filename\*?=(?:UTF-8'')?([^;]+)', content_disposition)
        filename = (
            filename_matches[0].strip().strip('"') if filename_matches
            else f"{task_id}_attachment"
        )
        # Decode URL-encoded characters if necessary (e.g., %20 to space)
        filename = requests.utils.unquote(filename)


        ATTACHMENTS[task_id] = {
            "name": filename,
            "content": response.content,
            "type": get_file_type(filename),
        }
        print(f"  Downloaded attachment '{filename}' for task {task_id}")
        return task_id
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"No attachment found for task {task_id} (404 Not Found).")
        else:
            print(f"HTTP error fetching attachment for task {task_id}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching attachment for task {task_id}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred fetching attachment for task {task_id}: {e}")
        return None


def get_youtube_transcript(video_url: str) -> str:
    """
    Extracts the transcript from a YouTube video URL.
    """
    try:
        video_id_match = re.search(
            r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)", video_url
        )
        if not video_id_match:
            print(f"Could not extract video ID from URL: {video_url}")
            return ""

        video_id = video_id_match.group(1)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry["text"] for entry in transcript])
    except Exception as e:
        print(f"Error getting YouTube transcript for {video_url}: {str(e)}")
        return ""