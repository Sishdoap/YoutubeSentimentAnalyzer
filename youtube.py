from googleapiclient.discovery import build
import re
from transformers import pipeline
import torch
from dotenv import load_dotenv
import os

load_dotenv()

hf_repo = "Sishdoap/spam-detector-transformer"

spam_classifier = pipeline(
    "text-classification",
    model=hf_repo,
    tokenizer=hf_repo,
    device=0 if torch.cuda.is_available() else -1
)

api_key = os.getenv("YOUTUBE_API_KEY")

youtube = build("youtube", "v3", developerKey=api_key)

def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1)


def get_comments(video_id, max_results=100):
    comments = []
    next_page_token = None

    while len(comments) < max_results:
        request = youtube.commentThreads().list(
            part = "snippet",
            videoId = video_id,
            maxResults = min(100, max_results - len(comments)),
            pageToken = next_page_token,
            textFormat = "plainText"
        )
        response = request.execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments


def get_comments_with_spam_check(video_id, max_results=100):
    comments_with_labels = []
    next_page_token = None

    while len(comments_with_labels) < max_results:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_results - len(comments_with_labels)),
            pageToken=next_page_token,
            textFormat="plainText"
        )
        response = request.execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]

            # Predict if spam
            prediction = spam_classifier(comment)[0]
            label = prediction['label']  # e.g., 'LABEL_0' or 'LABEL_1'

            # Optional: map label to human-readable
            readable_label = "SPAM" if label in ["LABEL_1", "spam", "SPAM"] else "NOT SPAM"

            comments_with_labels.append({
                "comment": comment,
                "label": readable_label,
                "score": round(prediction['score'], 4)
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments_with_labels

def get_video_metadata(video_id):
    request = youtube.videos().list(
        part="snippet",
        id=video_id
    )
    response = request.execute()
    items = response.get("items")
    if not items:
        return None

    snippet = items[0]["snippet"]
    return {
        "title": snippet["title"],
        "channel": snippet["channelTitle"],
        "thumbnail_url": snippet["thumbnails"]["high"]["url"]
    }
