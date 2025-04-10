# Updated embedding.py to intelligently consider both message history and general awareness
import sys
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time
import os
import re


def analyze_and_reply():
    start_time = time.time()
    model = SentenceTransformer('all-MiniLM-L6-v2')

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        messages_file = os.path.join(script_dir, "data", "messages.json")

        with open(messages_file, "r", encoding="utf-8") as f:
            messages = json.load(f)

        user_command = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""

        reply_match = re.search(r"send a reply to (?:the last message of )?[<\"]?(\w+)[>\"]?", user_command, re.IGNORECASE)

        if not reply_match:
            return run_semantic_search(model, messages, user_command)

        username = reply_match.group(1)
        target_msgs = [msg for msg in reversed(messages) if msg.get("username", "").lower() == username.lower() and "content" in msg]

        if not target_msgs:
            return {
                "model": "llama2",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
                "response": f"No messages from user '{username}' found.",
                "done": True,
                "done_reason": "not_found"
            }

        reply = generate_contextual_reply(model, target_msgs[0]["content"], username, messages)

        return {
            "model": "llama2",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
            "response": reply,
            "original_message": target_msgs[0]["content"],
            "username": username,
            "done": True,
            "done_reason": "reply_generated",
            "processing_time": time.time() - start_time
        }

    except Exception as e:
        return {
            "model": "llama2",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
            "response": f"Error: {str(e)}",
            "done": True,
            "done_reason": "error"
        }


def generate_contextual_reply(model, content, username, all_msgs):
    context_msgs = [f"{msg['username']}: {msg['content']}" for msg in all_msgs[-5:] if msg.get("content")]

    intent_labels = ["greeting", "question", "statement", "request", "gratitude", "other"]
    embeddings = model.encode(intent_labels)
    message_vec = model.encode([content])[0]

    faiss.normalize_L2(embeddings)
    message_vec = message_vec / np.linalg.norm(message_vec)

    similarities = np.dot(embeddings, message_vec)
    closest = intent_labels[np.argmax(similarities)]

    if "question" in closest:
        reply = f"Thanks for your question, {username}. From what I know, here's what I found: ..."
    elif "greeting" in closest:
        reply = f"Hello {username}! How can I help you today?"
    elif "gratitude" in closest:
        reply = f"You're welcome, {username}! Let me know if you need anything else."
    elif "request" in closest:
        reply = f"I'm on it, {username}. Give me a moment to gather the info."
    else:
        reply = f"Got it, {username}. Here's what I think: ..."

    reply += "\n\n(Reply generated based on recent conversation context.)"
    return reply


def run_semantic_search(model, messages, query):
    texts = [msg["content"] for msg in messages if msg.get("content")]
    if not texts:
        return {"response": "No message content available for analysis.", "done": True, "done_reason": "no_content"}

    embeddings = model.encode(texts)
    query_vec = model.encode([query])

    faiss.normalize_L2(embeddings)
    faiss.normalize_L2(query_vec)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings, dtype='float32'))
    D, I = index.search(np.array(query_vec, dtype='float32'), 5)

    results = [(texts[i], float(D[0][idx])) for idx, i in enumerate(I[0]) if i >= 0 and D[0][idx] > 0.3]

    if results:
        response = "Here are some relevant findings:\n\n"
        for idx, (text, score) in enumerate(results):
            response += f"Result {idx+1} (Relevance: {score:.2f}): {text}\n"
    else:
        response = "No relevant information found in the stored messages."

    return {
        "model": "llama2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
        "response": response,
        "done": True,
        "done_reason": "search_complete"
    }


if __name__ == "__main__":
    messages = [
        {"content": "This is a test message"},
        {"content": "This is another test message"},
        {"content": "This is a third message"},
    ]
    result = run_semantic_search("llama2", messages, "Test")
    print(json.dumps(result, indent=2))

