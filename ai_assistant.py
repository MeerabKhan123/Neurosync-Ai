"""
NeuroSync AI - AI Wellness Assistant
Supports Gemini, OpenAI and Hugging Face.
API keys are loaded from .env.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

SYSTEM_PROMPT = (
    "You are a wellness assistant for the NeuroSync AI burnout analytics platform. "
    "Give a concise, practical, encouraging response under 150 words. "
    "Focus on sleep, work hours, screen time, exercise, and stress management."
)


def gemini_available():
    return bool(GEMINI_API_KEY)


def openai_available():
    return bool(OPENAI_API_KEY)


def huggingface_available():
    return bool(HF_TOKEN)


def ask_gemini(question):
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"{SYSTEM_PROMPT}\n\nUser: {question}",
    )

    return response.text


def ask_openai(question):
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_tokens=250,
    )

    return response.choices[0].message.content


def ask_huggingface(question):
    from huggingface_hub import InferenceClient

    client = InferenceClient(
        token=HF_TOKEN
    )

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        model="Qwen/Qwen2.5-7B-Instruct",
        max_tokens=250,
    )

    return response.choices[0].message.content


def ask(provider, question):
    if provider == "Gemini":
        if not gemini_available():
            raise RuntimeError("GEMINI_API_KEY not set in .env file.")
        return ask_gemini(question)

    elif provider == "OpenAI":
        if not openai_available():
            raise RuntimeError("OPENAI_API_KEY not set in .env file.")
        return ask_openai(question)

    elif provider == "Hugging Face":
        if not huggingface_available():
            raise RuntimeError("HF_TOKEN not set in .env file.")
        return ask_huggingface(question)

    raise ValueError(f"Unknown provider: {provider}")