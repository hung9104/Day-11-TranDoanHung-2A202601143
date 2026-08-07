"""
Minimal Streamlit chat UI for local testing.

Run:
    streamlit run src/streamlit_app.py
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agents.agent import create_protected_agent
from assignment.pipeline import build_production_plugins
from core.config import DEFAULT_LLM_MODEL
from core.utils import chat_with_agent
from guardrails.output_guardrails import _init_judge


REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")


def run_async(coro):
    return asyncio.run(coro)


@st.cache_resource
def get_protected_runtime():
    plugins = build_production_plugins(use_llm_judge=False)
    _init_judge()
    return create_protected_agent(plugins=plugins)


async def send_message(user_text: str) -> str:
    agent, runner = get_protected_runtime()
    response, _ = await chat_with_agent(agent, runner, user_text)
    return response


st.set_page_config(page_title="VinBank Chat Test", layout="centered")
st.title("VinBank Chat Test")
st.caption("Chat đơn giản để test guardrails. App tự quyết định chặn hay trả lời.")

with st.sidebar:
    st.write(f"Model: `{DEFAULT_LLM_MODEL}`")
    st.write(f"OPENAI_API_KEY: `{'set' if os.getenv('OPENAI_API_KEY') else 'missing'}`")
    if st.button("Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Xin chào. Bạn cứ hỏi như chat bình thường, tôi sẽ tự áp dụng guardrails.",
        }
    ]


for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_prompt = st.chat_input("Nhập câu hỏi hoặc prompt để test...")

if user_prompt:
    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = run_async(send_message(user_prompt))
        st.markdown(reply)

    st.session_state["messages"].append({"role": "assistant", "content": reply})
