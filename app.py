import streamlit as st
import time
import re
from transformers import (
    AutoTokenizer, AutoModelForCausalLM
)


# -----------------------------
# Basic page setup
# -----------------------------
st.set_page_config(page_title="Custom LLM Chat", page_icon="💬", layout="centered")
st.title("💬 Chat with Your Custom LLM")


# -----------------------------
# Sidebar: model/config placeholders
# -----------------------------
with st.sidebar:
    clear = st.button("Clear Chat", use_container_width=True)


# -----------------------------
# Session state: message history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    # --- paths ---
    checkpoint_dir = "./best_model"   # change to your checkpoint-* dir

    # --- load model & tokenizer from checkpoint ---
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)

    # GPT2-like models may not have pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    st.session_state.model = model
    st.session_state.tokenizer = tokenizer

# Allow clearing chat from sidebar
if clear:
    st.session_state.messages = []


# -----------------------------
# Placeholder LLM call
# -----------------------------
def call_llm(user_input: str) -> str:
    inputs = st.session_state.tokenizer(user_input, return_tensors="pt").to(st.session_state.model.device)
    gen_ids = st.session_state.model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.9,
        top_p=0.95
    )
    res = st.session_state.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    return res


# -----------------------------
# Render chat history
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -----------------------------
# Chat input and response
# -----------------------------
user_input = st.chat_input("Type your message")
if user_input:
    # Add user message to session
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call LLM and display assistant reply with streaming effect
    response = call_llm(user_input)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        assembled = ""
        # Stream word-by-word while preserving whitespace/newlines
        for tok in re.findall(r"\S+\s*", response):
            assembled += tok
            placeholder.markdown(assembled)
            time.sleep(0.02)

    # Save assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": response})
