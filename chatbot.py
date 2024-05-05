import time
import streamlit as st
from llama_cpp import Llama
from transformers import LlamaTokenizerFast

MAX_TOKENS = 512
# LLaMAモデルの読み込み
llm = Llama(
    model_path="./models/Phi-3-mini-4k-instruct-q4.gguf",
    n_gpu_layers=0,
)


def response_generator(messages):
    output = llm.create_chat_completion(
        messages=messages,
        stream=True,
        max_tokens=MAX_TOKENS,
        stop=["<|end|>"],
    )

    for chunk in output:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            yield delta["content"]
        time.sleep(0.02)


def num_tokens_from_messages(messages: list):
    # https://platform.openai.com/docs/guides/text-generation/managing-tokens
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hf-internal-testing/llama-tokenizer"
    )

    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(tokenizer.tokenize(value))
            if key == "name":
                num_tokens += -1
    num_tokens += 2
    return num_tokens


def remove_non_system_messages(messages: list, max_tokens: int = MAX_TOKENS):
    num_tokens = num_tokens_from_messages(messages)

    if num_tokens >= max_tokens:
        updated_messages = []
        for message in messages:
            if message["role"] == "system":
                updated_messages.append(message)
        return updated_messages
    else:
        return messages


# Streamlitアプリケーションの設定
st.set_page_config(page_title="チャットボット", page_icon=":robot_face:")

# タイトルの表示
st.title("チャットボット")

if "messages" not in st.session_state:
    st.session_state.messages = []
    # add system prompt
    st.session_state.messages.append(
        {
            "role": "system",
            "content": "あなたはAIアシスタントです。ユーザーの質問に対して適切な情報を提供してください。",
        },
    )

# チャットの履歴を表示する
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What's up?"):
    st.session_state.messages = remove_non_system_messages(st.session_state.messages)
    user_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        response = st.write_stream(response_generator(messages))
    st.session_state.messages.append({"role": "assistant", "content": response})
