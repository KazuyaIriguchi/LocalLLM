import streamlit as st
from streamlit_chat import message
from llama_cpp import Llama

# LLaMAモデルの読み込み
llm = Llama(
    model_path="./models/Phi-3-mini-4k-instruct-q4.gguf",
    n_gpu_layers=0,
)


# チャットボットの応答を生成する関数
def generate_response(prompt):
    output = llm(
        f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
        max_tokens=512,
        stop=["<|end|>"],
        echo=True,
    )
    response = output["choices"][0]["text"].strip()
    return response


# Streamlitアプリケーションの設定
st.set_page_config(page_title="チャットボット", page_icon=":robot_face:")

# チャットの履歴を保存するリスト
chat_history = []

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
    user_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = llm.create_chat_completion(
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
            max_tokens=512,
            stop=["<|end|>"],
            # echo=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append(
        {"role": "assistant", "content": response["choices"][0][]}
    )
