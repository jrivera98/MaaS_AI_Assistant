# import streamlit as st
# from agent_core import ask_agent

# st.title("Metro Riksha AI")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# user_input = st.chat_input("Ask a question about metro riksha demand")

# if user_input:

#     st.session_state.messages.append(
#         {"role": "user", "content": user_input}
#     )

#     answer = ask_agent(user_input)

#     st.session_state.messages.append(
#         {"role": "assistant", "content": answer}
#     )

# for msg in st.session_state.messages:

#     with st.chat_message(msg["role"]):
#         st.write(msg["content"])

import streamlit as st
from agent_core import ask_agent
import time

# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(
    page_title="Feeder service Chat Bot",
    page_icon="",
    layout="centered"
)

# -------------------------
# Custom CSS
# -------------------------

st.markdown("""
<style>

.main-title {
    text-align:center;
    font-size:40px;
    font-weight:700;
    color:#2C3E50;
}

.sub-title {
    text-align:center;
    color:gray;
    margin-bottom:30px;
}

.chat-container {
    max-width:800px;
    margin:auto;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------

st.markdown('<div class="main-title">Feeder service Chat Bot</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="sub-title">Ask questions about metro passenger rikshaw demand</div>',
    unsafe_allow_html=True
)

# -------------------------
# Chat History
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Display Previous Messages
# -------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -------------------------
# User Input
# -------------------------

user_input = st.chat_input("Ask something about metro riksha demand...")

if user_input:

    # show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    # thinking animation
    with st.chat_message("assistant"):

        with st.spinner("🤖 Thinking..."):

            answer = ask_agent(user_input)

            time.sleep(0.5)

            st.write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )