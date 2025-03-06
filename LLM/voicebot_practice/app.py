from openai_api import stt, ask_gpt, tts
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import streamlit_chat as st_chat

st.title('나만의 LLM선생님')

# 세션 상태 초기화
if 'count' not in st.session_state:
    st.session_state.count = 0
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None

st.write('음성을 녹음하고 질문하세요')

# 녹음 시작 버튼
if st.button('녹음 시작'):
    st.session_state.recording = True
    st.session_state.audio_bytes = None  # 녹음 시작 시 초기화

# 녹음 중일 때 audio_recorder 호출하여 녹음을 계속 진행
if st.session_state.recording:
    st.session_state.audio_bytes = audio_recorder()

# 녹음 종료 버튼
if st.button('녹음 종료') and st.session_state.recording:
    st.session_state.recording = False

    if st.session_state.audio_bytes:
        st.audio(st.session_state.audio_bytes, format='audio/wav')
        st.session_state.count += 1

        # 음성을 텍스트로 변환
        with st.spinner('음성을 텍스트로 변환 중...'):
            with open(f"recording{st.session_state.count}.wav", "wb") as f:
                f.write(st.session_state.audio_bytes)
            stt_text = stt(is_click=True, count=st.session_state.count)

        # GPT에게 질문
        with st.spinner('GPT에게 질문 중...'):
            gpt_response = ask_gpt(stt_txt=stt_text)

        # GPT의 답변을 음성으로 변환
        with st.spinner('답변을 음성으로 변환 중...'):
            audio_tag = tts(response_txt=gpt_response, count=st.session_state.count)

        # 텍스트와 음성 출력
        st.write(f"질문: {stt_text}")
        st.write(f"답변: {gpt_response}")
        st.markdown(audio_tag, unsafe_allow_html=True)