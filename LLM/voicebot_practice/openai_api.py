import base64
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from openai import OpenAI


# 음성을 텍스트로 변환해주는 함수
def stt(is_click=False, count=0):
    
    if is_click:
        audio_file = f"recording{count}.wav"
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    with open(audio_file, 'rb') as f:
        transcription = client.audio.transcriptions.create(
            model='whisper-1',
            file=f
        )

    os.remove(audio_file)
    return transcription.text 

# GPT에게 질의
def ask_gpt(api_key=OPENAI_API_KEY, stt_txt=''):
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                'role': 'system',
                'content': '''
                            너는 내 LLM 선생님이야. 물어보는거 알려주고 LLM관련된 소리가 아니면 혼내줘
                
                            ### 지시사항 ###
                            - 핵심 용어 설명 해줄것
                            - 핵심 용어 5개에 대한 설명과 출처 작성할 것
                            
                            ### 출력 형식 ###
                            - 핵심 용어 1
                                * 설명 1 ~~
                            - 핵심 용어 2
                                * 설명 1 ~~
                                * 설명 2 ~~
                            '''
            },
            {
                'role': 'user',
                'content': stt_txt
            }
        ],
        temperature=0.5,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    return response.choices[0].message.content

# GPT의 답변을 음성으로 변환
def tts(api_key=OPENAI_API_KEY, response_txt='', count=0):
    client = OpenAI(api_key=api_key)
    
    with client.audio.speech.with_streaming_response.create(
        model='tts-1',
        voice='fable',
        input=response_txt
    ) as trans_tts:
        filename = f"tts_response{count}.mp3"
        trans_tts.stream_to_file(filename)
    
    with open(filename, 'rb') as f:
        data = f.read()
        base64_encoded = base64.b64encode(data).decode()
        audio_tag = f"""
        <audio autoplay="true">
            <source src="data:audio/mp3;base64,{base64_encoded}" type="audio/mp3">
        </audio>
        """
    os.remove(filename)
    return audio_tag