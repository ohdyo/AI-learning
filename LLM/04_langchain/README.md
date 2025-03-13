# LangChain

## PromptTemplate
- 질의할 형식을 미리 정의하는 함수
- 질의한 형식에 **f스트링 포맷이 필요 없음**
    - 종류
    1. PromptTemplate
    2. FewShotPromptTemplate
    3. ChatPromptTemplate
        - 유형별 메시지 작성이 가능한 템플릿
        ```python
        from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

        sys_msg = SystemMessagePromptTemplate.from_template('당신은 친절한 챗봇입니다.')
        hm_msg = HumanMessagePromptTemplate.from_template('질문: {question}')
        msg = ChatPromptTemplate.from_messages([sys_msg, hm_msg])

        msg.format_messages(question='AI는 배우려면 뭐부터 해야 하나요?')
        ```

## OutputParser
- 질의 후 출력할 형식을 지정할수 있게 만들어줌
- 종류
    1. `CommaSeparatedListOutputParser`
        - 컴마 기준으로 분리 후 리스트로 반환

---

###  HuggingFace Model
- langchain에서 huggingface를 이용하는 방식
- `from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace`
    - `HuggingFaceEndpoint`
        - 허깅페이스에 접속할 url주소의 엔드포인트
        - repo_id : 허깅페이스에서 사용할 모델 이름 담는 곳
        - task : 어떤 일에 쓰일지 적는 곳
    - `ChatHuggingFace`
        - 질의모델을 생성해주는 클래스
        - llm : 위에서 생성한 endpoint를 담는다.
        - verbose : 로그 출력 여부를 선택한다.
```python
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

endpoint = HuggingFaceEndpoint(
    repo_id='Bllossom/llama-3.2-Korean-Bllossom-3B',
    task='text-generation',
    max_new_tokens=1024
)

hf_model = ChatHuggingFace(
    llm=endpoint,
    verbose=True    # 로그 출력 여부
)

hf_model.invoke('저는 아침으로 사과를 먹었습니다. 저는 아침에 뭘 먹었을까요?')
```

---

## Chains
- 연쇄작용을 뜻하고 순차적으로 연결되어서 수행할수 있게 해줌
- 중간에 `|`를 사용해서 이어줌
```python
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI

prompt = PromptTemplate(
    template='{country}의 수도는 어디인가요?',
    input_variables=['country'],
)

model = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0
)

# 3. OutputParser (StrOutputParser)
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke('대한민국')
```

## ***Memory***
- 이전에 질문한 쿼리를 **기억**해서 다음 쿼리를 실행할수 있게 해줌
- `from langchain_core.chat_history import BaseChatMessageHistory`
    - 해당 클래스 내부의 메소드를 오버로딩하여 사용해서 클래스를 구현해서 사용한다.
    ```python
    class InMemoryHistory(BaseChatMessageHistory):
        def __init__(self):
            self.messages = []   # 대화 저장하는 리스트

        def add_messages(self, messages):
            self.messages.extend(messages)
            
        def clear(self):
            self.messages = []
            
        def __repr__(self):
            return str(self.messages)
    ```

- 순서
    - 구현해둔 모듈을 불러와서 딕셔너리에 저장해서 사용한다.
    ```python
    store = {}  # item(key=session_id, values=InMemoryHistory_인스턴스)

    def get_by_session_id(session_id):
        if session_id not in store:
            store[session_id] = InMemoryHistory()
        return store[session_id]
    ```
    - prompt와 model을 구현한다.
    ```python
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory

    prompt = ChatPromptTemplate.from_messages([
        ('system','너는 {skill}을 잘하는 AI 어시스턴트야.'),
        MessagesPlaceholder(variable_name='history'), # history라는 변수 이름에 이전 대화 내용 저장
        ('human','{query}')
    ])

    model = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.5)
    chain = prompt | model 
    ```    
    - 이전 대화의 내용을 저장하는 chain을 생성한다.
    ```python
    # 이전 대화 내용을 저장하는 chain을 생성
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history = get_by_session_id,
        input_messages_key='query',
        history_messages_key='history'
    )
    ```
    - invoke를 반복하면 store에 자동으로 질문과 답이 기록돼 이전 질의를 기억해서 답변을 도출해준다.
        - 여기서부터 반복해서 질의 문답이 진행된다.
    ```python
        response = chain_with_history.invoke(
        {'skill':'대화','query':'다람쥐는 도토리 나무를 세 그루 키우고 있습니다.'},
        config = {'configurable': {'session_id': 'squirrel'}}
    )

    print(response)
    print(store)
    ```

---

## Agent & Tools
- 모델이 학습할때 사용할 도구(tools)를 상호작용하게 만들어주는 모듈(agent)
- 사용할 도구와 모델을 담아서 사용한다.
- 순서
    1. 담을 도구와 agent를 정의한다.
        - `from langchain.agents import AgentType, initialize_agent, load_tools`
        ```python
        from langchain.agents import AgentType, initialize_agent, load_tools

        # 리스트에 담긴 툴을 로드해주고 모델은 llm인자로 담는다.
        tools = load_tools(['wikipedia', 'llm-math'], llm=model)
        agent = initialize_agent(
            tools,
            model,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,    # 에이전트가 어떻게 동작할지 정의, ZERO_SHOT_REACT_DESCRIPTION: 
            handle_parsing_error=True,  # Agent에서 에러시 다시 작동 여부 
            verbose=True
        )
        ```
- 내가 만들어둔 클래스 혹은 함수를 도구로서 사용 가능하다.
    - 정희한 함수를 `from langchain.tools import Tool` 모듈을 이용해 도구로서 사용한다.
    ```python
    from langchain.tools import Tool
    from langchain.agents import AgentType, initialize_agent, load_tools

    tools = [
        Tool(name='회의 요약', func=summarize_doc, description='긴 회의록 내용 요약'),
        Tool(name='이메일 요약', func=summarize_email, description='업무 이메일 내용 요약')
    ]
    agent = initialize_agent(
        tools=tools,
        llm=model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_error=True,
        verbose=True
    )
    ```

---

### Serp
- 구글에서 검색한 내용을 가져올수 있게 도와주는 api
```python
from langchain.utilities import SerpAPIWrapper
param= {
    'engine': 'google_news',
    'gl' : 'kr',
    'hl' : 'ko'
}
serp = SerpAPIWrapper(params=param)
# news_text = serp.run('코로나')
# news_text[:2]
```

---
## RAG
- 벡터화를 진행하여 문맥 기반 대답을 생성해준다.
- <a href='https://python.langchain.com/docs/tutorials/rag/#indexing'>문서 참고</a>

### Retrieval

