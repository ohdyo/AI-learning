# 딥러닝과 자연어 처리
- 자연어 처리는 인간이 사용하는 자연어를 컴퓨터가 이해하고 처리할 수 있도록 하는 기술

## 전통적 NLP 접근법
- 규칙기반 시스템
    - 수동으로 데이터를 삽입
    - 정확하지만 유연성 떨어짐
- 통계적 모델
    - 단어의 빈도나 출현 확률을 사용
    - 데이터의 품질과 양에 크게 의존, 복잡하면 의미 파악이 어려워짐

## 딥러닝 기반 NLP 접근
- 대량의 데이터를 활용해 자동으로 특징(feature) 학습
- 다층 구조를 통해 복잡한 비선형 관계 학습
- 장점
    - 문맥 이해 : 단어의 의미를 주변 단어와의 관계에서 파악
    - 자연스러운 언어 생성 : 인간과 유사한 수준의 텍스트 생성
    - 적응령 : 새로운 데이터나 언어에 대한 일반화 능력이 뛰어남

=> **딥러닝 모델은 NLP테스크에 높은 성능을 보임**

---
# 자연어 처리 기초
## NLTK (Natural Language Toolkit)
- 파이썬에서 사용하는 텍스트 처리 및 자연어 처리에 도움을 주는 오픈소스 라이브러리
- 주요 기능
    1. **토큰화** (Tokenization)
        - 문장을 단어 또는 문장 단위로 나누는 작업
        - `word_tokenize`, `sent_tokenize`, `ngrams`
    2. 품사 태깅 
        - 각 단어에 대해 해당 품사를 태깅
    3. **어근 및 어간 추출** (=형태소 분석)
        - 단어의 기본 형태를 찾아 동사의 기본형 혹은 복수/단수를 변환해주는 작업
    4. 텍스트 분류
        - Naive Bayes, MaxEnt등의 분류 모델을 통해 텍스트 분류
    5. **코퍼스**
        - 이용할 텍스트 전체 데이터(문장+문장)
        - 정형(CSV,EXCEL), 비정형(JSON,.txt), 반정형(HTML,XML) 데이터로 저장
    6. **불용어 제거**
        - 문장에서 의미 해석에 불필요한 단어 제거
        - `nltk.corpus.stopwords`

## 특성 백터화 (Feature Vectorization)
- 문자열을 컴퓨터가 읽을수 있게 벡터형식으로 변환
    1. **BOW (Bag of Word)**
        - 단어의 빈도수에 값을 부여해 추출하는 모델
        - `CountVectorizer`, `TfidVectorzier`
    2. **Word Embedding**
        - 단어의 밀집도를 분석해 단어의 의미와 관계를 보존하며 벡터 표현

| **구분**             | **Bag-of-Words (BOW)**                             | **Word Embedding**                             |
|----------------------|--------------------------------------------------|------------------------------------------------|
| **개념**             | 문서를 단어의 출현 빈도로 표현                   | 단어를 실수 벡터로 표현                       |
| **특징**             | - 단어의 순서와 의미를 고려하지 않음             | - 단어 간 의미적 유사성을 반영                |
|                      | - 고차원, 희소 벡터 생성                         | - 밀집된 저차원 벡터 생성                     |
| **대표 방법**        | Count Vector, TF-IDF                             | Word2Vec, GloVe, FastText                     |
| **장점**             | - 구현이 간단하고 이해하기 쉬움                  | - 문맥 정보 반영 가능                         |
|                      | - 단순 텍스트 데이터 분석에 유용                 | - 유사한 단어를 벡터 공간 상에서 가깝게 위치 |
| **단점**             | - 의미적 관계와 단어의 순서 정보 없음            | - 많은 데이터와 학습 시간 필요                |
|                      | - 고차원 희소 벡터 문제                          | - 구현이 상대적으로 복잡                      |

### 특성 벡터화 및 전처리
1. 정제 (Cleansing) + 정규화 (Normalization)
불필요한 특수 문자, 이모지, 구두점 등 **노이즈** 제거
    - 특수문자 : `string.punctuation`
        <code>punc_rem_dict = dict((ord(ch), None) for ch in string.punctuation)
    text = text.translate(punc_rem_dict)</code>
    - 소문자 변환
        `text = text.lower()`
    - 숫자 제거
        `text = re.sub(r"\\d+", "", text)`
    - 공백 제거
        `text = " ".join(text.split())`
    - **정규화**
        - 같은 의미의 표기를 일관된 표현으로 통일하는 작업
        `transformed_text = text.replace("United Kingdom", "UK").replace("Uh-oh", "uhoh")`
    
2. 토큰화
    - 토큰화 2가지 방식 (단어 | 문장 토큰화)
        `tokens = nltk.word_tokenize(text)`
        `tokens = nltk.sent_tokenize(text)`  
    - 토큰화 진행후 필요한 정제 다시 진행
        - 빈도수 낮은 단어 제거
        `word_counts = Counter(tokens)`
        `filtered_tokens = [token for token in tokens if word_counts[token] >= 2]`
        - 짧은 단어 제거
        `filtered_tokens = [token for token in tokens if len(token) > 2]`

3. 불용어 제거
    - **stopwords**
        `filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]`


4. **어근 분리 (Lemmatization)** | **어간 추출(stem)**
    - 말의 핵심적인 뜻을 분리시켜주는 역할
        - `nltk.stem.WordNetLemmatizer`
            - `[lemmatizer.lemmatize(token, pos='v') for token in tokens]`
    - 단어의 의미를 담고있는 단어의 핵심부분 추출
        - `from nltk.stem import PorterStemmer`
            - `[PorterStemmer.stemmer.stem(token) for token in tokens]`
5. 특성 벡터화 및 ngram 설정
    ```python
    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=lemmatize,
        stop_words='english',
        ngram_range=(1, 2),
        max_df=0.85,
        min_df=0.05
    )
    ```

---
## 토큰화 (Tokenization)
- 기본적인 문장(sent_tokenize) 과 단어(word_tokenize)외 다른 방식의 토큰화

### Subword 토큰화
- Out of Vocabulary(처음 보는 단어)에 대처 가능

- **BertTokenizer**
    - 단어를 부분단위로 쪼개어 희귀하거나 새로운 단어도 부분적으로 표현 가능 -> 어휘 크기 줄이고 다양한 언어 패턴 학습 가능
    - `BertTokenizer.from_pretrained('bert-base-uncased')`
        - 미리 라이브러리에서 제공된 설정 파일로 학습
        - WordPiece 토크나이저 방식으로 사전에 등록된 단어 우선 사용

- 문자 단위 토큰화
    - 문자 하나 별로 토큰화
    - `list(word)`

- **WordPunctTokenizer** & **TreebankWordTokenizer**
    - 단어/구두점으로 토큰 구분
    - 영어의 줄임마로 사용하는 ' 와 - 도 나눠서 표현

- **kss**
    - Korean Sentence Splitter
    - 한국어 문장단위 코튼 생성 지원
    - `kss.split_sentences(text)`

---

## 품사 태깅

### post_tag
- 단어별로 토큰화된 토큰들의 품사를 알려줌
- `pos_tags = pos_tag(tokens)`

---

## KoNLPy
- 한국어 자연어 처리 라이브러리
- **형태소 분석, 품사 태깅, 텍스트 전처리 지원**
    - 주로 Okt 사용
    - 형태소(의미를 담고있는 단어) 분석 예시
        `morphs = okt.morphs(text)`
- 한국어 불용어 제거
    - 한국어 불용어는 직접 불용어에 해당하는 단어들을 정희해야 한다.(ko_stopwords)
    `cleand_tokens = [token for token in tokens if token not in ko_stopwords]`

---

## 정규 표현식 (Regular Expression)
- 특정한 규칙을 가진 문자열 찾기 위한 패턴
- 대량의 텍스트 데이터에서 특정 패턴을 효율적으로 추출,삭제,대체 가능
- re 모듈을 통해 탐색

### 수량자를 통한 정규 표현식
1. 임의의 한글자 '**.**'
    - . 자리에 **하나의 문자가 반드시** 있어야 함
    - 여러 문자가 존재시 None 반환 앞뒤에 있는 문자도 같아야함
    ```python
    reg_exp = re.compile('a.c')
    print(reg_exp.search('aXc'))
    ```

2. 수량자 *
    - ' * ' **앞의 문자가 0개 이상의 문자**가 존재해야 함
    - 앞뒤의 문자만 해당 문자만 있다면 * 자리에 아무것도 없어도 상관없음
    ```python
    reg_exp = re.compile('ab*c')
    print(reg_exp.search('abc'))
    ```

3. 수량자 ?
    - ' ? ' **앞의 문자가 0개 혹은 1개**의 문자만 있어야함
    ```python
    reg_exp = re.compile('ab?c')
    print(reg_exp.search('ac'))
    ```

4. 수량자 +
    - ' + ' **앞의 문자가 1개 이상** 있어야함
    ```python
    reg_exp = re.compile('ab+c')
    print(reg_exp.search('abbbbbbbbc'))
    ```

5. 수량자 {n} : n개
    - '{n}' **앞의 문자가 n개 있어야** 함  
    ```python
    reg_exp = re.compile('ab{3}c')
    print(reg_exp.search('abbbc'))
    ```

6. 수량자 {min,max} : min개~max개
    - {min,max} **앞의 문자가 min~max개** 있어야 함
    ```python
    reg_exp = re.compile('ab{1,3}c')
    print(reg_exp.search('abbc'))
    ```

- **정규 표현식에 맞는 모든 패턴 찾기**
    - 정규 표현식을 사용하면 앞에서 한번찾고 그대로 반환
    - 이를 해결하기 위해 반복문을 통해 구현
    ```python
    reg_exp = re.compile('a.c')
    for temp in re.finditer(reg_exp, text):
        print(temp)
    ```
### 문자 매칭 정규표현식
1. 문자 매칭 [] : [] 안에 있는 것중 한글자
    - [] 안에 있는 문자 발견시 반환
    - [] 내 '-' 사용시 **문자의 범위**를 표현해주는 방법
    ```python
    # re.IGNORECASE : 대소문자 구분 없이 매칭
    reg_exp = re.compile('[abc]', re.IGNORECASE)
    reg_exp = re.compile('[a-zA-Z0-9]')
    ```
2. 시작하는 문자열 ^
    - 문장 내 해당 문자(or 단어) 갯수만큼 반환
    ```python
    reg_exp = re.compile('^who')
    print(reg_exp.search('who is who'))
    ```

### re 모듈 함수 && re 객체 메소드
1. search() : 문자열 패턴검사
2. match() : 시작하는 문자열 패턴 검사
    - 해당 문자로 반드시 문자열이 시작해야 함
    ```python
    reg_exp = re.compile('ab')
    print(reg_exp.match('abc'))
    ```
3. split() : 정규식 패턴으로 분할
    - 2개의 인자를 기본으로 사용
        1. 탐색할 문자
        2. 정규식을 적용할 문자열
    ```python
    split_text = re.split('[bo]', text, flags=re.IGNORECASE)
    # ['Apple ', 'anana ', 'range'] b와 o 제거후 반환
    ```

4. findall() : 매칭된 결과 모두 반환
    - 2개의 인자를 기본으로 사용
        1. 탐색할 문자
        2. 정규식을 적용할 문자열
    ```python
    # [0-9]+ : 0~9까지의 숫자 중 하나 이상을 반환, 하나 넘으면 계속 탐색후 없을때까지 반환
    nums = re.findall('[0-9]+-[0-9]+-[0-9]+', text)
    ```

5. sub() : 해당 문자를 제외한 나머지 반환
    - 2개의 인자를 기본으로 사용
        1. 탐색할 문자
        2. 정규식을 적용할 문자열
    ```python
    # ^ : not을 의미한다.
    sub_text  = re.sub('[^a-zA-Z ]', '', text)
    ```

---

## 정수 인코딩 (Integer Encodding)
- 자연어 처리는 텍스트 데이터를 수자로 변환하는 것이 핵심
- **등장 빈도에 따라 인덱스를 부여하는것이 일반적**
- 데이터의 노이즈를 중리고 모델의 성능 향상을 위해 5000개로 제한
### 정수 인코딩 구현
```python
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# 문장 토큰화
sentences = sent_tokenize(raw_text)

# 영문 불용어 리스트
en_stopwords = stopwords.words('english')

# 단어사전
vocab = {}

# 토큰화/정제/정규화 처리 결과
preprocessed_sentences = []

for sentence in sentences:
    sentence = sentence.lower()    # 대소문자 정규화 (소문자 변환)
    tokens = word_tokenize(sentence)    # 토큰화
    tokens = [token for token in tokens if token not in en_stopwords]    # 불용어 제거
    tokens = [token for token in tokens if len(token) > 2]    # 단어 길이가 2 이하면 제거

    # 딕셔너리로 선언한 vocab에 key로 token이 없으면 1로 초기화, 있으면 1을 더함
    # 이를 통해 각 단어의 빈도수를 계산
    for token in tokens:
        if token not in vocab:
            vocab[token] = 1
        else:
            vocab[token] += 1
    
    preprocessed_sentences.append(tokens)

# 빈도수 기반 역순 정렬
# vocab.items()를 통해 vocab의 key와 value를 튜플로 반환
# lambda 함수를 통해 value를 기준으로 정렬
vocab_sorted = sorted(vocab.items(), key=lambda item: item[1], reverse=True)

# 인덱스 단어사전 생성
# 단어에서 key값과 value값을 enumerate를 통해서 꺼내옴
# 이중 word를 key로 1번부터 순서대로 다시 딕셔너리 재생성
word_to_index = {word: i+1 for i, (word, cnt) in enumerate(vocab_sorted)}

# 인덱스 단어사전2 생성
# 위와 마찬가지로 이번엔 인덱스를 키로, 값을 단어로 재생성
index_to_word = {i+1: word for i, (word, cnt) in enumerate(vocab_sorted)}

# OOV(Out-Of-Vocabulary) 지정
word_to_index['OOV'] = len(word_to_index) + 1

# 수열 처리
encoded_sentences = []
oov_idx = word_to_index['OOV'] 

for sentence in preprocessed_sentences:
    # 문장별로 토큰화된 2차원 리스트를 이중 for문으 통해
    # 해당 단어가 가리키는 인덱스를 표현후 새로운 리스트에 추가
    encoded_sentence = [word_to_index.get(token, oov_idx) for token in sentence]
    print(sentence)
    print(encoded_sentence)
    print()
    encoded_sentences.append(encoded_sentence)
```
### Keras Tokenizer
- 위의 정수 인코딩 구현을 모듈로 제공해주는 라이브러리
```python
from tensorflow.keras.preprocessing.text import Tokenizer

# num_words: 단어사전에 포함할 최대 단어 수
# oov_token: 단어사전에 없는 단어를 대체할 토큰
tokenizer = Tokenizer(num_words=15, oov_token='<OOV>')

tokenizer.fit_on_texts(preprocessed_sentences)

tokenizer.word_index    # corpus의 모든 단어를 대상으로 생성
tokenizer.index_word    # corpus의 모든 단어를 대상으로 생성
tokenizer.word_counts    # corpus의 모든 단어를 대상으로 빈도수를 반환
# 정수 인코딩
sequences = tokenizer.texts_to_sequences(preprocessed_sentences)
```

---

## Padding
- 자연어 처리에서 각 문장의 길이를 고정되게 맞춰주는 작업
- 장점
    - 일관된 입력 형식
    - 병렬 연산 최적화
    - 유연한 데이터 처리

### 직접 구현
```python
import torch
from collections import Counter

class TokenizerForPadding:
    def __init__(self, num_words=None, oov_token='<OOV>'):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {}
        self.index_word = {}
        self.word_counts = Counter()

    def fit_on_texts(self, texts):
        # 빈도수 세기
        for sentence in texts:
            self.word_counts.update(word for word in sentence if word)

        # 빈도수 기반 vocabulary 생성 (num_words만큼만)
        vocab = [self.oov_token] + [word for word, _ in self.word_counts.most_common(self.num_words - 2 if self.num_words else None)]
        
        self.word_index = {word: i+1 for i, word in enumerate(vocab)}
        self.index_word = {i: word for word, i in self.word_index.items()}

    def texts_to_sequences(self, texts):
        return [[self.word_index.get(word, self.word_index[self.oov_token]) for word in sentence] for sentence in texts]

def pad_sequences(sequences, maxlen=None, padding='pre', truncating='pre', value=0):
    # 패딩 최대 길이 설정
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)

    padded_sequences = []
    for seq in sequences:
        # 문장의 길이가 길다면 잘라내기
        if len(seq) > maxlen:
            if truncating == 'pre':
                seq = seq[-maxlen:]
            else:    # post
                seq = seq[:maxlen]
        # 문장의 길이가 maxlen보다 작으면 0으로 패딩하기
        else:
            pad_length = maxlen - len(seq)
            if padding == 'pre':
                seq = [value] * pad_length + seq
            else:    # post
                seq = seq + [value] * pad_length
        padded_sequences.append(seq)
    
    # 패딩처리한 결과를 tensor형으로 형변환
    return torch.tensor(padded_sequences)

tokenizer = TokenizerForPadding(num_words=15)
tokenizer.fit_on_texts(preprocessed_sentences)
sequences = tokenizer.texts_to_sequences(preprocessed_sentences)
sequences

padded = pad_sequences(sequences, padding='post', maxlen=5, truncating='post')
padded
```

### Keras Tokenizer로 구현
```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences)
sequences = tokenizer.texts_to_sequences(preprocessed_sentences)
sequences

from tensorflow.keras.preprocessing.sequence import pad_sequences

padded = pad_sequences(sequences, padding='post', maxlen=3, truncating='post')
padded
```

---

## 원핫 인코딩 (One hot encoding)
- n개 이상의 '0'과 한개의 '1'로 구성된 리스트로 해당 단어를 가르키는 코딩 방식
- 최근에는 자주 사용 안함
### to_categorical 로 구현
```python
# 패딩 처리된 padded_seqs 까지 완료했을시 사용
# 원핫 인코딩
from tensorflow.keras.utils import to_categorical

one_hot_encoded = to_categorical(padded_seqs)
one_hot_encoded.shape
# 튜플 형식으로 반환
# 첫번째 튜플의 숫자 : 문장수
# 두번쨰 튜플의 숫자 : 문장 내 토큰 개수
# 세번째 튜플의 숫자 : 단어수
```

### 한국어로 토큰화 + 시퀀스 + 패딩 + 원핫 인코딩 구현
```python
texts = [
    "나는 오늘 학원에 간다.",
    "친구들이랑 맛있는 밥 먹을 생각에 신난다.",
    "오늘 구내식당에는 뭐가 나올까?"
]
```
#### 불용어 처리 밑 토큰화
```python
from konlpy.tag import Okt
import re

okt = Okt()

ko_stopwords = ["은", "는", "이", "가", "을", "를", "와", "과", "에", "의", "으로",
                "나", "내", "우리", "들"]

preprocessed_texts = []

for text in texts:
    tokens = okt.morphs(text, stem=True)
    tokens = [token for token in tokens if token not in ko_stopwords]
    tokens = [token for token in tokens if not re.search(r'[\s.,;:?]', token)]
    preprocessed_texts.append(tokens)
```
#### 시퀀스 처리
```
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(preprocessed_texts)
sequences = tokenizer.texts_to_sequences(preprocessed_texts)
sequences
```
#### 패딩 처리
```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_seqs = pad_sequences(sequences, maxlen=3)
```
#### 원핫 인코딩
```python
from tensorflow.keras.utils import to_categorical
one_hot_encoded = to_categorical(padded_seqs)
```

---

## 워드 클라우드
- 텍스트 데이터 시각화
- 단어의 빈도수에 따라 단어의 크기를 다르게 표현해줌
- 워드 클라우드 구현 순서
    1. 텍스트 전처리
    ```python
    okt = Okt()
    nouns = okt.nouns(corpus)
    ```
    2. 단어 빈도 계산
    ```python
    word_count = Counter(nouns)
    ```
    3. 단어 크기 결정
    4. 단어 배치
    ```python
    wordcloud = WordCloud(
    width=800,
    height=800,
    font_path='C:\\Windows\\Fonts\\H2GTRE.TTF',
    background_color='white'
    ).generate_from_frequencies(word_count)
    wordcloud
    ```
    5. 시각화
    ```python
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    ```

---
## Word Embedding
- 단어를 **고정된 차원의 벡터로 변환**
- 숫자로 표현된 단어 목록을 통해 단어의 관계파악 가능
    - 유사한 의미의 단어는 벡터 공간에 가깝게 위치
    - **밀집 벡터**
        - BOW, TF-IDF와는 다른 **저차원 밀집 벡터로 변환**
- 단어의 의미를 추론해준다.
    - **비지도 학습**

### 희소 표현 
- 원-핫 인코딩이 대표적인 희소표현 방식
    - 대부분의 값이 0, 해당 인덱스만 1인 인코딩 방식
    - 단어 벡터간 **유의미한 유사성 표현 불가능**

### 분산 표현
- Word Embedding이 대표적인 분산 표현 방식
- 비슷한 문맥에서 등장하는 단어들은 비슷한 의미를 가짐이 전제
- **단어의 의미를 여러 차원에 걸쳐 분산 표현**
- 원핫 백터는 단어의 갯수만큼 차원이 필요하지만 상대적으로 저차원으로 표현 가능
- **Word2Vec**로 단어간 유의미한 유사도를 계산해줌
    1. **CBOW** : 주변 단어로 중심단어 예측
    2. **Skip-gram** : 중심 단어로 주변 단어를 예측
    - **은닉층이 하나뿐인 얕은 신경망**
        - 학습 대상이 되는 주요 가중치
            1. **투사층**
                - 활성화 함수가 없음
                - 초기 가중치 W는 V(단어의 집합 크기), M(벡터의 차원)으로 이뤄진 행렬로 표현
                - W 행렬의 각 행이 학습 후 단어의 M차원 임베딩 벡터로 변환 (= 룩업 테이블)
            2. **출력층**
                - 투사층과 출력층 사이의 가중치는 W의 전치 행렬로 표현된다.
            3. 예측 과정
                1. 투사층을 통과한 룩업 테이블의 평균을 출력층의 가중치와 내적 후(M차원 벡터) 활성화 함수를 적용해 예측값 구한다(one-hot벡터와 같은 차원).
                2. 예측값 (벡터화된 스코어)는 실제 값과 비교해 크로스 엔트로피 함수로 손실값 계산한다. 
    - 일반적으로 skip-gram이 cbow보다 성능이 좋다.
```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=preprocessed_sentences, # corpus
    vector_size=100,                  # 임베딩 벡터 차원
    sg=0,                             # 학습 알고리즘 선택 (0=CBOW, 1=Skip-gram)
    window=5,                         # 주변 단어 수 (앞뒤로 n개 고려)
    min_count=5                       # 최소 빈도 (빈도 n개 미만은 제거)
)

# 학습된 임베딩 모델 저장
model.wv.save_word2vec_format('ted_en_w2v')

# 임베딩 모델 로드
from gensim.models import KeyedVectors
load_model = KeyedVectors.load_word2vec_format('ted_en_w2v')

# 학습한 모델과 저장된 모델이 같은지 파악
model.wv.most_similar('man')
load_model.most_similar('man')
```

---

## fastText
- 자연어 처리 작업에서 텍스트 분류 및 단어 임베딩에 빠르고 효율적인 도구로 사용
- 주요 특징
    1. **단어 벡터 학습** (Word Embeddings)
        - 기존의 word2vec와 유사하지만 **단어를 서브워드 단위로 처리**
        - **단어를 n-gram으로 분해하여 학습 (서브워드)**
            - 희귀단어나 철자 오류에도 강건
            - 각 서브워드 벡터의 합으로 표현
    2. 텍스트 분류
    3. 효율적인 구현
    5. Word2Vec와 구현이 유사하다.

