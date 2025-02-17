# 텍스트 분석 Text Analysis 요약

- 자연어 처리(NLP)와 텍스트 분석에 대해 다룹니다. 

## 설치 및 다운로드
- `nltk` 라이브러리 설치: `!conda install nltk -y`
- `nltk` 데이터 다운로드: 
    - `nltk.download('punkt')` 
    - `nltk.download('stopwords')`

## 토큰화
- 단어 토큰화: `word_tokenize(text)`
- 문장 토큰화: `sent_tokenize(text)`

## N-그램
- 바이그램(bigram)과 트라이그램(trigram) 생성: `ngrams(tokens, 2)`, `ngrams(tokens, 3)`

## 불용어 제거
<code> CountVectorizer()</code>의 인자로 기능 구현
- `stop_words='english'` : english에서 사용하는 대명사같은 불용어 제거

### 특성 벡터화 Feature Vectorization
1. BOW(Bag of Words): 문서가 가지는 모든 단어를 문맥이나 순서를 무시하고 일괄적으로 단어에 대해 빈도 값을 부여해 피처 값을 추출하는 모델이다.
    
   <img src="https://miro.medium.com/v2/resize:fit:1400/1*S8uW62e3AbYw3gnylm4eDg.png" width="500px">

2. Word Embedding: 단어를 밀집 벡터(dense vector)로 표현하는 방법으로, 단어의 의미와 관계를 보존하며 벡터로 표현한다.
    
    <img src="https://miro.medium.com/v2/resize:fit:1400/1*jpnKO5X0Ii8PVdQYFO2z1Q.png" width="500px">


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

---
## 특성 벡터화 및 전처리
- TfidVectorizer를 이용한 벡터화
    - 특수 문자 처리 : `string.punctuation`
    - 불용어 처리 : `nltk.word_tokenize()`
    - ngram 설정 : WordNetLemmatizer(ngrams=(1,2))
    - 어근 분리 (Lemmatization)

WordNetLemmatizer : `lemmatizer`