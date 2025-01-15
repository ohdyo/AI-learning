# 데이터 전처리
- 모델이 원하는 형태로 데이터 가공
    - 순서
    1. 입력 값이 존재 해야한다.
    2. 입력 값을 통해 원하는 형식으로 데이터를 프레임에 담는다.
        - df형식으로 하는걸 추천
    3. 해당 입력값에서 훈련 데이터와, 클래스를 분류한다.
        - train_test_split
        - 훈련 데이터와 테스트 데이터 샘플링 해야 한다.
            - ***train_test_split***
                - 훈련 데이터와 클래스, 테스트 데이터와 클래스를 임의로 생성해서 반환해줌
                - 필수 인자로 (전체 데이터, 전체 분류 라벨) 이 필요하다.
                - 그 외에 'test_size=.{num}' 과 'random_state={num}'을 쓸수 있다.
    - 정규화 
        - 정규화를 제공해주는 라이브러리 객체를 생성한다.
            - StandardScaler()
        - 훈련데이터를 fit() 함수의 인자로 담는다.
            - StandardScaler().fit(train_input)
        - 훈련데이터랑, 테스트 데이터를 정규화해서 변환한다.
            - StandardScaler().transform(train_input)
            - StandardScaler().transform(test_input)

### 이후 아래 모델들을 이용하여 데이터를 학습시킨다.
---

## K-최근접 이웃 분류 모델
- 입력 데이터와 정답 데이터를 같이 제공한다.
- 모델이 데이터를 로드한다.
    - kn.fit(train_data, train_label)
- 입력을 통해 학습한 모델에 대한 성능을 확인해본다.
    - kn.score(input_data, label_data) # 입력데이터와 정답데이터로만 이뤄저서 반드시 1을 반환
- 임의 데이터를 입력하여 예측값을 도출한다.
    - kn.predict([unknown_data_list | ,unknown_data_list2])
    - 인자로 반드시 2차원 리스트로 선언되어야 한다. (배치 차원으로의 확장)

### 최근접 이웃 모델의 작동 원리
- 주어진 데이터와 가장 가까운 k(기본 5개)개의 이웃을 찾고, 이웃 데이터를 이용한다.
    - k개의 갯수를 변경할려면 객체 선언한 코드의 인자로 n_neighbors = {원하는 갯수}를 인자로 받는다.
- distances : 주어진 데이터에서 다른 모든 데이터포인트와의 거리(유클리드 거리 측정 방식)
- index : 모델 데이터포인트의 인덱스
    - distances, index = kn.kneighbors([[standard_data_x, standard_data_y]]) # 5개의 distances 와 index에 대한 2차원 배열 반환

#### 이웃 분류모델 예제 (iris)

---

## 지도 학습