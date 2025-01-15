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
- sklearn모듈에 있는 dataset이다.
- load_iris()를 임포트해서 사용한다.
- iris_data는 딕셔너리 형식이여서 변환해줘서 사용하는게 편하다.
- 순서
  1. 데이터 전처리
        - 데이터를 프레임에 담아서 사용하기 편하게 만든다.
    ```python
    df = pd.DataFrame(data=iris_data['data'], columns=iris_data.feature_names)
    df['target'] = iris_data.target
    ```
  2. 정제된 데이터 형식인 numpy형식으로 변환한다.
     - 인자로 사용되는 데이터 형식이 numpy이기 때문에 바꿔준다.
  ```python
  flower_input = df[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']].to_numpy()
  flower_label = df['target'].to_numpy()      
  ```
  3. 훈련 & 테스트 데이터 생성
        - 훈련 및 테스트 데이터를 생성해주는 함수 ***train_test_split***을 사용해서 한번에 데이터를 생성한다.
  ```python
    train_input, test_input, train_label, test_label =\
        train_test_split(flower_input, flower_label, test_size=.25, random_state=42)
  ```
  4. 정규화
        - 표준 편차를 측정하고, 훈련 및 테스트의 표준 편차도 측정하여서 수치 규격을 맞춘다. 
   ```python
    standard_scaler = StandardScaler()
    standard_scaler.fit(train_input)
    train_scaled = standard_scaler.transform(train_input)
    test_scaled = standard_scaler.transform(test_input)  
   ```
  5. 훈련
        - 배울 모듈에 데이터를 학습시킨다.
  ```python
    kn = KNeighborsClassifier()
    kn.fit(train_scaled, train_label)
  ```
  6. 평가
     - 학습시킨 모듈을 평가한다.
  ```python
    kn.score(test_scaled, test_label)
  ```
  7. 예측
    - 예측값을 인자로 받아서 출력해본다.
  ```python
    kn.predict(:5)
  ```
---

## 지도 학습