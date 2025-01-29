# Machine Learning
- 주어진 데이터를 스스로 학습
- 다른 데이터가 주어질때 결과를 예측해주는 모델을 만드는 것
- 정담을 찾기 위해 주어지는 기반 데이터 = 특성 데이터
    - 특징(변수)
    - X (대문자) 데이터
    - 피처 데이터(feature data)
- 정답 데이터
    - 타겟(target)
    - y 데이터
    - 레이블 데이터 (label data = 분류된 데이터)
---
## 머신 러닝의 학습 종류
### 1. 지도 학습
- 정답을 가지고 학습습
- 분류 : 입력 데이터를 주어진 클래스로 분류
- 회귀 : 연속적인 값을 예측하는 문제
### 2. 비지도 학습
- 정답 없이 데이터를 받아 어떻게 구성할지 직접 학습
- 궂닙화 : 비슷한 특성끼리 데이터를 그룹화
### 3. 강화 학습
- 주어진 환경과 상호작용하여 보상을 통해 학습
- 자율주행 게임 등에 주로 사용용
---
## 머신 러닝 절차
### 1. 데이터 로드 및 정제
- 결측치 제거 & 이상치 처리
### 2. EDA
- Exploratory Data Analysis
- 데이터에 대한 이해로 어떤 데이터를 대상으로 어떤 모델을 적용시킬건지 알아낸 목적으로 데이터 분석
    - 분포, 평균, 상관 관계에 있는 데이터는 어떤 것들이 있는지 특징을 알아내야 한다.
        - 각각의 변수값이 결과에 영향을 미치는지
        - 학습에 어떤 변수값을 사용하는지
        - 해당 데이터에 대해 어떤 모델을 적용할건지
### 3. <a href="https://github.com/ohdyo/AI-learning/blob/main/machine-learning/01_overview/README.md#%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC">전처리</a>
- 인코딩
    - ***레이블 인코딩***
        - 글자를 숫자로 변환
    - ***원 - 핫 인코딩***
        - 배열에서 어떤 데이터에 해당할 때 해당 위치를 1로 표시하는 인코딩 방식
- 피처 스케일링
    - ***표준화*** (Standardization)
        - 평균균이 0이고 분산이 1인 가우시안 정규 분포를 가진 값으로 변환
        - 수치 값은 무관
        - 일반적으로 주로 사용, 회긔 분류 등에 사용
    - ***정규화*** (Nomalization) = MinMaxScalar
        - 최솟값이 0이고 최대값이 1 사이인 값으로 변환
        - 수치 값은 무관
        - 픽셀 데이터와 같은 이미지 처리에 사용

### 4. 데이터 분류
- 학습 데이터와 테스트 데이터 분류
- 학습 시 7:3의 비율로 학습 데이터와 테스트 데이터로 분류
    - 그래야 학습도 하고 확인 절차도 가능하기 때문

### 5. 모델 선정
- 어떤 모델을 사용하지 결정 하는 단계
    - EDA를 통해 모델을 통해 해결해야 하는 상황을 판단 하고 문제 해결을 위한 모델을 선정 가능

### 6. 학습
- 선정한 모델을 나눈 학습용 데이터 학습 단계

### 7. 평가
- 성능에 대한 평가
- 원하는 값이 안나오면 ***4단계로 이동***후 성능 올리기 가능
    - 정답 데이터가 부족하면 추가
    - 전처리 다시 진행
    - 모델을 바꾸는 방식

### 8. 피드백
- 평가 결과 성능이 안좋으면 데이터 분류 이하의 순서로 다시 돌아가 절차 진행
- 성능이 좋아도 실전 데이터로 예측하면 성능이 떨어지는 경우 존재

---

###  학습 종류별 모델
- **지도학습 분류 모델** (데이터를 사전에 정의된 범주로 분류하는 데 사용)

| **모델 명** | **설명** |
| --- | --- |
| <a href="https://github.com/ohdyo/AI-learning/blob/main/machine-learning/04_classification/README.md#logistic-regression">로지스틱 회귀 (Logistic Regression)</a> | 이진 또는 다중 클래스 분류에 사용되며, 출력은 특정 클래스에 속할 확률로 나타남. |
| <a href="https://github.com/ohdyo/AI-learning/blob/main/machine-learning/01_overview/README.md#k-%EC%B5%9C%EA%B7%BC%EC%A0%91-%EC%9D%B4%EC%9B%83-%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8">K-최근접 이웃 (K-Nearest Neighbors, KNN)</a> | 가장 가까운 K개의 이웃을 기반으로 클래스를 예측하는 모델. 데이터의 분포에 따라 성능이 크게 달라질 수 있음. |
| <a href="https://github.com/ohdyo/AI-learning/blob/main/machine-learning/04_classification/README.md#svmsupport-vector-machine">서포트 벡터 머신 (Support Vector Machine, SVM)</a> | 클래스 간의 최대 마진을 찾는 분류 모델로, 고차원 데이터에서도 잘 동작함. |
| <a href="https://github.com/ohdyo/AI-learning/blob/main/machine-learning/04_classification/README.md#%EA%B2%B0%EC%A0%95-%ED%8A%B8%EB%A6%AC---%EB%B6%84%EB%A5%98">결정 트리 (Decision Tree)</a> | 의사결정 규칙을 시각화할 수 있는 트리 구조로 분류. 이해와 해석이 쉬우나, 과적합에 취약할 수 있음. |
| <a href="https://github.com/ohdyo/AI-learning/blob/main/machine-learning/05_ensemble/README.md#bagging"> 랜덤 포레스트 (Random Forest) </a> | 여러 결정 트리를 결합해 예측하는 앙상블 모델로, 과적합을 방지하고 성능을 향상시킴. |
| 나이브 베이즈 (Naive Bayes) | 조건부 확률 기반 분류 모델로, 가정이 간단하고 계산 효율이 높아 텍스트 분류 등에 자주 사용됨. |
| <a href="https://github.com/ohdyo/AI-learning/blob/main/machine-learning/05_ensemble/README.md#gradientboosting"> 그래디언트 부스팅 머신 (Gradient Boosting Machine, GBM) </a> | 이전 트리의 오차를 줄이는 방식으로 여러 트리를 결합해 성능을 향상시키는 모델. **XGBoost**, **LightGBM** 등이 대표적. |

---
지도학습 회귀 모델 (연속적인 값을 예측하는 데 사용)

| **모델 명** | **설명** |
| --- | --- |
| <a href="https://github.com/ohdyo/AI-learning/blob/main/machine-learning/01_overview/README.md#%EC%98%88%EC%B8%A1%EC%BB%AC%EB%9F%BC%EC%9D%B4-%ED%95%98%EB%82%98%EC%9D%B8-%EA%B2%BD%EC%9A%B0">선형 회귀 (Linear Regression)</a> | 독립 변수와 종속 변수 간의 선형 관계를 가정하여 값을 예측하는 모델. |
| <a href="https://github.com/ohdyo/AI-learning/blob/main/machine-learning/01_overview/README.md#%EC%98%88%EC%B8%A1%EC%BB%AC%EB%9F%BC%EC%9D%B4-%EC%97%AC%EB%9F%AC%EA%B0%9C%EC%9D%B8-%EA%B2%BD%EC%9A%B0">다중 선형 회귀 (Multiple Linear Regression)</a> | 여러 독립 변수를 사용해 종속 변수의 값을 예측하는 선형 모델. |
| <a href="https://github.com/ohdyo/AI-learning/blob/main/machine-learning/03_regression/README.md"> 릿지 회귀 (Ridge Regression)</a> | 과적합을 방지하기 위해 L2 정규화(term)를 추가한 선형 회귀 모델. |
| <a href="https://github.com/ohdyo/AI-learning/blob/main/machine-learning/03_regression/README.md">라쏘 회귀 (Lasso Regression)</a> | 과적합을 방지하기 위해 L1 정규화(term)를 추가한 선형 회귀 모델. |
| <a href="https://github.com/ohdyo/AI-learning/blob/main/machine-learning/03_regression/README.md">다항 회귀 (Polynomial Regression)</a> | 독립 변수와 종속 변수 간의 비선형 관계를 나타낼 때 사용하는 회귀 모델. |
| <a href="https://github.com/ohdyo/AI-learning/blob/main/machine-learning/05_ensemble/README.md#histgradientboostingregressor"> 그래디언트 부스팅 회귀 (Gradient Boosting Regression) </a> | 회귀 문제에 적합한 앙상블 학습 모델로, 여러 약한 학습기를 결합해 성능을 향상시키는 방식. **XGBoost**, **LightGBM** 등 사용. |

---

- 비지도학습 모델 (라벨이 없는 데이터를 기반으로 데이터의 패턴을 발견하는 데 사용)

| **모델 명** | **설명** |
| --- | --- |
| <a href="https://github.com/ohdyo/AI-learning/blob/main/machine-learning/07_clustering/README.md#k-%ED%8F%89%EA%B7%A0-%EA%B5%B0%EC%A7%91k-means-clustering"> K-평균 군집화 (K-Means Clustering) </a> | 데이터를 K개의 군집으로 나누어 각 군집의 중심과 가까운 데이터를 그룹화하는 군집화 기법. |
| 계층적 군집화 (Hierarchical Clustering) | 데이터 간의 거리를 측정해 계층적으로 군집을 형성하며, 트리 구조로 시각화 가능. |
| <a href="https://github.com/ohdyo/AI-learning/blob/main/machine-learning/07_clustering/README.md#dbscan-density-based-spatial-clustering-of-application-with-noise"> DBSCAN (Density-Based Spatial Clustering of Applications with Noise) </a> | 밀도 기반의 군집화 기법으로, 밀집된 군집과 잡음을 구분함. 비정형 데이터에 강한 성능을 보임. |
| <a href="https://github.com/ohdyo/AI-learning/blob/main/machine-learning/06_dim_reduction/README.md#pcaprincipal-component-analysis"> 주성분 분석 (PCA, Principal Component Analysis) </a> | 차원을 축소해 데이터의 주요 성분을 추출하는 기법으로, 데이터 시각화나 차원 축소에 사용됨. |
| t-SNE (t-distributed Stochastic Neighbor Embedding) | 고차원의 데이터를 2차원 또는 3차원으로 변환해 데이터의 분포를 시각화하는 비지도학습 기법. |
| UMAP (Uniform Manifold Approximation and Projection) | 차원 축소 기법으로, t-SNE와 유사하지만 더 빠르고 정확한 성능을 제공함. |
| GAN (Generative Adversarial Network) | 생성자와 판별자와의 경쟁 속에서 더 나은 데이터 샘플을 만들어 내는 방식. 이미지 생성, 데이터 증강에 효과적. |
