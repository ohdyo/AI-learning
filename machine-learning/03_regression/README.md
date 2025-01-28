# 경사하강법 (Gradient Descent)
## 학습률 (Learning Rate)
- 최적의 해를 빠르게 혹은 천천히 조금씩 찾아가는 '정도'를 가르키는 하이퍼 파라미터
- 기본 값으로 보통 0.001 사용

## 잔차제곱합(Residual Sum of Squares ,RSS)
- 잔차 = 실제 값 - 예측값
- 잔차제곱합 = (실제값 - 예측값) 의 제곱합
- 회귀 모델의 정확도를 측정하는 지표
  - RSS가 작을수록 정확한 모델
  - RSS가 클수록 잘못된 예측 모델
- 모든 회귀 모델은 RSS가 최소가 되는 방향으로 학습 진행 = 회귀계수(=절편)는 RSS가 최소가 되도록 학습
- 비용함수 R(w)가 가장 작을 때의 w를 찾는 것이 회귀 모델의 목표
  - 매 회차에 계산된 R(w)에서 순간변화율(기울기) 구해야함 -> 미분 사용
  - 우리가 구해야 하는 회귀계수는 하나 이상이므로 우리는 ***편미분*** 사용
    - wO(절편) 고정한 채로 w1의 미분을 구하고, w1을 고정한 채로 wO 미분을 구함
    - 
**경사하강법 수식**
$w_1$ $w_0$을 반복적으로 업데이트하며 최적의 회귀계수를 찾음
<br/>
$w_1 = w_1 - (-η\frac{2}{N}\sum^{N}_{i=1} x_i * (실제값_i - 예측값_i))$
<br/>
$w_0 = w_0 - (-η\frac{2}{N}\sum^{N}_{i=1}(실제값_i - 예측값_i))$

**경사하강법 공식**

$w1 = w1 - (미분값)$

$w1 = w1 - (-학습률 * 2 / N * (x * (실제값 - 예측값))의 합)$

$w0 = w0 - (미분값)$

$w0 = w0 - (-학습률 * 2 / N * (실제값 - 예측값)의 합)$

### 경사하강법 실행 예시
- w0 : 절편(=bias), w1: 기울기(=weights) 초기화
```python
w0 = np.zeros((1, 1))
w1 = np.zeros((1, 1))
```
- 예측값(pred) 계산
```python
# 내적 연산함수(dot())를 통해 X와 w1의 각각의 행열을 내적 연산한다.
y_pred = w0 + np.dot(X, w1)
```
- 잔차(diff) 계산
```python
# 현재값 - 예측값
diff = y - y_pred
```
- 학습률(learning_rate) 설정
```python
learning_rate = 0.1
```
- 데이터 개수 파악
```python
# 평균 오차 계산때 사용
N = len(X)
```
- 가중치(절편 + 기울기) 업데이트
```python
# w0 편미분 (w0를 갱신할 값)
# w0 = w0 - (-학습률 * 2 / N * (실제값 - 예측값)의 합)
w0_diff = -learning_rate * 2 / N * np.sum(diff)
# 절편(w0) 갱신
w0 = w0 - w0_diff

# w1 편미분 (w1를 갱신할 값)
# w1 = w1 - (-학습률 * 2 / N * (x * (실제값 - 예측값))의 합)
w1_diff = -learning_rate * 2 / N * np.dot(X.T, diff)    # (100, 1) (100, 1)
# 가중치(w1) 갱신
w1 = w1 - w1_diff

print(f'1회 업데이트된 회귀계수 w0: {w0}, w1: {w1} ')
```

## 선형 회귀 (Linear Regression)
- 선형 모델은 독립 변수와 종속 변수 간의 관계를 선형 방정식으로 모델링하는 알고리즘
  - 선형 회귀
  - 규제 선형 모델
    - 선형 회귀의 단점을 보완하기 위해 추가적인 제약을 도입한 모델, 과적합 완하, 모델의 일반화 성능향상
      ## 1. ****Ridge Regression (L2 규제)****
      - L2 규제 적용한 회귀 모델
      $
      J(\theta) = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 + \lambda \sum_{j=1}^{p} \theta_j^2
      $
        **손실 함수에 가중치의 제곱합**을 추가하여 ***큰 가중치 억제***
        - 특징 
          - 큰 가중치를 줄여 ***과적합 문제 완화***
          - 가중치가 0에 가까워질 뿐, 완전히 0이 아님
        - 하이퍼 파라미터
          - ***λ (규제 강도)***: 값이 클수록 규제가 강해진다.


      ## 2. ***Lasso Regression (L1 규제)***
      $
      J(\theta) = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 + \lambda \sum_{j=1}^{p} |\theta_j|
      $
      L1 규제를 사용하는 선형 회귀 모델
      손실 함수에 가중치의 절댓값 합을 추가
      - 특징
        - ***일부 가중치를 0으로 만들어 불필요한 특성 제거***
      - 하이퍼 파라미터
        - λ (규제 강도): 값이 클루속 더 많은 가중치를 0으로 만듬
      ## 3. ElasticNet (L2 + L1)
      $
      J(\theta) = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 + \alpha \left( \lambda_1 \sum_{j=1}^{p} |\theta_j| + \lambda_2 \sum_{j=1}^{p} \theta_j^2 \right)
      $
      - 특징
        - Lasso의 특성 선택 기능과 Ridge의 안정성을 모두 제공
      - 하이퍼 파라미터
        - λ1, λ2: L1과 L2 규제의 가중치를 조절한다.
        - α: 규제 강도를 조절한다.
  - ## 다중 선형 회귀(PolynomialFeatures)
    - 회귀식이 선형이 아닌 ***곡선***으로 표현되는 회귀 모델

=> 모든 규져들은 따로 작동하여 학습을 통해 예측값을 도출할 수 있다.
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
#데이터 로드
boston_df = pd.read_csv('./data/boston_housing_train.csv')


```