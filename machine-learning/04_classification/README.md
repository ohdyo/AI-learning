# 분류 (Classification)
- 입력 데이터를 미리 정외된 여러 클래스 중 하나로 예측하는 것(범주형 데이터)
    - 이진 분류 : 양성(1), 음성(0) 중에 하나를 맞추는 것
    - 다중 분류 : 여러 클래스 중 하나를 맞추는 것

##  Logistic Regression
- 선형 회귀 방식으로 분류 문제를 해결하는 모델
    - 이진 분류 : 이진 분류를 위한 로지스틱 함수(시그모이드)를 통해 확률값을 계산하고 0 또는 1로 분류 
    - 다중 분류 : 다중 분류를 위한 소프트맥스 함수를 통해 각 클래스별 확률값을 계산해 다중 분류
### ***이진 분류를 위한 Sigmoid 함수 + Logistic Regression 이용***
- 선형회귀식을 통해 도출한 예측값(z)을 0과 1 사이의 수로 변환해주는 활성화 함수(Activation Function)
$
    시그모이드(z) = \frac{1}{1+e^{-z}}
$
    - - 시그모이드의 값은 ***z값의 크기와 반비례*** 한다.
```python
# z = 선형회귀 결과 모델
# 시그모이드 시각화
z = np.linspace(-5,5,100) # 선형회귀 결과값
sigmoid_value = 1 /(1 + np.exp(-z)) # np.exp(-z) = e^-z

plt.plot(z, sigmoid_value)
plt.xlabel('Z')
plt.ylabel('sigmoid(z)')
plt.grid()
plt.show()
```
- ***로지스틱 분류 구현***

```python
# 데이터 불러오기기
fish_df = pd.read_csv('./data/fish.csv')
is_bream_orsmelt = (fish_df['Species'] == 'Bream') | (fish_df['Species'] == 'Smelt')
fish_df = fish_df[is_bream_orsmelt]

# 훈련데이터 - 테스트데이터 셋팅 및 정규화
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = fish_df.drop('Species', axis=1)
y = fish_df['Species']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# 로지스틱 회귀모델 학습 및 평가가
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression()
lr_clf.fit(X_train_scaled, y_train)

lr_clf.score(X_train_scaled, y_train), lr_clf.score(X_test_scaled, y_test)  
# (1.0, 1.0)

# 예측값들을 통한 분석
y_pred = lr_clf.predict(X_test_scaled[:3])
y_pred # 'Bream', 'Smelt', 'Smelt'

print(lr_clf.classes_) # ['Bream' 'Smelt']
lr_clf.predict_proba(X_test_scaled[:3])
#array([[0.96120317, 0.03879683],
    #    [0.00842591, 0.99157409],
    #    [0.01439468, 0.98560532]])

# 선형회귀값 직접 계산
z1 = np.dot(X_test_scaled[:3], lr_clf.coef_[0]) + lr_clf.intercept_
# 선형 회귀값 계산 함수 (decision_function)
z2 = lr_clf.decision_function(X_test_scaled[:3])
# (array([-3.20984727,  4.76798194,  4.22639728]),
#  array([-3.20984727,  4.76798194,  4.22639728]))

# 시그모이드 함수 적용
sigmoid_value = 1 / (1 + np.exp(-z1))
sigmoid_value # array([0.03879683, 0.99157409, 0.98560532])

# 시그모이드 함수 적용됬는지 확인
['Smelt' if value >= 0.5 else 'Bream' for value in sigmoid_value] # ['Bream', 'Smelt', 'Smelt']
```


### ***다중 분류를 위한 Softmax함수 + logistic Regression***

    - 다중 클래스 분류를 위한 활성화 함수로 각 클래스에 대한 확률값 계산
    - k 개의 클래스가 존재할 때 주어진 입력에 대해 다음과 같이 계산

$
    softmax(z_i) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$


        - $z_k$ : 각 클래스에 대한 점수 (입력값)
        - $e^{z_k}$ : 해당 점수에 대한 지수 함수 적용
        - $\sum_{j=1}^{K} e^{z_j}$ : 모든 클래스 점수에 대해 지수 함수 적용 후 총합
    - **다중 클래스 확률 계산 순서**
        1. 샘플에 대한 회귀 결고 z 계산
        2. 소프트 맥스 함수 적용
            - z를 e의 지수로 적용해 값을 확대(클래스별 z의 차이를 극대화)
            - ***합을 각 클래스의 값으로 나눠 비율을 계산하고 반환***
        3. 가장 높은 확률 값을 가진 클래스를 선택
```python
# 데이터 셋 생성
from sklearn.datasets import make_classification

X,y = make_classification(    # 분류 문제 연습을 위한 가상 데이터셋 생성 함수
    n_samples=100,      # 샘플 갯수
    n_features=4,       # 전체 특성(=컬럼) 개수 
    n_informative=3,    # 유의미한 특성 개수
    n_redundant=0,      # 중복 특성 개수
    n_classes=3,        # 클래스 수
    random_state=42     # 랜덤 시드
)
df = pd.DataFrame(X, columns=['feat1','feat2','feat3','feat4'])
df['target'] = y

# 데이터 분리
X_train,X_test, y_train,y_test = train_test_split(X,y,random_state=42)

# predict_proba = 클래스별 
lr_clf = LogisticRegression(max_iter=1000)
lr_clf.fit(X_train, y_train)
lr_clf.score(X_train,y_train), lr_clf.score(X_test,y_test)

y_pred = lr_clf.predict(X_test[:5])
y_pred_proba = lr_clf.predict_proba(X_test[:5])
y_pred_proba, y_pred_proba.sum(axis=1)

# 직접 게산
W = lr_clf.coef_
B = lr_clf.intercept_

W.shape, B.shape #((3,4) = (클래스수, 특성수), (3,) = (클래스수))
# 결정함수 (선형회귀값 계산)
Z = lr_clf.decision_function(X_test[:5])

# softmax 함수
def softmax(z):
    exp_z = np.exp(z)
    # sum의 형식을 유지해야만 값을 계산 가능하다.
    sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
    
    return exp_z / sum_exp_z

y_pred_proba = softmax(Z)
y_pred_proba # 아래와 같은 값 나옴

# scipy의 softmax 함수 (다중분류를 위한 softmax함수를 구현해둔 라이브러리)
import scipy
import scipy.special

y_pred_proba = scipy.special.softmax(Z, axis=1)
y_pred_proba # 위랑 같은 값 나옴옴
```

***최종적으로 우리가 구해야할건 Z(선형 모델의 결과 값)의 값 과 이를 통한 softmax함수의 인자로 사용하여서 확률을 구한다. ***

---

## 결정 트리 - 분류
- 스무고개 처럼 질문/결정을 통해 데이터를 분류하는 모델
    - ***데이터 스케일링 영향이 적음***
    - 선형 구조가 아닌 복잡한 구조의 데이터에 적합
    - 과대 적합 되기 쉬움 -> 가지치기 등을 통해 과대적합 방지
- graphviz 모듈을 설치해야 하는데 os에 직접 설치해서 Path로 경로를 이어줘야 한다. 그리고 파이썬 모듈에 'graphviz'를 설치하면 임포트를 하지 않아도 자동으로 사용이 가능하다.
### 이진 분류
```python
# 데이터 로드
wine_df = pd.read_csv('./data/wine_simple.csv')
wine_df

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 분리 및 스케일링
X = wine_df.drop('class', axis=1)
y = wine_df['class']
y.value_counts()

X_train, X_test, y_train,y_test = train_test_split(X,y,random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#모듈 학습
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(random_state=0, max_depth=3)
dt_clf.fit(X_train, y_train)

dt_clf.score(X_train, y_train), dt_clf.score(X_test, y_test)

# 트리 출력
from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(
    dt_clf,
    filled=True,
    feature_names=X.columns,
    class_names=['red wine', 'white wine']
    )
plt.savefig('./images/wind_simple.png')
plt.show()

# 특성 중요도
# [alchole, suagr, pH] 중요도
# - 지니불순도 감소에 기여한 만큼 중요도가 높아짐짐
dt_clf.feature_importances_
```
- 모듈 학습 부분
    - DecisionTreeClassifier()에 사용되는 인자
        - 해당 트리의 깊이를 제한시켜줌
- 트리 출력 부분
    - plt_tree()로 사용되는 이자
        1. dt_clf : 사용되는 학습된 모델
        2. filled=True : 색 채우기 옵션 (특정 클래스의 비율 표현)
        3. feature_name=X.columns : 특성 이름을 순서대로 매칭시켜 가져옴
        4. calss_names=[] : 클래스 이름, 해당 노드 중 더 많은 데이터가 들어간 클래스의 이름을 보여줌
    - plt.savefig('Path')
        - 생성된 트리를 확장장자명에 맞게 변환해서 저장시켜줌

- 트리에 담긴 데이터 확인
```plain text
루트 노드의 출력 데이터
sugar <= 0.284      # DecisionTreeClassifier가 정한 분할기준 (자식 노드에서 지니계수가 최대로 낮아질 수 있는 분할 기준)
gini = 0.373    # 지니계수 = 1 - (음성클래스비율^2 + 양성클래스비율^2)
samples = 4872  # 현재 노드의 전체 샘플수
value = [1207,3665] # 클래스별 샘플 개수 (0번 클래스 1207개, 1번 클래스 3665개)
class = white wine  # 현재 노드의 클래스 (value에서 많은 클래스 선택)
```
- 트리 모델에서 사용되는 주요 메서드
    - feature_importances_
        - 특성 중요도
        - 지니 불순도 감소에 기여한 만큼 중요도가 높아짐

### 다중 분류
```python

# 데이터 로드 및 분리
from sklearn.datasets import load_iris

iris_data = load_iris() # data: x데이터, target : y 데이터
X_train,X_test,y_train,y_test = train_test_split(iris_data.data,iris_data.target,random_state=0)

# 모델 학습 및 평가
dt_clf = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_clf.fit(X_train,y_train)

dt_clf.score(X_train,y_train), dt_clf.score(X_test, y_test)

# 트림모델 시각화
plt.figure(figsize=(20,10))
plot_tree(
    dt_clf,
    filled=True,
    feature_names=iris_data.feature_names,
    class_names=iris_data.target_names
)

plt.show()
```
    
## DecisionTreeRegressor - 회귀
- 각 노드에서 MSE를 최소화하는 방향으로 노드 분할
- 최종 노드(리프노드)에서는 각 샘플들의 평균값을 계산해 예측값으로 사용
```python
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

housing_data = fetch_california_housing()
housing_df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
housing_df[housing_data.target_names[0]] = housing_data.target
housing_df.info()

#학습
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(X_train,y_train)

# 예측
pred_train = dt_reg.predict(X_train)
pred_test = dt_reg.predict(X_test)

# 평가
mse_train = mean_squared_error(y_train,pred_train)
r2_train = r2_score(y_train, pred_train)

mse_test = mean_squared_error(y_test,pred_test)
r2_test = r2_score(y_test, pred_test)

print('train 데이터 평가 :', mse_train, '|', r2_train)
print('test 데이터 평가 : ', mse_test, '|', r2_test)

# 시각화
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(
    dt_reg,
    filled=True,
    feature_names=housing_data.feature_names,
    max_depth=3
)
plt.show()
```
## SVM(Support Vector Machine)
- 이진 분류 문제 해결 (분류 모델)
- SVM호출한 함수의 인자에 담겨지는 하이퍼 파라미터의 요소에 따라 규제를 줘서 성능에 영향을 줄수 있다.
    - C : 학습 데이터의 오류 허용도 결정
        - 값의 크기에 비례하여 마진의 범위가 넓어짐



## SVR(Suppoter Vector Regressor)
- 연속적인 값 예측 (회귀 모델)
- SVR 또한 호출한 함수의 인자에 담긴 하이퍼 파라미터의 요소에 따라 데이터 변환 형식이 다름
    - ***kernel***
        - linear : 선형 커널
            - 데이터가 선형적으로 분리 가능한 경우
        - rbf : Radial Basis Function, 가우시안 커널로 비선형 데이터 처리리
        - poly : 다항식 커널
            - 비선형 관계, 차수 degree로 지정
```python
#데이터 로드
from sklearn.datasets import fetch_california_housing

housing_data = fetch_california_housing()

# 데이터 분리 및 스케일링
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = housing_data.data
y = housing_data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42, test_size=0.2)

scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.fit_transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1))
y_test_scaled = scaler_y.fit_transform(y_test.reshape(-1,1))

# SVR 모델 훈련 및 평가
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

# 학습
svr_model.fit(X_train_scaled,y_train_scaled)

y_pred_scaled = svr_model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))

mse = mean_squared_error(y_test, y_pred)
mse
```

| 특징                   | SVM                                    | SVR                                    |
|----------------------|---------------------------------------|---------------------------------------|
| **목적**              | 이진 분류 문제 해결                   | 연속적인 값 예측                      |
| **결정 경계**         | 서포트 벡터와의 거리를 최대화하여 생성 | 데이터 포인트와의 오차를 최소화하여 생성 |
| **마진/허용 오차**    | 마진을 최대화하여 일반화 성능 향상    | ε 매개변수로 허용 오차 범위 설정       |
| **결과**              | 클래스 예측 (이진 분류)               | 연속적인 값 예측                      |


---
