# 평가
---
## 분류 모델 평가
### 정확도(Accuracy)
- 전체 샘플 중에서 올바르게 예측한 샘플의 비율
- 데이터가 불균형한 경우 정확도는 비현실적인 성능을 낼 수 있음
- 분류 모델에서 주로 사용하는 평가 방법

```python
# 잘못 학습된 모델 만들어보기
from sklearn.base import BaseEstimator
import numpy as np

# 성별로만 판별하는 모델 작성
class MyTitanicClassifier(BaseEstimator):
    def fit(self, X, y):
        # 훈련 메서드
        pass

    def predict(self, X):
        # 결과 예측 메서드
        pred = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            sex = X['Sex'].iloc[i]
            if sex == 0:    # 여성
                pred[i] = 1     # 생존
        return pred

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/titanic.csv')

# 입력/라벨 데이터 분리
X = df.drop('Survived', axis=1)
y = df['Survived']

# 전처리
X = preprocess_data(X)

# 훈련/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.metrics import accuracy_score

# 모델 훈련
my_classifier = MyTitanicClassifier()
my_classifier.fit(X_train, y_train)

# 예측
pred_train = my_classifier.predict(X_train)
pred_test = my_classifier.predict(X_test)

# 평가 (accuracy_score 사용)
print("훈련 데이터 정확도:", accuracy_score(y_train, pred_train))
print("평가 데이터 정확도:", accuracy_score(y_test, pred_test))
```

###  혼동 행렬(Confusion Matrix)
<img width=50% src="https://d.pr/i/rtYBJv+">

- **정밀도(Precision)**
    - 양성이라고 예측한 것 (TP + FP) 중에 실제 양성(TP) 일 확률
    - 정밀도가 중요한 지표인 겨우
        - 음성인 데이터를 양성으로 예측하면 안되는 경우 (스팸메일 분류 등)


- **재현율 (Recall)**
    - 실제 양성(TP+FN) 중에 양성을 예측(TP)한 확률
    - 재현율이 중요한 지표인 경우 : 양성인 데이터를 음성으로 예측하면 안되는 경우(암 진단 보험 / 금융 사기)

```python
from sklearn.metrics import confusion_matrix, precision_score, recall_score

matrix = confusion_matrix(y_test, pred_test)
matrix
# array([[115,  24],
#       [ 25,  59]])

# 정밀도 (Precision)
p_score = 59 / (24 + 59)
p_score, precision_score(y_test, pred_test)

# 재현율
recall_score(y_test, pred_test)
```

### 정밀도-재현율의 trade-off
- **분류 결정 임계치 (Threshold)**
    - 모델이 어떤 클래스로 분류할지 결정하는 기준 값
    - 0과 1 사이의 값으로 이뤄져 임계치 값을 넘기는 확률의 경우에는 클래스 1을 아래는 0으로 분류해줌
- 분류 결정 임계치를 낮추면?
    - Positive로 예측할 확률이 올라간다
        - 정밀도는 낮아지고, 재현율이 높아진다
- 분류 결정 임계치를 높히면?
    - Positive로 예측할 확률이 낮아진다
        - 정밀도는 높아지고, 재현율이 낮아진다.

```python
from sklearn.preprocessing import Binarizer

temp_X = [[1,-1,2], [2,0,0], [0,1.1,1.2]]
binarizer = Binarizer(threshold=0)
adj_X = binarizer.fit_transform(temp_X)
adj_X
```

```python
# 정밀도 - 재현율 변화 과정 시각화

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt 
# thresholds에 따른 precision, recall 값 반환
precisions, recalls, thresholds = precision_recall_curve(y_test, predict_proba_1)

# 기준 thresholds 211 개
# - 정밀도와 재현율에 추가된 마지막 값은 극단적 임계값 설정에 대한 계산 결과
# - 정밀도는 마지막에 1 추가 (임계값이 매우 낮아 모든 샘플이 양성으로 에측된 경우)
# - 재현율은 마지막에 0 추가 (임계값이 매우 높아 모든 샘플이 음성으로 예측된 경우)
# precisions.shape, recalls.shape, thresholds.shape  

plt.figure(figsize=(6,4))
plt.plot(thresholds, precisions[: -1], linestyle = '--', label = 'Precision')
plt.plot(thresholds, recalls[: -1], label = 'Recall')
plt.xlabel('Thresholds')
plt.ylabel('Precision and Recall Values')
plt.legend()
plt.show()
```

---

## 교차검증 (Cross Validation, CV)
- 모델을 더욱 신뢰성 있게 평가하는 방법
- 데이터셋을 여러 개로 나누고, 각 부분이 한번씩 검증 데이터로 사용되도록 하는 방법
- 훈련-검증을 반복하면서 학습을 진행
- 과대적합 방지 효과
<img width=50% src="https://d.pr/i/0pWjyI+">

### K-fold
```python
from sklearn.datasets import load_iris

iris_input, iris_target = load_iris(return_X_y=True)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# 교차검증 KFold 객체 생성
lr_clf = LogisticRegression()
kfold = KFold(n_splits=5, shuffle=True, random_state=42) #n_splits : 폴드 개수, shuffles : 폴드로 나누기 전에 섞을건지 여부 (디폴트 : False)

# k번 반복하면서 평가한 정확도를 저장할 배열
cv_accuracy = []

for train_index,val_index in kfold.split(iris_input):
    X_train, y_train = iris_input[train_index], iris_target[train_index]
    X_val , y_val = iris_input[val_index], iris_target[val_index]

    print(np.unique)
    
    lr_clf.fit(X_train, y_train)    # 모델 학습
    y_pred = lr_clf.predict(X_val)  # 검증 데이터로 예측
    acc_score = accuracy_score(y_val, y_pred)   # 정확도 계산
    cv_accuracy.append(acc_score)   # cv_accuracy 배열에 정확도 추가
    
print('훈련별 정확도: ', cv_accuracy)
print('분류모델 정확도: ', np.mean(cv_accuracy))
    
```
- KFold 교차 검증을 통해 데이터셋을 여러 폴드로 나누고 이를 로지스틱회귀모델을 학습해 정확도를 계산
- 결과적으로, 각 폴드의 정확도와 전체 모델의 평균 정확도를 얻을수 있음

### **Stratified-K-Fold**
```python
# Stratified-K-Fold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# 교차검증 StratifiedKFold 객체 생성
lr_clf = LogisticRegression()
stratifiedKFold_kfold = StratifiedKFold(n_splits=5) #n_splits : 폴드 개수

# k번 반복하면서 평가한 정확도를 저장할 배열
cv_accuracy = []

for train_index,val_index in stratifiedKFold_kfold.split(iris_input,iris_target):
    X_train, y_train = iris_input[train_index], iris_target[train_index]
    X_val , y_val = iris_input[val_index], iris_target[val_index]

    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_val,return_counts=True))
    print('=================')
    
    lr_clf.fit(X_train, y_train)    # 모델 학습
    y_pred = lr_clf.predict(X_val)  # 검증 데이터로 예측
    acc_score = accuracy_score(y_val, y_pred)   # 정확도 계산
    cv_accuracy.append(acc_score)   # cv_accuracy 배열에 정확도 추가
    
print('훈련별 정확도: ', cv_accuracy)
print('분류모델 정확도: ', np.mean(cv_accuracy))
```

### cross_val_score
- 교차 검증을 통해 모델 성능을 평가하는 함수
- 내부적으로 지정한 횟수만큼 학습/검증을 나누어 반복 처리
```python
from sklearn.model_selection import cross_val_score

lr_clf = LogisticRegression(max_iter=100)

# 첫 번째 인자 : 모델
# 두 번째 인자 : 입력 데이터
# 세 번째 인자 : 라벨 데이터
# scoreing: 평가 지표 (accuracy, precisdion, recall, f1)
scores = cross_val_score(lr_clf,iris_input,iris_target,scoring='accurcy',cv=5)

print('훈련별 정확도: ', scores)
print('모델 정확도: ', np.mean(scores))
```

## Hyper Parameter Tuning
- hyper parameter : 모델 설정과 관려해 직접 지정할 수 있는 매개변수
- model parameter : 회귀계수(가중치), 절편 등 모델의 학습 대상이 되는 변수

### GridSearchCV
- 가능한 모든 하이퍼파라미터 조합을 시도하여 최적의 하이퍼파라미터를 찾는 방법
- 작동 방식
    - 미리 정의된 **하이퍼파라미터 값들의 "그리드"**를 사용하여 모든 조합을 테스트합니다.
- 장점
    - 모든 가능한 조합을 시도하므로 최적의 하이퍼파라미터 탐색 가능
- 단점
    - 시간 소모적
    - 하이퍼파라미터 공간이 큰 경우 매우 비효율적
```python
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# 데이터 로드
iris_input, iris_target = load_iris(return_X_y=True)

# 모델 생성
knn = KNeighborsClassifier()

# 테스트할 파라미터 값
params = {
    'n_neighbors': range(1, 13, 2)
}

# 첫 번째 인자: 모델
# 두 번째 인자: 테스트 할 파라미터 (딕셔너리)
# scoring: 평가 지표 (accuracy, precision, recall, f1)
# cv: 반복 횟수
grid = GridSearchCV(knn, params, scoring='accuracy', cv=5)
grid.fit(iris_input, iris_target)

print("최적의 파라미터:", grid.best_params_)
print("최적화된 모델 객체:", grid.best_estimator_)
print("최적화된 점수:", grid.best_score_)

best_knn = grid.best_estimator_
best_knn.fit(iris_input, iris_target)
best_knn.score(iris_input, iris_target)
```

### RandomSearchCV
 - 하이퍼 파라미터의 값 목록이나 값의 범위를 제공하는데, 이 범위 중에 랜덤하게 값을 뽑아내 최적의 하이퍼 파라미터 조합을 찾는다.
    - 탐색범위가 넓을 때 짧은 시간 내에 좋은 결과를 얻을 수 있다.
    - 랜덤하게 값을 추출해 계산하므로, 전역 최적값을 놓칠 수 있다.
- 장점
    - 효율적
    - 큰 하이퍼파라미터에서 적당히 좋은 값 탐색 가능
- 단점
    - 최적의 하이퍼파라미터가 보장 불가능
```python
from sklearn.model_selection import RandomizedSearchCV

# 모델 생성
knn = KNeighborsClassifier()

# 테스트할 파라미터 생성
params = {
    'n_neighbors': range(1, 100, 2)
}

# n_iter: 탐색할 최적의 하이퍼 파라미터 조합 수 (기본값: 10)
#         값이 크면 시간이 오래 걸림 / 값이 작으면 좋은 조합을 찾을 가능성 저하
rd_search = RandomizedSearchCV(knn, params, cv=5, n_iter=10, random_state=0)
rd_search.fit(iris_input, iris_target)

print("최적의 파라미터:", rd_search.best_params_)
print("최적화된 모델 객체:", rd_search.best_estimator_)
print("최적화된 점수:", rd_search.best_score_)
rd_search.cv_results_
```

---
## 회귀모델 평가
```python
# 샘플 데이터
y_true = [3, 0.5,2,7]
y_pred=[2.5,0,2,9]

from sklearn.metrics import mean_squared_error  # MSE(평균 제곱 오차)
from sklearn.metrics import root_mean_squared_error # RMSE(평균 제곱 오차 제곱근)
from sklearn.metrics import mean_absolute_error # MAE (평균 절대값 오차)
from sklearn.metrics import mean_squared_log_error  # MSLE(평균 제곱 로그 오차)
from sklearn.metrics import root_mean_squared_log_error # RMSLE (평균 제곱 로그 오차 제곱근)
from sklearn.metrics import r2_score    #R^2(결정 계수)

print(mean_squared_error(y_true, y_pred))
print(root_mean_squared_error(y_true,y_pred))
print(mean_absolute_error(y_true,y_pred))
print(mean_squared_log_error(y_true,y_pred))
print(root_mean_squared_log_error(y_true,y_pred))
print(r2_score(y_true,y_pred))
```