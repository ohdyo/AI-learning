## 앙상블(ensemble)
- 다양한 모델을 결합하여 예측 성능을 향상시키틑 방법
- ***투표(voting), 배깅(Bagging), 부스팅(Boosting), 스태킹(stacking)*** 네가지로 구분

##### 모든 예시의 데이터셋은 이걸 기준으로 한다.
```python
from sklearn.datasets import load_breast_cancer
data= load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.info()
```

### voting
- #### hard voting : 여러 개의 예측지에 대해 다수결로 결정
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

knn_clf = KNeighborsClassifier()
lr_clf = LogisticRegression()
dt_clf = DecisionTreeClassifier()

voting_clf = VotingClassifier(
    estimators=[
        ('knn_clf', knn_clf),
        ('lr_clf', lr_clf),
        ('dt_clf', dt_clf)
    ]
)

#앙상블 모델 학습
voting_clf.fit(X_train,y_train)

y_pred_train = voting_clf.predict(X_train)
acc_score_train = accuracy_score(y_train, y_pred_train)
print('학습 점수 : ', acc_score_train)

y_pred_test = voting_clf.predict(X_test)
acc_score_test = accuracy_score(y_test, y_pred_test)
print('테스트  평가 점수 : ' , acc_score_test)
```
- 선언한 votingClassifier의 인자로 voting= 뒤에 쓰는 문자열에 따라 soft인지 hard인지 결정
- 작동 원리는 다수결로 각각의 모델의 결과값(=pred) 중 각 인덱스에 해당하는 값이 더 많은것을 가지고 예측한다.
```python
# hard votin 작동 원리 == 다수결
start, end = 40,50
voting_clf_pred = voting_clf.predict(X_test[start:end])


for classfier in [knn_clf,lr_clf,dt_clf]:
    #개별 학습 및 예측
    classfier.fit(X_train,y_train)
    pred = classfier.predict(X_test)
    acc_score = accuracy_score(y_test, pred)
    
    class_name = classfier.__class__.__name__ # 클래스의 이름 메타데이터 가져옴
    print(f'{class_name} 개별 정확도: {acc_score:.4f}')
    print(f'{class_name} 예측값 : {pred[start:end]}')
```
- hard voting의 경우 예측 결과값을 가져와 주는 predict를 통해 합친 모델들의 ***predict값을 확인***해서 더 많이 값을 출력한 인덱스를 앙상블 모델의 결과값으로 출력함
<br>
<br>
- ***soft voting*** : 여러 개의 예측 확률을 평균내어 결정
    - 여러개의 모델을 동시에 데이터를 입력하여 받은 값의 각각의 속성들의 평균을 가지고 예측하는 방법
    - pred_proba를 통해서 값을 예측해서 가져온 평균을 가지고 soft voting의 값을 증명 가능하다.
```python
# hard votin 작동 원리 == 다수결
start, end = 40,50
voting_clf_pred_proba = voting_clf.predict_proba(X_test[start:end])

averages = np.full_like(voting_clf_pred_proba, 0)


for classfier in [knn_clf,lr_clf,dt_clf]:
    #개별 학습 및 예측
    classfier.fit(X_train,y_train)
    pred = classfier.predict(X_test)
    acc_score = accuracy_score(y_test, pred)
    pred_proba = classfier.predict_proba(X_test[start:end])
    
    averages += pred_proba
    
    class_name = classfier.__class__.__name__ # 클래스의 이름 메타데이터 가져옴옴
    # print(f'{class_name} 개별 정확도: {acc_score:.4f}')
    # print(f'{class_name} 예측값 : {pred_proba}')
    
calc_averages = averages / 3
print('각 모델별 예측값 평균 : \n',calc_averages)
print(np.array_equal(voting_clf_pred_proba, calc_averages))
```
- soft voting의 경우 predict값으로 투표를 진행하는게 아니라 확률의 평균의 경우 ***predict_proba의 개별 정확도***를 가져와 각 모델의 정확도의 평균을 가지고 값을 예측한다.

```python
# soft voting 작동 원리 == 각 예측기의 확률값 평균

start, end = 40, 50

voting_clf_pred_proba = voting_clf.predict_proba(X_test[start:end])
print('앙상블 예측값:', voting_clf_pred_proba)

averages = np.full_like(voting_clf_pred_proba, 0)

for classfier in [knn_clf, lr_clf, dt_clf]:
    # 개별 학습 및 예측
    classfier.fit(X_train, y_train)
    pred = classfier.predict(X_test)
    acc_score = accuracy_score(y_test, pred)
    pred_proba = classfier.predict_proba(X_test[start:end])

    # 예측 확률 평균을 위한 합계
    averages += pred_proba
    
    class_name = classfier.__class__.__name__    # 클래스의 이름 속성
    # print(f'{class_name} 개별 정확도: {acc_score:.4f}')
    # print(f'{class_name} 예측 확률: {pred_proba}')

# 예측 확률 평균 계산 및 출력
calc_averages = averages / 3
print("각 모델별 예측값 평균:", calc_averages)
print(np.array_equal(voting_clf_pred_proba, calc_averages))
```
<br>
=> 최종적으로 ***hard-voting은 모델이 predict의 예측값을 다수결을 통해서 앙상블의 예측값***으로 사용하는거고 ***soft-voting은 predict_proba를 통해서 얻은 확률의 평균을 가지고 앙상블의 확률***로 사용한다.

---

## Bagging
- Bootstrap Aggregation
- Bootstrap 방식의 샘플링 : 각 estimator 마다 훈련 데이터를 뽑을 때, ***중복 값을 허용***하는 방식
<img src= "https://mblogthumb-phinf.pstatic.net/20141205_226/muzzincys_1417764856096vPrIa_PNG/%B1%D7%B8%B21.png?type=w2">
- 분류 모델의 경우, 각 tree(estimator)의 예측값을 다수결(hard voting)결정
- 회귀 모델의 경우, 각 tree(estimator)의 예측값을 평균내어 결정
- 기본적으로 100개의 tree 사용
    - n_estimators={num}
- 가질수 있는 여러 하이퍼 파라미터를 통해서 규제를 적용할수 있다.
    - 대부분이 트리 회귀 모델과 하이퍼 파라미터가 비슷하다.
    - class_weight = 클래스별로 가중치를 정해 불균형 데이터를 어떻게 처리할지 결정해주는 하이퍼 파라미터터
- 가장 대표적인 예시가 ***RandomForestClassifier***이다.

**하이퍼 파라미터**
| **하이퍼파라미터**      | **설명**                                                                                     | **기본값**      |
|--------------------------|--------------------------------------------------------------------------------------------|-----------------|
| `n_estimators`           | 생성할 트리의 개수 지정 (트리의 개수가 많을수록 성능이 좋아질 수 있지만 계산 비용 증가) | 100             |
| `criterion`              | 분할 품질을 측정하는 기준 (분류에서는 "gini" 또는 "entropy"를 사용)                 | "gini"          |
| `max_depth`              | 각 트리의 최대 깊이 (설정하지 않으면 트리는 잎 노드가 순수해질 때까지 계속 확장) | None            |
| `min_samples_split`      | 내부 노드를 분할하기 위해 필요한 최소 샘플 수 (과적합 방지 목적)                   | 2               |
| `min_samples_leaf`       | 잎 노드가 되기 위해 필요한 최소 샘플 수 (과적합 방지 목적)                          | 1               |
| `max_features`           | 각 트리를 분할할 때 고려할 최대 특성 수 ()"auto", "sqrt", "log2" 중 선택하거나, 특정 숫자 지정 가능) | "auto"          |
| `bootstrap`              | 각 트리를 만들 때 부트스트랩 샘플링을 사용할지 여부를 결정                               | True            |
| `random_state`           | 결과의 재현성을 위해 난수 시드 고정                                                  | None            |
| `n_jobs`                 | 병렬 계산을 위해 사용할 CPU 코어 수를 지정 (-1로 설정하면 모든 코어를 사용)           | None            |
| `class_weight`           | 각 클래스의 가중치를 자동으로 계산하거나 직접 지정 가능 (불균형 데이터 처리에 유용)    | None            |


```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100,max_depth=5, random_state=0)
# 학습
rf_clf.fit(X_train,y_train)

y_pred_train = rf_clf.predict(X_train)
acc_score_train = accuracy_score(y_train, y_pred_train)
print('학습 점수: ', acc_score_train)

y_pred_test = rf_clf.predict(X_test)
acc_score_test = accuracy_score(y_test, y_pred_test)
print('테스트 평가 점수 : ', acc_score_test)

# 100개의 DecisionTree 확인 
# print(rf_clf.estimators_)

# 100개의 DecisionTree가 사용한 샘플 데이터 확인
# print(rf_clf.estimators_samples_)

# Bootstrap 방식의 샘플링 확인인
for i, sample_indexs in enumerate(rf_clf.estimators_samples_):
    print(sample_indexs.shape)
    print(f'{i}번째 DecisionTree의 샘플 인덱스 : {sorted(sample_indexs)}')

# Series로 각 feature의 중요도 확인
feat_imptc_sr = pd.Series(rf_clf.feature_importances_, index=data.feature_names)\
    .sort_values(ascending=False)
feat_imptc_sr

# feature별 중요도를 시각화해서 확인해보기
plt.figure(figsize=(8,6))
sns.barplot(
    x=feat_imptc_sr,
    y=feat_imptc_sr.index,
    hue=feat_imptc_sr.index
)
plt.xlabel('feature importance')
plt.ylabel('feature')
plt.show()
```
- 중요하게 자주 볼 항목들
    - feature_importances
        - 학습할 때 사용한 피쳐별 중요도(확률값)을 파악 가능하다.

---
## Boosting
- 깊이가 얕은 결정트리를 사용해 이전 트리의 오차를 보정하는 방식
- 순차적으로 경사하강법을 사용해 이전 트리의 오차를 줄여나감
    - 분류 모델에서는 손실함수 Logloss를 사용해 오차를 줄임
    - 회귀모델에서는 손실함수 MSE를 사용해 오차를 줄임
<img src="https://mblogthumb-phinf.pstatic.net/20141205_289/muzzincys_1417764856320XKWT7_PNG/%B1%D7%B8%B22.png?type=w2">
- Boosting 계열은 일반적으로 결정트리 개수를 늘려도 과적합에 강함
- 대표적인 알고리즘(모델) : GradientBoosting, HistGradientBoosting, XGBoost,LightGBM, CatBoost

### GradientBoosting
- RandomForest 모델과 달리 ***learning_rate***를 통해 오차를 줄여나가는 방식
```python
# GradientBoostingClassifier로 유방암 데이터 예측
from sklearn.ensemble import GradientBoostingClassifier

# 데이터 로드 및 분리
data = load_breast_cancer()
X_train, X_test, y_train, y_test = \
    train_test_split(data.data, data.target, random_state=0)

# 모델 생성
gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.01,
    max_depth=3
)

# 학습
gb_clf.fit(X_train, y_train)

# 예측 및 평가
y_pred_train = gb_clf.predict(X_train)
y_pred_test = gb_clf.predict(X_test)
print(f'학습 정확도: {accuracy_score(y_train, y_pred_train)}')
print(f'평가 정확도: {accuracy_score(y_test, y_pred_test)}')
```

### HistGradientBoosting
- 고성능 모델로 대규모 데이터셋 처리에 적합
- 결측치가 존재해도 전처리가 필요 없음
- LigtGBM의 영향을 받아 만들어진 scikit-learn의 모델

#### HistGradinetBoostingClassifier
    - 분류 모델에 적용시키는 Boosting기법
```python
from sklearn.ensemble import HistGradientBoostingClassifier

data = load_breast_cancer()
X_train, X_test, y_train, y_test = \
    train_test_split(data.data, data.target, random_state=42)

hist_gb_clf = HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=3,
    max_bins=255,    # 255개의 구간으로 나누어 처리 (1개는 결측치 전용)
    early_stopping=True,    # 반복 중 '일정 횟수' 이상 성능 향상이 없으면 학습 종료
    n_iter_no_change=5      # '일정 횟수' 지정 (기본값: 10)
)
hist_gb_clf.fit(X_train, y_train)

y_pred_train = hist_gb_clf.predict(X_train)
y_pred_test = hist_gb_clf.predict(X_test)
print(f'학습 정확도 : {accuracy_score(y_train, y_pred_train)}')
print(f'평가 정확도 : {accuracy_score(y_test, y_pred_test)}')
```

- #### HistGradientBoostingRegressor
    - 회귀모델에 적용시키는 Boosting 기법
```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

hist_gb_reg = HistGradientBoostingRegressor(
    max_iter=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=0,
    l2_regularization=0.5,
    min_samples_leaf=5
)
hist_gb_reg.fit(X_train, y_train)

y_pred_train = hist_gb_reg.predict(X_train)
y_pred_test = hist_gb_reg.predict(X_test)

print(f'학습 MSE: {mean_squared_error(y_train, y_pred_train)} | 학습 R2: {r2_score(y_train, y_pred_train)}')
print(f'평가 MSE: {mean_squared_error(y_test, y_pred_test)} | 평가 R2: {r2_score(y_test, y_pred_test)}')
```

##### GrideSearchCV
    - 최고의 성능을 내는 하이퍼 파라미터 탐색 모듈
    - GradientBoosting을 통해 학습된 모델을 인자로 받아 탐색
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_iter': [100, 200, 300],
    'max_depth': [1, 3, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_samples_leaf': [10, 20, 30],
    'l2_regularization': [0.0, 0.1, 1.0],
    'max_bins': [255, 127]
}

hist_gb_reg = HistGradientBoostingRegressor(random_state=0)
grid_search = GridSearchCV(hist_gb_reg, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

grid_search.best_params_    # 최고의 성능을 내는 하이퍼 파라미터
```

### XGBoosting
    - L1 과 L2 규제를 적용시킨 GradientBoosting의 개선된 버전
    - 학습 과정에서 손실함수 최소화
    - 결측값 자동 처리
**핵심 파라미터**
1. **learning_rate**: 각 트리의 기여도를 조절하는 학습률로, 값이 작을수록 모델의 복잡도가 낮아지지만 더 많은 트리를 필요로 한다.
2. **n_estimators**: 트리의 개수를 의미하며, 많을수록 복잡한 모델이 된다.
3. **max_depth**: 각 트리의 최대 깊이로, 트리가 너무 깊으면 과적합될 수 있다.
4. **objective**: 손실 함수의 종류로, 회귀 문제의 경우 'reg:squarederror', 분류 문제의 경우 'binary:logistic' 등을 사용한다.

#### XGBClassfier
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier    # xgboost에서는 sklearn과 비슷한 api 제공

iris_data = load_iris()
X_train, X_test, y_train, y_test = \
    train_test_split(iris_data.data, iris_data.target, random_state=0)

xgb_clf = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=0
)

xgb_clf.fit(X_train, y_train)

y_pred_train = xgb_clf.predict(X_train)
y_pred_test = xgb_clf.predict(X_test)

print(accuracy_score(y_train, y_pred_train))
print(accuracy_score(y_test, y_pred_test))

print(classification_report(y_test, y_pred_test))
```

#### LightGBM
- Leaf-wise는 가장 큰 손실을 줄일 수 있는 리프를 먼저 확장하므로, 더 낮은 손실을 가진 복잡한 트리를 생성
- 과적합의 위험이 있기 때문에 max_depth 같은 파라미터로 제어
<img src="https://velog.velcdn.com/images/chlwldns00/post/73384f09-3d3d-433f-ab69-f3740b40d36b/image.PNG">
**핵심 파라미터**
<table border="1">
  <thead>
    <tr>
      <th>하이퍼파라미터</th>
      <th>설명</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>num_leaves</b></td>
      <td>한 트리에서 사용할 수 있는 리프의 최대 수를 지정한다. 모델의 복잡도를 결정하며, 값을 크게 하면 과적합(overfitting) 가능성이 높아진다.</td>
    </tr>
    <tr>
      <td><b>max_depth</b></td>
      <td>트리의 최대 깊이를 제한한다. <code>num_leaves</code>와 함께 과적합을 방지하기 위해 조정된다.</td>
    </tr>
    <tr>
      <td><b>learning_rate</b></td>
      <td>각 단계에서 트리의 기여도를 조정하는 학습률이다. 작은 값을 설정하면 모델 학습이 느리지만 성능이 더 좋을 수 있다.</td>
    </tr>
    <tr>
      <td><b>n_estimators</b></td>
      <td>생성할 트리의 수를 지정한다. 보통 <code>learning_rate</code>가 작을수록 큰 값을 설정한다.</td>
    </tr>
    <tr>
      <td><b>min_data_in_leaf</b></td>
      <td>리프 노드에서 최소 데이터 수를 제한하여 과적합을 방지한다.</td>
    </tr>
    <tr>
      <td><b>feature_fraction</b></td>
      <td>각 트리를 학습할 때 사용할 피처의 비율을 지정한다. 이 값을 줄이면 피처 샘플링 효과를 얻을 수 있다.</td>
    </tr>
    <tr>
      <td><b>bagging_fraction & bagging_freq</b></td>
      <td>데이터 샘플링을 통한 앙상블 효과를 얻기 위한 옵션으로, 일부 데이터만을 사용해 트리를 학습한다.</td>
    </tr>
    <tr>
      <td><b>lambda_l1 & lambda_l2</b></td>
      <td>L1 및 L2 정규화 항을 추가하여 모델의 가중치를 제한한다.</td>
    </tr>
    <tr>
      <td><b>boosting</b></td>
      <td>Boosting의 종류를 지정할 수 있다. 일반적으로 <code>gbdt</code>(Gradient Boosting Decision Tree)를 사용하지만, <code>dart</code>(Dropouts meet Multiple Additive Regression Trees)나 <code>goss</code>(Gradient-based One-Side Sampling)도 선택할 수 있다.</td>
    </tr>
  </tbody>
</table>

```python
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.7,
    early_stopping_rounds=100,
    verbose=1
)
eval_set = [(X_tr, y_tr), (X_val, y_val)]
lgbm.fit(X_tr, y_tr, eval_set=eval_set)
```
