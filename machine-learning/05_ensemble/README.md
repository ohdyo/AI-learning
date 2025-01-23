## 앙상블(ensemble)
- 다양한 모델을 결합하여 예측 성능을 향상시키틑 방법
- 투표(voting), 배깅(Bagging), 부스팅(Boosting), 스태킹(stacking) 네 가지로 구분

##### 모든 예시의 데이터셋은 이걸 기준으로 한다.
```python
from sklearn.datasets import load_breast_cancer
data= load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.info()
```

### voting
- ***hard voting*** : 여러 개의 예측지에 대해 다수결로 결정
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
- 작동 원리는 다수결로 각각의 모델의 결과값 중 각 인덱스에 해당하는 값이 더 많은것을 가지고 예측한다.
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
<br>
=> 최종적으로 ***hard-voting은 모델이 predict의 예측값을 다수결을 통해서 앙상블의 예측값***으로 사용하는거고 ***soft-voting은 predict_proba를 통해서 얻은 확률의 평균을 가지고 앙상블의 확률***로 사용한다.

---

## Bagging
- Bootstrap Aggregation
- Bootstrap 방식의 샘플링 : 각 estimator 마다 훈련 데이터를 뽑을 때, ***중복 값을 허용***하는 방식
- 분류 모델의 경우, 각 tree(estimator)의 예측값을 다수결(hard voting)결정
- 회귀 모델의 경우, 각 tree(estimator)의 예측값을 평균내어 결정
- 기본적으로 100개의 tree 사용
    - n_estimators={num}
- 가질수 있는 여러 하이퍼 파라미터를 통해서 규제를 적용할수 있다.
    - 대부분이 트리 회귀 모델과 하이퍼 파라미터가 비슷하다.
    - class_weight = 클래스별로 가중치를 정해 불균형 데이터를 어떻게 처리할지 결정해주는 하이퍼 파라미터터
- 가장 대표적인 예시가 RandomForestClassifier이다.

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
- Boosting 계열은 일반적으로 결정트리 개수를 늘려도 과적합에 강함
- 대표적인 알고리즘(모델) : GradientBoosting, HistGradientBoosting, XGBoost,LightGBM, CatBoost