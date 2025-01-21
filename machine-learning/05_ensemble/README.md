## 앙상블(ensemble)
- 다양한 모델을 결합하여 예측 성능을 향상시키틑 방법
- 투표(voting), 배깅(Bagging), 부스팅(Boosting), 스태킹(stacking) 네 가지로 구분

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
    
    class_name = classfier.__class__.__name__ # 클래스의 이름 메타데이터 가져옴옴
    print(f'{class_name} 개별 정확도: {acc_score:.4f}')
    print(f'{class_name} 예측값 : {pred[start:end]}')
```



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
## Bagging
- Bootstrap Aggregation
- Bootstrap 방식의 샘플링 : 각 estimator 마다 훈련 데이터를 뽑을 때, 중복 값을 허용하는 방식
- 분류 모델의 경우, 각 tree(estimator)의 예측값을 다수결(hard voting)결정
- 회귀 모델의 경우, 각 tree(estimator)의 예측값을 평균내어 결정
- 기본적으로 100개의 tree 사용
- 가질수 있는 여러 하이퍼 파라미터를 통해서 규제를 적용할수 있다.
```python
from sklearn.ensemble import RandomForestClassifier

rt_clf = RandomForestClassifier(n_estimators=100,max_depth=5, random_state=0)
# 학습
rt_clf.fit(X_train,y_train)

y_pred_train = rt_clf.predict(X_train)
acc_score_train = accuracy_score(y_train, y_pred_train)
print('학습 점수: ', acc_score_train)

y_pred_test = rt_clf.predict(X_test)
acc_score_test = accuracy_score(y_test, y_pred_test)
print('테스트 평가 점수 : ', acc_score_test)
```