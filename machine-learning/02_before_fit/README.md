# EDA

## EDA 절차
1. 데이터 로드
    - 다양한 소스에서 데이터를 로드한다. (파일, 데이터베이스, API 등에서 데이터를 불러온다.)
    - 데이터가 로드된 후에는 구조를 확인하기 위해 첫 몇 줄의 데이터를 출력해 본다.
2. 데이터 구조 및 기초 통계 확인
    - 데이터의 구조를 파악하기 위해 컬럼 정보와 데이터 타입을 확인한다.
    - 데이터의 기본 통계 정보를 출력하여 각 변수의 분포와 특성을 살펴본다.
    - df.head(), df.info(), df.describe() 등의 함수를 사용한다.
3. 결측치 및 이상치 탐색
    - 결측치(NaN) 값이 존재하는지 확인하고, 이를 처리하는 방법을 결정한다.
    - 데이터에 존재하는 이상치(Outlier)를 탐지하고, 이를 어떻게 처리할지 결정한다.
    - df.isnull().sum(), df.boxplot() 등의 함수를 활용한다.
4. 데이터 시각화를 통한 탐색
    - 데이터를 시각화하여 변수 간의 관계, 분포 등을 파악한다.
    - 히스토그램, 박스플롯, 상관관계 행렬 등 다양한 그래프를 통해 데이터의 특성을 시각적으로 확인한다.
    - sns.countplot(), sns.heatmap() 등의 함수를 사용한다.
5. 데이터 정제 및 전처리
    - 필요 없는 변수나 중복 데이터를 제거한다.
    - 범주형 데이터를 처리하거나, 스케일링 및 정규화를 통해 모델에 적합한 형태로 데이터를 변환한다.
    - df.drop(), df.fillna(), pd.get_dummies() 등의 함수를 활용한다.
6. 데이터 변환 및 피처 엔지니어링
    - 새로운 피처를 생성하거나 기존 피처를 변환하여 분석에 적합한 형태로 데이터를 조정한다.
    - 로그 변환, 다항식 피처 추가 등 다양한 기법을 통해 데이터를 변환할 수 있다.
    - np.log(), PolynomialFeatures() 등의 함수를 활용한다.
7. 데이터 분할
    - 학습용과 테스트용 데이터로 분할한다.
    - 이 과정은 모델을 평가하고 성능을 검증하는 데 중요한 단계이다.
    - train_test_split() 함수를 사용한다.

---

**titanic.csv를 통한 예제**
    
***1. 데이터 로드***
```python
df = pd.read_csv('./data/titanic.csv')
```

***2. 데이터 구조 및 기초 통계 확인인***
```python
df.describe()
```

***3. 결측치 및 이상치 탐색***
```python
# 결측치 탐색
df.isnull().sum()
# 이상치 탐색
df['Age'].plot(kind='box') #1 박스형식으로 봐서 이상치 확인 가능
df[df['Age'] < 1] # 1살 아래의 이상치 탐색
```

***4. 데이터 시각화를 통한 탐색***
```python
sns.countplot(data=df, x='Survived', hue='Sex')
plt.show()
```
```python
df['Age'].hist(bins=5)
plt.show()
```
```python
# **상관관계의 특징**
# - 두 변수가 어떻게 함께 변화하는지를 나타냅니다. 
# - 한 변수가 변화할 때 다른 변수가 어떻게 변화하는지를 # 보여줍니다. 
# - 상관관계를 측정하는 개념으로는 상관 지수, 상관 계수 등이 있습니다. 
# - 상관관계의 방향에 따라 양의 상관관계와 음의 상관관계로 구분됩니다.
corr_matrix = df.corr(numeric_only=True)
corr_matrix

sns.heatmap(corr_matrix, annot=True)
```

***5. 데이터 정제 및 처리***
- 결측치에 대한 값을 드랍 혹은 평균값으로 대체
    - 학습에 영향을 미칠것 같으면 평균값으로 대체
    - 영향을 주지 않을것 같으면 삭제
    - 지금은 평균값으로 대체
```python
def fillna(df):
    """
    결측치 처리 함수
    - Age : 평균치로 대체
    - Cabin : 'N' 기본값으로 대체
    - Embarked : 'N' 기본값으로 대체
    """
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Cabin'] = df['Cabin'].fillna('N')
    df['Embarked'] = df['Embarked'].fillna('N')

    return df
```
- 필요없는 행에 대한 값을 삭제한다.
```python
def drop_feature(df):
    """
    모델 훈련과 관련 없는 속성 제거
    - PassengerId, Name, Ticket
    """
    return df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
```
- 중복되는 의미를 가진 컬럼에 대한 값들을 병합 후 새 행에 값을 삽입하고 기존 행을을 제거한다.
```python
df['Family'] = df['SibSp'] + df['Parch']
df = df.drop(['SibSp','Parch'],axis=1)
```
- 편향된 데이터를 목적에 맞게 사용하기 위해 x의 값을 변형시켜준다.
```python
df['LogFare'] = df['Fare'].apply(lambda x: np.log1p(x))
df['LogFare'].hist()
```
- 데이터를 컴퓨터가 알수있게 **범주형 데이터** 파악하고 0과 1로 인코딩 해준다.
```python
from sklearn.preprocessing import LabelEncoder
def encode_feature(df):
    """
    범주형 데이터를 숫자로 인코딩
    - Sex, Cabin, Embarked
    """
    df['Cabin'] = df['Cabin'].str[:1]    # Cabin 데이터의 앞 글자만 가져옴

    categories = ['Sex', 'Cabin', 'Embarked']
    for cate_item in categories:
        label_encoder = LabelEncoder()
        df[cate_item] = label_encoder.fit_transform(df[cate_item])
```
- 해당 전처리함수들을 모두 불러오는 함수 호출
```python
from sklearn.preprocessing import StandardScaler
def preprocess_data(df):
    df = drop_feature(df)
    df = fillna(df)
    df = encode_feature(df)

    return df
df = preprocess_data(df)
```

***7. 훈련-테스트 데이터 분리***
- 학습을 위해 원하는 데이터를 가지고 분할한다.
```python
from sklearn.model_selection import train_test_split

# 입력-라벨 데이터 분리
titanic_input = df.drop(['Survived'], axis=1)
titanic_label = df['Survived']

X_train, X_test, y_train, y_test = \
    train_test_split(titanic_input, titanic_label, test_size=.2, random_state=0)
```
***8. 특성 스케일링***
```python
X_scaled_train, X_scaled_test = scailing_feature(X_train, X_test)
```

***9. LogisticRegression 훈련***
```python
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression()
lr_classifier.fit(X_scaled_train, y_train)
```

***10. 평가***
```python
lr_classifier.score(X_scaled_train, y_train), \
    lr_classifier.score(X_scaled_test, y_test)
```
---

---

# 전처리 (preprocessing)
- Data cleansing
- Data Encoding : 텍스트 데이터 -> 숫자로 변환 (범주형 데이터)
- Data Scaling : 숫자값 정규화
- Outlier : 이상치
- Feature Engineering : 속성 생성/수정/가공

---
## Data Encoding

### Label Encoder
- 범주형 데이터에 대해 적절히 숫자로 변환하는 것
- 텍스트를 값으로 갖는 컬럼을 숫자로 변환할 때 사용
- 중복값을 제거하고 오름차순으로 정렬해준다.

```pytnon
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
# 학습이랑은 좀 다른 느낌
# item의 값들을 비교해서 다른 값이 나올때 마다 인덱스로 추가됨
encoded_items = encoder.fit_transform(items)
encoded_items
```

---

### One-hot Encoder
- 주어진 데이터를 희소배열로 변환 (One-vs-Reset 배열)
- 희소 배열이란 대부분이 0이고 특정 인덱스만 값을 가지고 있는 배열
- 결과 값이 2차원 배열이기에 넘겨주는 값도 2차원 배열이어야한다.

```python
from sklearn.preprocessing import OneHotEncoder
items = np.array(items).reshape(-1,1)
encoder = OneHotEncoder()
# 중복값을 제거, 오름차순 정렬 -> 그 인덱스에만 1을 준 희소행렬
oh_items = encoder.fit_transform(items)
print(oh_items.toarray())
```
- **DataFrame에서 One-hot encoding 하기**
```python
# 데이터 타입 명시 안할시, True(1) | False(0) 로 표현
df_dummies=pd.get_dummies(df, dtype=int)
# 2차원 배열로 바꿔줌 np.array(df_dummies) 와 같음
nd_dummies = df_dummies.to_numpy()
```

---

## Data Scaling(Feature Scaling)
- Scaling 작업은 train 데이터, test 데이터에 동일하게 적용해야 함
    - fit() : train 데이터
    - transform() : train 데이터, test 데이터터
- 배우기 전 미리 데이터 셋팅
```python
from sklearn.datasets import load_iris
iris_data = load_iris()
iris_data.data
```

### 표준 정규화(StandardScaler)
- 평균이 0, 표준편차가 1인 값으로 변환
- 이상치에 덜 민감하고, 선형회귀 및 로지스틱 회귀 등의 알고리즘에 적합
- 데이터가 정규분포인 경우 더욱 적합함
```python
from sklearn.preprocessing import StandardScaler
standard_sc = StandardScaler()
standard_sc.fit_transform(iris_data.data)
```

### 최소최대 정규화(MinMaxScaler)
- 0~1 사이의 값으로 변환
- SVM 및 KNN과 같은 거리 기반 모델에 적합
- 이상치에 민감하게 반응, 이상치가 있는경우 데이터 왜곡 가능성x
```python
from sklearn.preprocessing import MinMaxScaler
minmax_sc = MinMaxScaler()
minmax_sc.fit_transform(iris_data.data) #(값 - 최소값) / (최대값 - 최소값)
```


