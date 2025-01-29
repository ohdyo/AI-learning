# 차원 축소
- 고차원 데이터를 저차원으로 변환하여 데이터를 간소화하는 과정
![PCA-LDA](https://d.pr/i/hI0WaD+)

## 특성 선택 (Feature Selection)
- 데이터를 구성하는 중요한 특성만 선택
    - 유의마한 변수 선택, 상관분석
## 특성 추출 (Feature Extraction)
- 기존 특성에서 새로운 저차원 특성을 생성하는 방법
    - PCA, LLE, t-SNE

---

### PCA(Principal Component Analysis)
- 데이터를 가장 잘 설명할 수 있는, 분산을 최대화하는 방향으로 축을 결정
- PCA의 주요 과정
    - 1. 데이터 정규화
        - 모든 특성을 동일한 스케일로 조정한다.
    - 2. 공분산 행렬 계산
        - 데이터의 분산과 상관관계 계산
    - 3. 고유백터와 고유값 계산
        - 공분산 행렬의 고유벡터와 고유값을 계산하여 주성분을 구함
    - 4. 주성분 선택 및 변환
        - 가장 큰 고유값에 해당하는 고유벡터를 선택하여 데이터를 저차원으로 투명한다.
```python
# PCA 주성분 분석
from sklearn.decomposition import PCA

pca = PCA(n_components=2)    # 2차원으로 축소

X = iris_df.iloc[:, :-1]    # 독립변수(특성 데이터)만 모아서 X

pca.fit(X)
iris_pca = pca.transform(X)    # 변환

iris_pca_df = pd.DataFrame(iris_pca, columns=['pca_col1', 'pca_col2'])
iris_pca_df['target'] = iris_data.target
iris_pca_df

# 축소한 차원으로 시각화
markers = ['^', 's', 'o']

for i, marker in enumerate(markers):
    x = iris_pca_df[iris_pca_df['target'] == i]['pca_col1']
    y = iris_pca_df[iris_pca_df['target'] == i]['pca_col2']
    plt.scatter(x, y, marker=marker, label=iris_data.target_names[i])

plt.legend()
plt.xlabel('pca_col1')
plt.ylabel('pca_col2')
plt.show()
```
- <cod>pca.explained_variance_ratio_</code>
    - 해당 코드를 통해서 PCA를 통해 생성된 주성분들의 실제 데이터의 총 분산에 얼마만큼의 비율을 보여주는지 확인 가능

```python
fruits_pca = pca.transform(fruits_1d)

# 주성분 비율
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())
plt.plot(pca.explained_variance_ratio_)
plt.show()

# 라벨 데이터 생성
target = np.array([0] * 100 + [1] * 100 + [2] * 100)
target.shape

# 원본 데이터 교차검증 점수 확인

# cross_val_score : 각 cv의 점수 반환
# cross_validate : 각 cv의 학습시간, 테스트시간, cv의 점수 반환 (평가지표 여러 개 사용 가능)
from sklearn.model_selection import cross_validate

lr_clf = LogisticRegression(max_iter=1000)
result = cross_validate(lr_clf, fruits_1d, target, cv=3)
result

# PCA 데이터 교차검증 점수 확인
result_pca = cross_validate(lr_clf, fruits_pca, target, cv=3)
result_pca
```
- <code>cross_validate()</code>
    - 필요로 하는 인자
        - 1. 학습 모델
        - 2. 평가할 데이터
        - 3. 분류 모델
        - 4. 폴드 교차 검증 횟수

### LDA (Linear Discriminant Analysis)
- ***타겟 클래스 간 분리를 최대로 하는 축***으로 결정
- 데이터의 분포에 영향을 받으므로 표준화 진행
```python
from sklearn.preprocessing import StandardScaler

iris_data = load_iris()

scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris_data.data)
iris_scaled

# LDA 변환
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
iris_lda = lda.fit_transform(iris_scaled, iris_data.target)

# LDA로 축소한 데이터 시각화 
iris_lda_df = pd.DataFrame(iris_lda, columns=['lda_col1', 'lda_col2'])
iris_lda_df['target'] = iris_data.target

markers = ['^', 's', 'o']

for i, marker in enumerate(markers):
    x = iris_lda_df[iris_lda_df['target'] == i]['lda_col1']
    y = iris_lda_df[iris_lda_df['target'] == i]['lda_col2']
    plt.scatter(x, y, marker=marker, label=iris_data.target_names[i])

plt.legend()
plt.xlabel('lda_col1')
plt.ylabel('lda_col2')
plt.show()
```

### LLE(Locally Linear Embedding)
- 데이터 포인트를 근접한 이웃과 선형 결합을 ㅗ표현하고 이를 유지하도록 저차원 공간에 매핑
```python
from sklearn.manifold import LocallyLinearEmbedding

iris_data = load_iris()

# LLE 변환
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
iris_lle = lle.fit_transform(iris_data.data)

# 시각화
iris_lle_df = pd.DataFrame(iris_lle, columns=['lle_col1', 'lle_col2'])
iris_lle_df['target'] = iris_data.target

markers = ['^', 's', 'o']

for i, marker in enumerate(markers):
    x = iris_lle_df[iris_lle_df['target'] == i]['lle_col1']
    y = iris_lle_df[iris_lle_df['target'] == i]['lle_col2']
    plt.scatter(x, y, marker=marker, label=iris_data.target_names[i])

plt.legend()
plt.xlabel('lle_col1')
plt.ylabel('lle_col2')
plt.show()

# LLE 변환 데이터 교차 검증 확인
result = cross_validate(lr_clf, iris_lle_df[['lle_col1', 'lle_col2']], iris_lle_df['target'], cv=3)
result

# LLE 속성
print(lle.n_neighbors)    # 이웃 수
print(lle.n_components)    # 축소된 차원수
print(lle.reconstruction_error_)    # 재구성 오차
```
