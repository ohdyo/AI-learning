# 군집 (Clustering)
- 비지도 학습 알고리즘의 한 종류
- 레이블이 없는 데이터를 유사성에 따라 그룹을 나누는데 사용
- 군집은 데이터의 내재된 구조를 파악하거나, 탐색적 데이터 분석(EDA)에 사용
- **군집의 목적**
    - 데이터의 그룹화
    - 데이터 분포 이해
    - ***노이즈 제거***
    - 새로운 데이터의 레이블 생성
- 군집과 준류의 차이
    - 군집 : 레이블이 없는 데이터를 그룹화 (비지도 학습)
    - 분류 : 이미 정의된 레이블에 데이터 매핑 (지도 학습)
- **군집 알고리즘 종류**
    1. K-평균 군집(K-Means Clustering)
    2. 계측정 군집(Hierachical Clustering)
    3. DBSCAN (Density-Based Clustering of Applications with Noise)

## K-평균 군집(K-Means Clustering)
- K개의 그룹(중심점을 기준)으로 데이터 포인트를 나눔
![](https://upload.wikimedia.org/wikipedia/commons/e/ea/K-means_convergence.gif?20170530143526)
- 작동 단계
    1. K개의 중심점 임의 선택
    2. 각 데이터 포인트를 가장 가까운 중심점에 할당 > 군집 형성
    3. 각 군집의 데이터 포인트 기반으로 새로운 중심점 계산
    4. 2~3 단계를 중짐점의 변화가 거의 없을 떄까지 반복
- **장단점**
    - 장점
        - 간단한 개념과 구현
        - 빠른 계산 속도
        - 일반적인 군집화에서 가장 많이 활용되며 대용량 데이터에도 활용 가능
    - 단점
        - 거리 기반 알고리즘으로 속성의 개수가 매우 많을 경우 군집화 정확도 떨어짐
        - 반복적으로 수행하므로 반복 횟수가 많아지면 수행 시간 느려짐
        - 이상치(outlier) 데이터에 취약
        - 군집이 원형 구조가 아닐 경우 성능 저하 가능성

```python
# 이미지 시각화 함수
def draw_fruits(arr, ratio=1):
    N = len(arr)
    rows = int(np.ceil(N / 10))
    cols = N if rows < 2 else 10
    fig, ax = plt.subplots(rows, cols, figsize=(cols * ratio, rows * ratio), squeeze=False)
    
    for i in range(rows):
        for j in range(cols):
            if i * 10 + j < N:
                ax[i, j].imshow(arr[i * 10 + j], cmap='gray_r')
            ax[i, j].axis('off')
    
    plt.show()

# KMeans 군집 적용을 위한 reshape
fruits_1d = fruits.reshape(-1, 100 * 100)

# KMeans 적용
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(fruits_1d)
kmeans.transform(fruits_1d)

# kmeans.labels_
np.unique(kmeans.labels_, return_counts=True)

# 각 클러스터별 이미지 시각화
draw_fruits(fruits[kmeans.labels_ == 0])
draw_fruits(fruits[kmeans.labels_ == 1])
draw_fruits(fruits[kmeans.labels_ == 2])

# 중심점 시각화
draw_fruits(kmeans.cluster_centers_.reshape(-1, 100, 100), ratio=3)

# PCA 적용 후 클러스터링
pca = PCA(n_components=2)
fruits_pca = pca.fit_transform(fruits_1d)
fruits_pca.shape

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(fruits_pca)
kmeans.labels_

np.unique(kmeans.labels_, return_counts=True)

draw_fruits(fruits[kmeans.labels_ == 0])
draw_fruits(fruits[kmeans.labels_ == 1])
draw_fruits(fruits[kmeans.labels_ == 2])

pred = kmeans.predict(fruits_pca[100:101])
print(pred)
draw_fruits(fruits[100:101])   # 파인애플 == 0 클러스터
```
- <code>KMeans(n_cluster=3)</code>
    - n_clutser
        - 군집의 갯수를 위한 변수

### 최적의 k 값 찾기
- inertia : 중심점으로부터 각 데이터포인트의 분산값
    - 이너셔 값이 작을수록 군집이 잘 되어 있다고 볼 수 있음

- Elbow 기법 : inertia값이 급격히 감소하는 k값을 최적의 k 값으로 판단
```python
inertias = []

for k in range(2, 12):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(fruits_pca)
    inertias.append(kmeans.inertia_)    # inertia 속성 확인 가능

inertias

plt.plot(range(2, 12), inertias)
plt.xlabel('K')
plt.ylabel('inertia')
plt.show()
```

## 군집 평가 (Silhouette Score)
- ***실루엣 계수***를 통해 군집화의 품질 평가
- 실루엣의 계수는 **-1에서 1 사이의 값**
    - 1에 가까울수록 군집도가 좋음 (다른 군집과 잘 분리)
    - 0은 군집의 경계에 위치
    - -1은 다른 군집과 겹치거나 잘못 분류된 경우
- **주요 속성**
- silhouette_samples: 개별 데이터 포인트의 점수
- silhouette_score: 전체 데이터포인트의 평균값
```python
# 데이터 로드
from sklearn.datasets import load_iris

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

# KMeans 군집화
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100, random_state=0)
kmeans.fit(iris.data)    # 입력 데이터를 통해 중심점 계산
iris_df['cluster'] = kmeans.labels_

# 실루엣 계수 측정
from sklearn.metrics import silhouette_samples, silhouette_score

sil_samples = silhouette_samples(iris.data, kmeans.labels_)
# sil_samples.shape
iris_df['sil_score'] = sil_samples

# 전체 클러스터의 실루엣 계수 = 개별 데이터 포인트의 실루엣 계수 평균
sil_score = silhouette_score(iris.data, kmeans.labels_)
sil_score, iris_df['sil_score'].mean()

# k값(클러스터 개수)별 실루엣 계수 시각화
def visualize_silhouette(n_clusters, X):
    """
    :param n_clusters: [2, 3, 4, 5] 테스트할 k값 목록 
    :param X: 입력데이터
    :return: 
    """
    import matplotlib.cm as cm
    
    # k개수
    n_cols = len(n_clusters)
    # subplot 생성
    fig, axs = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    for index, k in enumerate(n_clusters):
        # 군집 
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(X) # 중심점 계산 및 클러스터 결과 반환
        # 실루엣 계수
        sil_samples = silhouette_samples(X, labels)
        sil_score = silhouette_score(X, labels)
        
        # y축 범위
        y_lower = 10
        
        # plot 그리기
        for i in range(k): # range(2) = 0, 1  range(3) = 0, 1, 2
            ith_cluster_sil_samples = sil_samples[labels == i] # 이번 클러스터에 실루엣 계수 필터링
            ith_cluster_sil_samples.sort() # inplace 연산
            
            # 크기(영역) 계산
            ith_size = ith_cluster_sil_samples.shape[0]
            y_upper = y_lower + ith_size
            
            # 색상 지정
            color = cm.nipy_spectral(float(i) / k) # 클러스터 별 고유한 색상
            # print(color) # (r, g, b, a)
            axs[index].fill_betweenx(
                np.arange(y_lower, y_upper),    # y축 범위
                0,                              # x축1
                ith_cluster_sil_samples,        # x축2
                facecolor=color,                # 색상
                edgecolor=color,                # 테두리 색상
                alpha=0.7                       # 투명도
            )
            # 텍스트 추가
            axs[index].text(-0.05, y_lower + 0.5 * ith_size, str(i))

            y_lower = y_upper + 10 # 다음 차례의 아래경계 계산
        
        # 전체 실루엣 계수
        axs[index].axvline(x=sil_score, color='red', linestyle='--')
        
        # x, y축 라벨
        axs[index].set_xlabel('Silhouetter Coefficient')
        axs[index].set_ylabel('Cluster')
        
        # axis별 제목
        axs[index].set_title(f'K={k}')
        
        # x, y tick조정
        axs[index].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        axs[index].set_yticks([])

visualize_silhouette([2, 3, 4, 5], iris.data)   
```

## 가우시안 혼합_GMM (Gaussian Mixture Model)
- 데이터가 여러 개의 가우시안 분포(= 정규 분포)로 구성된다 가정, 이를 기반으로 군집을 수행하는 비지도 학습 알고리즘
- **GMM과 K-평균의 차이**
    - K-평균 : 각 데이터가 **특정 클러스터에 완전히 속한다 가정**
    - GMM : 각 데이터가 **특정 클러스터에 속활 확률을 기반으로 군집화**
- 가우시안 혼합의 학습
    - ***기대 최적화(Expectation-Maximization,EM)***알고리즘을 사용하여 학습
- GMM의 장단점
    - 장점
        - 클래스터 간 경계를 확률적으로 표현하여 더 유연
        - 데이터가 가우시안 분포를 따른다면 더 높은 성능을 보임
    - 단점
        - 가우시안 분포 가정이 부적절한 경우 성능 저하
        - EM 알고리즘은 초기값에 민감, 지역 최적점에 빠질수 있음
```python
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# 데이터 생성
X, _ = make_classification(
    n_samples=300,           # 데이터 개수
    n_features=2,            # 특성 개수
    n_informative=2,         # 유의미한 특성 개수
    n_redundant=0,           # 중복 특성 개수
    n_clusters_per_class=1,  # 클래스 당 클러스터 수
    n_classes=2,             # 클래스(레이블) 개수
    random_state=42
)

# 생성 데이터 시각화
plt.scatter(X[:, 0], X[:, 1], s=50, c='gray', marker='o', edgecolors='k')
plt.title('scikit-learn making data')
plt.show()

# GMM 적용 및 군집화 결과 시각화
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X)

labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, edgecolors='k')
plt.title('GMM cluster result')
plt.show()

# 군집 포함 확률 시각화
probs = gmm.predict_proba(X)
plt.scatter(X[:, 0], X[:, 1], c=probs.max(axis=1), s=50, edgecolors='k')
plt.title('cluster rate')
plt.colorbar(label='rate')
plt.show()
```

## DBSCAN (Density-Based Spatial Clustering of Application with Noise)
- 밀도 기반(데이터포인트의 간격) 군집 알고리즘
- 핵심 파라미터 $\epsilon$(거리 임계값)과 minPts(최수 이웃 데이터 수)
    - **핵심 포인트 (Core Point)**
        - $\epsilon$거리 내에 최소 minPts이상의 이웃이 있는 데이터
    - **경계 포인트 (Border Point)**
        - 핵심 포인트 주변에 있지만, minPts에는 미치지 못하는 데이터
    - **노이즈 포인트(Noie Point)**
        - 어느 군집에도 속하지 않는 데이터
- 장점
    - 비구형 클러스터 탐지
    - 노이즈 데이터 처리
    - 비지도 학습 (클러스터 개수를 사전에 알 필요가 없음)
- 단점
    - 데이터 밀도가 자주 변하거나 아예 변하지 않으면 군집화 성능 저하
    - 특성 개수가 많으면 군집화 성능 저하 (고차원 데이터에서의 밀도 불균형)
    - 매개변수 민감성
![](https://d.pr/i/Re9qoB+)

![](https://d.pr/i/T3srVy+)

![](https://d.pr/i/tiIr6K+)

```python
# 반달형 데이터포인트 생성
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

dbscan = DBSCAN(eps=0.2, min_samples=6)    # eps(이웃 정의 거리 반지름, 0.5)
                                           # min_samples(minPts, 5)
dbscan.fit(X)    # 클러스터링 계산
# print(dbscan.labels_)

# 데이터포인트 산점도
plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels_)
plt.show()
```
- 가운데에 색이 다른 데이터포인트(Noise Point)가 있음
    - 이 점이 DBSCAN의 단점으로 군집에 이격돼있는 포인터는 노이즈 포인터로 처리한다.

#### iris_data DBSCAN적용
```python
# 데이터 로드
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# DBSCAN 적용
dbscan = DBSCAN(eps=0.6, min_samples=4)
dbscan.fit_predict(iris.data)    # dbscan은 transform, predict가 없음

iris_df['cluster'] = dbscan.labels_
iris_df.groupby('target')['cluster'].value_counts()

from sklearn.decomposition import PCA

# 시각화를 위한 PCA
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(iris.data)
iris_df['pca1'] = pca_transformed[:, 0]
iris_df['pca2'] = pca_transformed[:, 1]

# 시각화
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(iris_df['pca1'], iris_df['pca2'], c=iris_df['target'])    # iris 실제 target
ax[1].scatter(iris_df['pca1'], iris_df['pca2'], c=iris_df['cluster'])   # DBSCAN 결과
plt.show()
```