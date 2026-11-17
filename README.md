# 🌍 Countries Clustering

An unsupervised learning project that groups **167 countries** by socio-economic and health indicators to estimate their overall level of development. The goal is to help an NGO such as HELP International decide where to focus its funding. The data is standardized and reduced with **PCA**, then clustered with five algorithms that are compared using Silhouette and Calinski-Harabasz scores.

![Python](https://img.shields.io/badge/Language-Python%203.10-3776AB?style=flat-square)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?style=flat-square)
![scikit-learn](https://img.shields.io/badge/Library-scikit--learn-F7931E?style=flat-square)
![Plotly](https://img.shields.io/badge/Maps-Plotly-3F4F75?style=flat-square)
![Task](https://img.shields.io/badge/Task-Clustering%20%2B%20PCA-6A5ACD?style=flat-square)

## 📦 Dataset

`Country-data.csv`: **167 countries × 10 columns**, with no missing values.

| Feature | Description |
|---------|-------------|
| `country` | Name of the country |
| `child_mort` | Deaths of children under 5 per 1,000 live births |
| `exports` | Exports of goods and services, as % of GDP per capita |
| `imports` | Imports of goods and services, as % of GDP per capita |
| `health` | Total health spending, as % of GDP per capita |
| `income` | Net income per person |
| `inflation` | Annual growth rate of total GDP |
| `life_expec` | Average life expectancy of a newborn (years) |
| `total_fer` | Number of children born per woman |
| `gdpp` | GDP per capita |

## 🔄 Workflow

1. **EDA:** pair plot, box plots, distributions, correlation heatmap, top 5 / bottom 5 countries per indicator, and world choropleth maps.
2. **Scaling:** `StandardScaler` on the 9 numeric features.
3. **Dimensionality reduction:** PCA down to **5 components** (94.5% of the variance).
4. **Choosing *k*:** elbow plot, Silhouette and Calinski-Harabasz scores for k = 2 to 10, and a hybrid score `√(Silhouette × Calinski-Harabasz)`.
5. **Clustering** on the PCA components with K-Means, Hierarchical (Ward), Gaussian Mixture, Mini-Batch K-Means, and DBSCAN (a grid search over `eps` and `min_samples`).
6. **Mapping:** each result is shown on a world map, colored by help priority.
7. **Validation:** classifiers are trained to predict the K-Means cluster from the raw indicators.

## 🧮 PCA

| Components | Cumulative variance explained |
|:---:|:---:|
| 2 | 63.1% |
| 3 | 76.1% |
| 4 | 87.2% |
| **5 (used)** | **94.5%** |
| 9 (all) | 100% |

- **PC1 (46% of variance)** is an overall **development axis**. `life_expec`, `income`, and `gdpp` load positively, and `child_mort` and `total_fer` load negatively, so a higher PC1 means a more developed country.
- **PC2 (17%)** mainly captures **trade openness**, driven by `imports` and `exports`.

## 📊 Results

| Model | Clusters | Silhouette | Calinski-Harabasz | Cluster sizes |
|-------|:--------:|:----------:|:-----------------:|---------------|
| K-Means | 2 | 0.30 | **77.07** | 68 / 99 |
| Hierarchical (Ward) | 2 | **0.33** | 73.09 | 117 / 50 |
| Gaussian Mixture | 2 | 0.28 | 71.36 | 94 / 73 |
| Mini-Batch K-Means | 3 | 0.32 | 73.09 | 48 / 87 / 32 |
| DBSCAN (`eps=1.0`, `min_samples=6`) | 3 + noise | 0.15 | n/a | 72 / 17 / 22 / 56 noise |

How *k* was chosen:

- **K-Means:** the hybrid score picked k = 2 (Silhouette alone favored 4).
- **Hierarchical:** k = 2 (Silhouette alone favored 6).
- **Gaussian Mixture:** both metrics agreed on k = 2.
- **Mini-Batch K-Means:** the hybrid score picked k = 3.

Notes on the results:

- **Two broad groups:** most algorithms split the world into lower- and higher-development countries.
- **Mini-Batch K-Means** gives a three-tier view with a Silhouette (0.32) close to the best (0.33). Its tiers line up with the development axis (PC1 centers of −2.41, +0.36, and +2.83): **Help Needed** (48 countries), **Might Need Help** (87), and **No Help Needed** (32).
- **DBSCAN** finds three dense groups but labels 56 countries (about a third) as noise or outliers, and it has the lowest Silhouette.

### 🌲 Cluster validation

Five classifiers were trained to predict the K-Means cluster from the original indicators (80/20 split, 34 test countries):

| Random Forest | Decision Tree | AdaBoost | Gradient Boosting | XGBoost |
|:---:|:---:|:---:|:---:|:---:|
| 0.971 | 0.941 | 0.971 | 0.941 | 0.971 |

High accuracy shows the clusters are clearly defined by the raw indicators.

## 📁 Repository Structure

```
CountriesClustering/
├── countries-clustering.ipynb   # EDA, PCA, clustering, evaluation, maps
└── README.md
```

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone <this-repository-url>
   cd <repository-folder>
   ```
2. Install the dependencies (the notebook was run with Python 3.10):
   ```bash
   pip install pandas numpy scipy matplotlib seaborn plotly scikit-learn \
               yellowbrick xgboost tqdm jupyter
   ```
   `scikit-learn` 1.2 or newer is needed for the `metric=` argument of `AgglomerativeClustering`.
3. Download `Country-data.csv` and update the path in the data-loading cell. The notebook was written on Kaggle, so the default path is a Kaggle one:
   ```python
   df = pd.read_csv("path/to/Country-data.csv")
   ```
4. Launch Jupyter and run the cells in order:
   ```bash
   jupyter notebook countries-clustering.ipynb
   ```

> Newer seaborn versions deprecate `sns.distplot` and the `size` argument of `sns.pairplot`. If you hit warnings or errors, replace them with `histplot`/`displot` and `height`.

## 🎯 Learning Objectives

- Apply PCA before clustering and interpret the components
- Compare centroid-based, hierarchical, model-based, and density-based clustering
- Choose the number of clusters with elbow, Silhouette, and Calinski-Harabasz methods
- Tune DBSCAN with a k-distance plot and a parameter grid search
- Communicate clustering results with choropleth maps
