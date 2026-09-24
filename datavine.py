# Import required libraries
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set visualization style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 100
np.random.seed(42)

# ==========================================
# Step 1: Data Preparation & Overview
# ==========================================

# 1. Load Datasets
wine_raw = load_wine(as_frame=True)
wine_df = wine_raw.frame
wine_df.rename(columns={'target': 'wine_class'}, inplace=True)

chickwts_df = pd.read_csv('data/chickwts.csv')
usarrests_df = pd.read_csv('data/USArrests.csv')
if 'State' in usarrests_df.columns:
    usarrests_df = usarrests_df.set_index('State')

# 2. Inspect Missing Values
print("Missing Values Check")
print(f"Wine missing values: {wine_df.isnull().sum().sum()}")
print(f"Chickwts missing values: {chickwts_df.isnull().sum().sum()}")
print(f"USArrests missing values: {usarrests_df.isnull().sum().sum()}")

# 3. Dataset Structural Summary
print("Dataset Summaries")
print(f"Wine Dataset: {wine_df.shape[0]} samples, {wine_df.shape[1] - 1} features, {wine_df['wine_class'].nunique()} classes")
print(f"Chickwts Dataset: {chickwts_df.shape[0]} samples, {chickwts_df['feed'].nunique()} distinct feed types")
print(f"USArrests Dataset: {usarrests_df.shape[0]} regions, {usarrests_df.shape[1]} features")

print("\nWine class distribution:")
print(wine_df['wine_class'].value_counts().sort_index())

print("\nChickwts feed distribution:")
print(chickwts_df['feed'].value_counts())

print("\nUSArrests descriptive stats:")
print(usarrests_df.describe().round(2))

# Light Exploratory Data Analysis
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

corr = wine_df.drop(columns=['wine_class']).corr()
sns.heatmap(
    corr,
    cmap='coolwarm',
    center=0,
    ax=ax1,
    annot=True,
    fmt='.2f',
    annot_kws={'size': 8},
    cbar_kws={'shrink': 0.8},
)
ax1.set_title('Wine Features Correlation Matrix', fontsize=12, pad=10)
ax1.tick_params(axis='x', labelsize=9, rotation=45)
ax1.tick_params(axis='y', labelsize=9)

sns.boxplot(
    data=chickwts_df,
    x='feed',
    y='weight',
    ax=ax2,
    palette='Set2',
    showfliers=False,
)
sns.stripplot(
    data=chickwts_df,
    x='feed',
    y='weight',
    ax=ax2,
    color='black',
    alpha=0.5,
    size=4,
)
ax2.set_title('Chick Weight by Feed Type', fontsize=12, pad=10)
ax2.tick_params(axis='x', rotation=45)

sns.scatterplot(
    data=usarrests_df,
    x='Murder',
    y='Assault',
    size='Rape',
    sizes=(40, 200),
    hue='UrbanPop',
    palette='viridis',
    ax=ax3,
)
ax3.set_title('USArrests: Murder vs Assault', fontsize=12, pad=10)
ax3.legend(
    bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.0, fontsize=8
)

sns.despine(ax=ax2)
sns.despine(ax=ax3)
plt.tight_layout()
plt.show()

print("EDA takeaways: Wine features show several strong correlations (good candidate for PCA).")
print("Chick weights differ markedly by feed — sunflower and casein produce higher gains.")
print("USArrests violence metrics are positively associated; states vary widely in profile.")

# ==========================================
# Step 2: Wine Classification System
# ==========================================

X_wine = wine_df.drop(columns=['wine_class'])
y_wine = wine_df['wine_class']

X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X_wine, y_wine, test_size=0.3, random_state=42, stratify=y_wine
)

scaler_wine = StandardScaler()
X_train_w_scaled = scaler_wine.fit_transform(X_train_w)
X_test_w_scaled = scaler_wine.transform(X_test_w)

pca_wine = PCA(n_components=0.95, random_state=42)
X_train_w_pca = pca_wine.fit_transform(X_train_w_scaled)
X_test_w_pca = pca_wine.transform(X_test_w_scaled)

print(f"Original feature count: {X_train_w_scaled.shape[1]}")
print(f"PCA components retaining 95% variance: {pca_wine.n_components_}")
print(f"Explained variance ratio (first 5): {pca_wine.explained_variance_ratio_[:5].round(3)}")
print(f"Cumulative variance explained: {pca_wine.explained_variance_ratio_.sum():.3f}")

param_grid = {
    'n_neighbors': list(range(1, 15)),
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'weights': ['uniform', 'distance']
}
grid_search = GridSearchCV(
    KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train_w_pca, y_train_w)
best_knn = grid_search.best_estimator_

print(f"\nOptimal Hyperparameters: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

y_pred_knn = best_knn.predict(X_test_w_pca)
knn_acc = accuracy_score(y_test_w, y_pred_knn)
print(f"\nk-NN Test Accuracy: {knn_acc:.4f}")
print("\nClassification Report (k-NN):")
print(classification_report(y_test_w, y_pred_knn, target_names=wine_raw.target_names))

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_w_pca, y_train_w)
y_pred_lr = lr.predict(X_test_w_pca)
lr_acc = accuracy_score(y_test_w, y_pred_lr)
print(f"Logistic Regression (baseline) Test Accuracy: {lr_acc:.4f}")

cm_knn = confusion_matrix(y_test_w, y_pred_knn)
cm_lr = confusion_matrix(y_test_w, y_pred_lr)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
disp_knn = ConfusionMatrixDisplay(
    confusion_matrix=cm_knn, display_labels=wine_raw.target_names
)
disp_knn.plot(cmap='Blues', ax=axes[0])
axes[0].set_title(f'k-NN Confusion Matrix\n(Accuracy = {knn_acc:.3f})')

disp_lr = ConfusionMatrixDisplay(
    confusion_matrix=cm_lr, display_labels=wine_raw.target_names
)
disp_lr.plot(cmap='Greens', ax=axes[1])
axes[1].set_title(f'Logistic Regression Baseline\n(Accuracy = {lr_acc:.3f})')

plt.tight_layout()
plt.show()

print(f"\nSummary: Tuned k-NN achieves {knn_acc:.1%} test accuracy vs {lr_acc:.1%} for the logistic baseline.")
print("Both models perform strongly; PCA + k-NN remains competitive while being non-parametric.")

# ==========================================
# Step 3: Agricultural Feed Recommendation Engine
# ==========================================

scaler_chick = StandardScaler()
chickwts_df['weight_scaled'] = scaler_chick.fit_transform(chickwts_df[['weight']])

feed_profiles = chickwts_df.groupby('feed')['weight_scaled'].agg(
    ['mean', 'std', 'median', 'min', 'max', 'count']
).fillna(0)

print("Feed Profiles (standardized weight statistics):")
print(feed_profiles.round(3))

n_comp = min(3, feed_profiles.shape[1] - 1)
pca_feed = PCA(n_components=n_comp, random_state=42)
feed_pca = pca_feed.fit_transform(feed_profiles)
feed_representation = pd.DataFrame(
    feed_pca,
    index=feed_profiles.index,
    columns=[f'PC{i+1}' for i in range(n_comp)]
)

print(f"\nPCA components used: {n_comp}")
print(f"Explained variance ratio: {pca_feed.explained_variance_ratio_.round(3)}")
print(f"Cumulative variance: {pca_feed.explained_variance_ratio_.sum():.3f}")

print("\nFeed representation in PCA space:")
print(feed_representation.round(3))

cos_sim_matrix = cosine_similarity(feed_representation)
cos_sim_df = pd.DataFrame(
    cos_sim_matrix, index=feed_profiles.index, columns=feed_profiles.index
)

print("\nCosine Similarity Matrix:")
print(cos_sim_df.round(3))

def recommend_feeds(feed_name, top_n=2):
    sims = cos_sim_df[feed_name].drop(feed_name).sort_values(ascending=False)
    return sims.head(top_n)

print("\nRecommended Alternative Feeds")
print("-" * 50)
for feed in cos_sim_df.index:
    recs = recommend_feeds(feed)
    formatted_recs = ", ".join([f"{idx} (Sim: {val:.3f})" for idx, val in recs.items()])
    print(f"{feed.capitalize():12} -> {formatted_recs}")

plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    cos_sim_df,
    annot=True,
    fmt='.2f',
    cmap='YlOrRd',
    square=True,
    annot_kws={'size': 10},
    cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.8}
)
plt.title('Feed Similarity Heatmap (PCA-based)', fontsize=13, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()

# ==========================================
# Step 4: Regional Crime Pattern Analysis
# ==========================================

selected_features = ['Murder', 'Assault', 'Rape']
X_crime = usarrests_df[selected_features]
scaler_crime = StandardScaler()
X_crime_scaled = scaler_crime.fit_transform(X_crime)

pca_crime = PCA(n_components=2, random_state=42)
X_crime_pca = pca_crime.fit_transform(X_crime_scaled)
pca_df = pd.DataFrame(
    X_crime_pca, columns=['PC1', 'PC2'], index=usarrests_df.index
)

print(f"PCA explained variance ratio: {pca_crime.explained_variance_ratio_.round(3)}")
print(f"Cumulative variance explained: {pca_crime.explained_variance_ratio_.sum():.3f}")

k_range = range(2, 9)
inertia_scores, bic_scores, sil_scores = [], [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_crime_pca)
    inertia_scores.append(km.inertia_)
    sil_scores.append(silhouette_score(X_crime_pca, km.labels_))
    
    gmm = GaussianMixture(n_components=k, random_state=42).fit(X_crime_pca)
    bic_scores.append(gmm.bic(X_crime_pca))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].plot(list(k_range), inertia_scores, 'o-', color='steelblue', linewidth=2)
axes[0].set_xlabel('Number of clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method (K-Means)')
axes[0].axvline(3, color='crimson', linestyle='--', alpha=0.7, label='Selected k=3')
axes[0].legend()

axes[1].plot(list(k_range), bic_scores, 'o-', color='darkorange', linewidth=2)
axes[1].set_xlabel('Number of components')
axes[1].set_ylabel('BIC')
axes[1].set_title('BIC (GMM) — lower is better')
axes[1].axvline(3, color='crimson', linestyle='--', alpha=0.7, label='Selected k=3')
axes[1].legend()

axes[2].plot(list(k_range), sil_scores, 'o-', color='forestgreen', linewidth=2)
axes[2].set_xlabel('Number of clusters (k)')
axes[2].set_ylabel('Silhouette Score')
axes[2].set_title('Silhouette Analysis')
axes[2].axvline(3, color='crimson', linestyle='--', alpha=0.7, label='Selected k=3')
axes[2].legend()

for ax in axes:
    sns.despine(ax=ax)

plt.tight_layout()
plt.show()

print("Model selection notes:")
print(f"  Silhouette scores: {dict(zip(k_range, [round(s, 3) for s in sil_scores]))}")
print("  Elbow and BIC both support a small number of clusters; silhouette peaks around k=2–3.")
print("  We select k=3 as a balanced, interpretable solution for policy segmentation.")

optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
pca_df['KMeans_Cluster'] = kmeans_final.fit_predict(X_crime_pca).astype(str)

gmm_final = GaussianMixture(n_components=optimal_k, random_state=42)
pca_df['GMM_Cluster'] = gmm_final.fit_predict(X_crime_pca).astype(str)

sil_km = silhouette_score(X_crime_pca, kmeans_final.labels_)
sil_gmm = silhouette_score(X_crime_pca, gmm_final.predict(X_crime_pca))
print(f"\nFinal Silhouette — K-Means: {sil_km:.3f} | GMM: {sil_gmm:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

sns.scatterplot(
    data=pca_df,
    x='PC1',
    y='PC2',
    hue='KMeans_Cluster',
    palette='Set1',
    s=100,
    ax=axes[0],
    edgecolor='k',
    linewidth=0.5
)
km_centers = kmeans_final.cluster_centers_
axes[0].scatter(
    km_centers[:, 0], km_centers[:, 1], c='black', marker='X', s=250, label='Centroids', zorder=10
)
axes[0].set_title(
    f'K-Means Hard Clustering (k={optimal_k})\nSilhouette = {sil_km:.3f}', fontsize=12, pad=10
)
axes[0].legend(title='Cluster', loc='upper right')

sns.scatterplot(
    data=pca_df,
    x='PC1',
    y='PC2',
    hue='GMM_Cluster',
    palette='Set2',
    s=100,
    ax=axes[1],
    edgecolor='k',
    linewidth=0.5
)
gmm_means = gmm_final.means_
axes[1].scatter(
    gmm_means[:, 0], gmm_means[:, 1], c='black', marker='X', s=250, label='Cluster Means', zorder=10
)
axes[1].set_title(
    f'GMM Probabilistic Clustering (k={optimal_k})\nSilhouette = {sil_gmm:.3f}', fontsize=12, pad=10
)
axes[1].legend(title='Cluster', loc='upper right')

sns.despine(ax=axes[0])
sns.despine(ax=axes[1])
plt.tight_layout()
plt.show()

profile_df = usarrests_df[selected_features].copy()
profile_df['Cluster'] = pca_df['KMeans_Cluster']
cluster_summary = profile_df.groupby('Cluster')[selected_features].agg(['mean', 'std', 'count'])

print("K-Means Cluster Profiles (original scale):")
print("-" * 55)
print(cluster_summary.round(2))

print("\nInterpretation guide:")
print("  Higher Murder/Assault/Rape means -> higher-violence cluster.")
print("  Policy teams can target interventions by cluster membership.")