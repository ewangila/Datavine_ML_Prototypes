# DataVine ML Prototypes

**Applied machine learning prototypes** demonstrating supervised classification (k-NN + PCA) and unsupervised clustering (K-Means, GMM) on real-world datasets.

**Author:** Eugin Wangila  
**License:** MIT

---

## Overview

This repository contains three end-to-end machine learning prototypes developed for boutique consulting use cases. Each solution follows a complete data science workflow: data loading and inspection, exploratory analysis, preprocessing, modeling, evaluation, and interpretation.

| Prototype | Technique | Goal |
|-----------|-----------|------|
| **Wine Classification System** | PCA + k-NN (tuned) vs Logistic Regression | Automated multi-class wine classification |
| **Agricultural Feed Recommendation Engine** | PCA + Cosine Similarity | Ranked alternative feed suggestions based on weight-gain profiles |
| **Regional Crime Pattern Analysis** | K-Means & GMM + model selection (Elbow, BIC, Silhouette) | Discover natural groupings in public-safety data for policy segmentation |

---

## Repository Structure
```
Datavine_ML_Prototypes/
├── data/                          # Input datasets
│   ├── chickwts.csv
│   └── USArrests.csv
├── Datavine_Analytics 2.ipynb     # Full interactive notebook (recommended)
├── datavine.py                    # Script version of the same pipeline
├── requirements.txt               # Python dependencies
├── LICENSE
└── README.md
```
**Note:** The Wine dataset is loaded directly from scikit-learn (`load_wine`).
---

## Installation & Setup

```bash
# Clone the repository
git clone [https://github.com/ewangila/Datavine_ML_Prototypes.git](https://github.com/ewangila/Datavine_ML_Prototypes.git)
cd Datavine_ML_Prototypes

# (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
## How to Run

### Option A — Jupyter Notebook (recommended for exploration)
```bash
jupyter notebook "Datavine_Analytics 2.ipynb"
```
### Option B — Python Script
```Bash
python datavine.py
```

## Dependencies

```text
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

## Methodology Highlights

- **Reproducibility**: Fixed random seed (`np.random.seed(42)`) throughout
- **Preprocessing**: `StandardScaler` applied before PCA / distance-based methods
- **Model Selection**:
  - **Classification**: `GridSearchCV` + stratified train/test split
  - **Clustering**: Elbow + BIC + Silhouette consensus
- **Evaluation**: Accuracy, classification report, confusion matrices, silhouette scores

---

## License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

## Contact

**Eugin Wangila**  
Data Scientist · Nairobi  
[GitHub](https://github.com/ewangila)
