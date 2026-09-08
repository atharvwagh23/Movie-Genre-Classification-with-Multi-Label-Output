# Movie Genre Classification with Multi-label Output

Predicting **all applicable genres** for a movie using only its plot summary text — a multi-label text classification project, unlike traditional single-genre classifiers.

## 📌 Overview

Real movies usually belong to **multiple genres at once** (e.g. a movie can be Action, Sci-Fi, and Thriller together). This project builds a **multi-label classification model** that predicts a *set* of genres given only a plot summary, using TF-IDF features and a One-vs-Rest Logistic Regression classifier.

## Movie Genre Classification with Multi-Label Output/
│
├── Data/
│   ├── movie_metadata.tsv     # Movie details incl. genre tags (id, title, release date, revenue, runtime, languages, countries, genre dict)
│   └── plot_summaries.tsv     # Movie id + plot summary text
│
├── Movie_Genre_Classifier_with_Multi_label_Output.ipynb   # Main notebook
└── README.md
*(Note: exact source/provenance of this dataset is unconfirmed — update this section if you know where it originally came from.)*

## 🔍 Approach

1. **Load & Merge** — loaded `movie_metadata.tsv` and `plot_summaries.tsv` separately, then merged them on `movie_id` since genre info and plot text live in different files.
2. **Clean Genre Column** — the raw genre field is a JSON-style dictionary string (e.g. `{"/m/01jfsb": "Thriller", "/m/06n90": "Science Fiction"}`); parsed it into a clean Python list of genre names per movie.
3. **Remove Empty Rows** — dropped movies left with zero genre tags after cleaning.
4. **Exploratory Data Analysis** — visualized frequency of all 363 unique genre tags found in the dataset; found the distribution is highly imbalanced (`Drama` and `Comedy` dominate, most other genres are long-tail/rare).
5. **Text Preprocessing** — cleaned plot text by:
   - Removing punctuation/numbers (keeping only letters)
   - Lowercasing
   - Removing English stopwords
   - Stemming words to their root form (Porter Stemmer)
6. **Multi-label Target Encoding** — converted each movie's genre list into a multi-hot binary vector using `MultiLabelBinarizer`.
7. **Feature Extraction** — converted cleaned plot text into numeric features using **TF-IDF** (top 10,000 features).
8. **Model Training** — trained a **One-vs-Rest Logistic Regression** classifier (fits one binary classifier per genre — 363 in total).
9. **Evaluation** — measured performance using **F1-score (micro-average)**, since standard accuracy doesn't work well for multi-label problems. Compared results at two different prediction thresholds.
10. **Prediction** — built a reusable function to predict genre tags for any new plot summary, tested on 5 real movies (Avatar, Titanic, The Conjuring, The Hangover, La La Land).

## 📊 Results

| Threshold | F1 Score (micro) |
|---|---|
| 0.5 (default) | 32.5% |
| 0.2 (tuned) | **47.4%** |

**Key insight:** Lowering the classification threshold significantly improved F1 score — multi-label classifiers tend to be overly conservative by default, so allowing lower-confidence genre predictions captures more of a movie's true genre set. Performance is naturally lower than a single-label classifier, since the model has to correctly predict *multiple* genres per movie out of 363 possible options, many of which are rare/underrepresented in training data.

## 🛠️ Tech Stack

- **Python 3**
- **Pandas / NumPy** — data handling
- **Matplotlib / Seaborn** — visualization
- **NLTK** — stopwords removal, stemming, frequency distribution
- **Scikit-learn** — TfidfVectorizer, MultiLabelBinarizer, train-test split, OneVsRestClassifier, LogisticRegression, F1-score

## 🚀 How to Run

1. Clone this repository:
```bash
   git clone <your-repo-url>
   cd "Movie Genre Classification with Multi-Label Output"
```

2. Install dependencies:
```bash
   pip install numpy pandas matplotlib seaborn nltk scikit-learn
```

3. Open the notebook and run all cells:
```bash
   jupyter notebook Movie_Genre_Classifier_with_Multi_label_Output.ipynb
```
   Or upload it to [Google Colab](https://colab.research.google.com) and upload `movie_metadata.tsv` / `plot_summaries.tsv` from the `Data/` folder when prompted.

4. The notebook will output:
   - Genre frequency visualization (top 50 genres)
   - Trained multi-label model with F1-score evaluation at two thresholds
   - Live genre predictions for 5 sample movie plots

## 📁 Dataset

- `movie_metadata.tsv` — 81,741 rows, tab-separated, no header row. Columns: `movie_id`, freebase_id, `title`, release_date, revenue, runtime, languages (dict), countries (dict), `genre` (dict)
- `plot_summaries.tsv` — 42,306 rows, tab-separated, no header row. Columns: `movie_id`, `plot summary text`
- After merging + cleaning: **41,793** usable movie records, spanning **363 unique genre tags**

## 📈 Future Improvements

- Filter to the top ~20-30 most frequent genres instead of all 363, to reduce label sparsity and likely improve F1 score
- Try other multi-label-friendly models (Random Forest, XGBoost, or neural approaches)
- Use word embeddings (Word2Vec, GloVe) or transformer-based models (BERT) for richer text representation
- Address genre imbalance with per-label class weighting

## 📄 License

This project is for educational purposes.
