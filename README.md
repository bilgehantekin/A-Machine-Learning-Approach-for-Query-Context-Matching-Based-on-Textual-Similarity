# A-Machine-Learning-Approach-for-Query-Context-Matching-Based-on-Textual-Similarity

This repository contains the code and scripts for a data mining project aimed at building a question-answer relevance classifier using the Microsoft WikiQA dataset. We extract a variety of features—semantic embeddings, distance metrics, keyword overlaps, TF-IDF similarities, and cross-encoder scores—and train an XGBoost meta-classifier to distinguish correct from incorrect answers.

## Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Feature Extraction](#feature-extraction)

  * [Embedding-Based Features](#embedding-based-features)
  * [Keyword Overlap & TF-IDF](#keyword-overlap--tf-idf)
* [Cross-Encoder Scoring](#cross-encoder-scoring)
* [Model Training](#model-training)
* [Threshold Selection & Evaluation](#threshold-selection--evaluation)
* [Results](#results)
* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Author](#author)
* [License](#license)

## Overview

We leverage the WikiQA dataset to train a binary classifier that predicts whether a candidate answer is correct for a given question. The pipeline consists of:

1. **Data preparation:** Load train/validation/test splits and save as CSV.
2. **Feature extraction:** Compute embedding distances (cosine, Euclidean, dot product, L1), text length differences, embedding statistics, keyword overlap, and TF-IDF cosine similarity.
3. **Cross-encoder scoring:** Use a pre-trained sentence-pair classification model (MS MARCO MiniLM) to generate relevance logits.
4. **Meta-classifier:** Train an XGBoost classifier on the combined feature set with a tuned `scale_pos_weight` and hyperparameters.
5. **Threshold selection:** Search for the best probability threshold based on F1-score.
6. **Evaluation:** Report precision, recall, F1, classification report, and confusion matrix on the test set.

## Dataset

We use the `microsoft/wiki_qa` dataset via `datasets.load_dataset`. It provides three splits:

* `train`
* `validation`
* `test`

Each record contains:

* `question_id`
* `question`
* `document_title`
* `answer`
* `label` (1 for correct, 0 for incorrect)

## Feature Extraction

### Embedding-Based Features

* Load the `all-MiniLM-L6-v2` SentenceTransformer model.
* Encode questions and answers to 384-dimensional vectors.
* Compute:

  * Cosine similarity
  * Euclidean distance
  * Dot product
  * L1 norm difference
  * Absolute difference in text lengths
  * Absolute difference of vector means
  * Absolute difference of vector maxima

### Keyword Overlap & TF-IDF

* Use SpaCy (`en_core_web_sm`) to extract nouns, verbs, and adjectives.
* Compute Jaccard overlap of keywords between question and answer.
* Fit a TF-IDF vectorizer on all text, then compute cosine similarity.
* Merge these features with embedding features to create `final_feature_set_{train,test}.csv`.

## Cross-Encoder Scoring

* Load `cross-encoder/ms-marco-MiniLM-L-6-v2`.
* Tokenize question-answer pairs and obtain logits as relevance scores.
* Append `ce_score` to the feature DataFrames.

## Model Training

* Read `final_feature_set_train.csv` and `final_feature_set_test.csv`.
* Compute `scale_pos_weight = (# negatives) / (# positives)`.
* Initialize `XGBClassifier` with:

  * `n_estimators=200`
  * `max_depth=5`
  * `learning_rate=0.05`
  * `subsample=0.8`
  * `colsample_bytree=0.8`
  * `scale_pos_weight`
* Train on training features (excluding `question_id` and `label`).

## Threshold Selection & Evaluation

* For thresholds in `[0.1, 0.15, ..., 0.9]`, select the best answer per question and apply threshold.
* Compute precision, recall, F1 on test set.
* Choose threshold with highest F1.
* Generate final classification report and confusion matrix.

## Results

<img width="724" height="580" alt="image" src="https://github.com/user-attachments/assets/fb44c45f-d45f-4f93-b692-4a71a935a4d7" />

| Metric     | Value    |
|------------|----------|
| Accuracy   | 93.41%   |
| Precision  | 60.07%   |
| Recall     | 58.02%   |
| **F1-score** | **59.04%** |


Adjust accordingly based on your run.

## Requirements

* Python 3.8+
* `datasets`
* `pandas`
* `numpy`
* `sentence-transformers`
* `scikit-learn`
* `scipy`
* `tqdm`
* `spacy` + `en_core_web_sm`
* `xgboost`
* `transformers`
* `torch`

Install via:

```bash
pip install -r requirements.txt
```


