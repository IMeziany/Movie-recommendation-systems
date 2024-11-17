# Movie Recommender Systems using Regularized Movie and User Effect Model

This project implements a series of **movie recommendation models** using collaborative filtering techniques, culminating in a **regularized movie and user effect model**. The goal is to analyze the **MovieLens 100K dataset**, explore the underlying data patterns, and evaluate different approaches for recommending movies based on user preferences.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
4. [Recommendation Models](#recommendation-models)
5. [Results](#results)
6. [Getting Started](#getting-started)


---

## Introduction

Recommender systems aim to predict a user's preference for an item, such as a movie, song, or product, based on their historical interactions. They are widely used across industries to improve user satisfaction by suggesting personalized content.

This project focuses on **collaborative filtering**, which builds recommendations based on user-item interactions without requiring explicit content information. Specifically, it evaluates the following models:
1. **Simple Overall Average Model**
2. **Movie Effect Model**
3. **Movie and User Effect Model**
4. **Regularized Movie and User Effect Model**

These models are implemented and benchmarked on the **MovieLens 100K** dataset.

---

## Dataset Description

The project uses the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/), which contains:
- **100,000 ratings** (1–5 scale) by 943 users on 1,682 movies.
- Each user has rated at least 20 movies.
- Metadata about movies (e.g., genres, titles).
- Basic demographic information for users (e.g., age, gender, occupation).

---

## Exploratory Data Analysis

### Goals of EDA
1. **Understand Rating Patterns**:
   - Distribution of ratings across users and movies.
   - Popularity of movies based on the number of ratings.
2. **Analyze Genres**:
   - Count movies by genre and analyze average ratings for each genre.
3. **Explore User Demographics**:
   - Distribution of users by age, gender, and occupation.
4. **Temporal Patterns**:
   - Ratings over time to identify trends in user engagement.

### Key Insights
- **Ratings Distribution**: Most ratings fall between 3 and 4, with very few extreme ratings.
- **Movie Popularity**: A small number of movies receive a majority of the ratings.
- **Genre Trends**: Drama and Comedy are the most represented genres, while Fantasy and Film-Noir are underrepresented.
- **User Demographics**: Users aged 18–34 dominate the dataset.

---

## Recommendation Models

### 1. Simple Overall Average Model
- **Description**: Predicts the same rating for all user-item pairs based on the global average rating in the training data.

### 2. Movie Effect Model
- **Description**: Adjusts predictions by considering the average rating for each movie (movie bias).


### 3. Movie and User Effect Model
- **Description**: Combines movie bias and user bias to predict ratings.
  - **Movie Bias**: The average deviation of each movie's ratings from the global average.
  - **User Bias**: The average deviation of a user's ratings from the global average and the movie bias.


### 4. Regularized Movie and User Effect Model
- **Description**: Adds regularization to the Movie and User Effect Model to penalize large effects for movies and users with limited data.

- **Tunable Hyperparameter**:
  - Regularization parameter (\(\lambda\)): Controls the penalty for large deviations.

### 5. Collaborative Filtering Using SVD
- **Description**: Implements matrix factorization to discover latent factors underlying user-item interactions.
- **Key Outputs**:
  - **Cumulative Explained Variance**: Determines the number of latent factors required to capture a significant portion of the variance.
  - **RMSE**: Evaluates the prediction accuracy across train-test splits.

---

## Results

### Model Performance (Average RMSE)
| Model                              | RMSE         |
|------------------------------------|--------------|
| Simple Overall Average Model       | ~1.03        |
| Movie Effect Model                 | ~0.98        |
| Movie and User Effect Model        | ~0.95        |
| Regularized Movie and User Effect Model | **~0.94**  |

### Best Regularization Parameter (\(\lambda\)): **3.5**
- RMSE: **0.9436**

### Cumulative Explained Variance (SVD):
- **90% variance** explained with ~200 latent factors.

---

## Getting Started

### Prerequisites
- Python 3.x
- Libraries: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/IMeziany/movie-recommender-system.git
   cd movie-recommender-system

2. Install dependencies:
 
    ```bash
    pip install -r requirements.txt



## Future Work
- **Content-Based Filtering**:
  - Leverage movie metadata (e.g., genres, release year) to recommend similar items.
- **Hybrid Methods**:
  - Combine collaborative and content-based approaches for improved performance.
- **Advanced Matrix Factorization**:
  - Implement models like **Non-Negative Matrix Factorization (NMF)** or **Alternating Least Squares (ALS)**.
- **Deep Learning**:
  - Use neural networks to capture complex user-item interactions.
- **Cold Start Solutions**:
  - Incorporate demographic or contextual information to address cold start scenarios.


