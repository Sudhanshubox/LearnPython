# m45 · Project: a recommender system with matrix factorization

**Phase 4 finale.** You'll build the core of a movie recommender, the kind that won the Netflix Prize, entirely in NumPy. It brings together the whole phase: vectorization (m37), the SVD and low-rank structure (m39), gradients (m40), optimization with Adam and regularization (m41), and honest train/test evaluation (m43).

**Why it matters for AI:** matrix factorization learns an **embedding** for every user and every item, so that a user's predicted rating of an item is the dot product of their embeddings. It's the ancestor of the embedding-based models behind modern recommendations, search and RAG, and a clean, small example of "learn representations by minimizing a loss with gradient descent", the core idea of deep learning.

---

## The problem

You have a ratings matrix R: rows are users, columns are items (movies), and most entries are **missing**, since nobody rates everything. The goal is to predict the missing ratings, then recommend each user the items with the highest predictions.

The key assumption: tastes are **low-rank**. A few hidden factors ("how much action", "how serious", "how old") explain most ratings. So we look for user vectors uᵢ and item vectors vⱼ in ℝᵏ with

rating(i, j) ≈ baseline(i, j) + uᵢ · vⱼ

## The plan

1. **Data.** `make_ratings()` (already written) generates realistic synthetic ratings from hidden low-rank factors plus user and item biases and noise, with about 30% of entries observed. Split the observed entries into **train** and **test** masks.
2. **Metric.** Root mean squared error (RMSE) on the *test* entries only.
3. **Baseline.** Predict global mean + user bias + item bias. Any fancier model must beat this.
4. **SVD approach.** Fill missing entries with the baseline, then take the best rank-k approximation (m39). Simple, but the filled-in values bias the result.
5. **Matrix factorization.** Learn U (users × k) and V (items × k) by minimizing, over the **observed training entries only**:

   L = (1/n) Σ₍ᵢ,ⱼ₎ observed (uᵢ · vⱼ − tᵢⱼ)² + λ (‖U‖² + ‖V‖²)

   where t = R − baseline (the residuals the biases don't explain), and λ is **L2 regularization** to prevent overfitting. With E = mask ⊙ (UVᵀ − T):

   ∂L/∂U = (2/n) E V + 2λU    ∂L/∂V = (2/n) Eᵀ U + 2λV

   Optimize with Adam (m41).
6. **Use it.** Recommend the top unrated items per user, and find "similar items" by cosine similarity of their learned embeddings (m38).

`exercises.py` walks through these steps in order. The tests check each piece, including a gradient check of your loss, and finally require your model to beat both baselines on the test set by a wide margin.

## Results to aim for

On the default data, the reference implementation gets a test RMSE of roughly:

| Method | Test RMSE |
|---|---|
| global mean | 1.79 |
| baseline (biases) | 1.67 |
| SVD of the filled-in matrix, k = 5 | 1.39 |
| matrix factorization, k = 3 | ~0.4 |

Matrix factorization wins because it only fits the ratings that actually exist, instead of treating made-up filled-in values as data.

## Stretch goals (optional)

- **Overfitting study:** plot train and test RMSE against k (1–50) for λ = 0 and λ = 1e-3. Where does test error start rising? Explain using m43's ideas.
- **Hyperparameter search:** use a *validation* split (carved out of train) to choose k and λ, then report the test RMSE once. Why is choosing hyperparameters on the test set cheating?
- **Cold start:** how would you recommend items to a brand-new user with no ratings?
- **Real data:** download MovieLens 100K (grouplens.org), load it with your m09 skills, and run the same pipeline. What RMSE do you get?

## Go deeper (optional, research-level)

1. Read *Matrix Factorization Techniques for Recommender Systems* (Koren, Bell & Volinsky, 2009), the Netflix Prize winners' overview. Which of their extensions (biases, implicit feedback, temporal dynamics) did you implement?
2. Implicit feedback (clicks, watches) has no negative ratings. Read about weighted matrix factorization / ALS for implicit data (Hu, Koren & Volinsky, 2008).
3. Modern two-tower retrieval models replace U and V with neural networks that compute embeddings from features. How does that solve the cold-start problem?
