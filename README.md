## About the Project – **FitGenie: AI-Powered Clothing Recommendation**

### 🌟 Inspiration
Online shopping offers endless choices but creates **decision fatigue**.  
Shoppers often struggle to find clothes that *fit* their body and *match* their style, leading to abandoned carts and high return rates.  
**FitGenie** was born to act as a personal AI stylist — combining semantic understanding of product text (BERT-based embeddings) with a deep collaborative ranking model to recommend items that both *look like* what a user wants and *behave like* what similar users buy.

---

### 🛠️ How We Built It
We implemented a **hybrid recommendation pipeline** consisting of:

1. **Textual Semantic Encoder (Sentence-BERT)**  
   - Each product is represented by a rich text string (name, category, color, price range, style keywords).  
   - Encoded with the Sentence-BERT model `all-MiniLM-L6-v2` to produce dense vector embeddings for semantic similarity.

2. **Neural Collaborative Ranking Model (PyTorch)**  
   - A custom `DeepRecommendationModel` with embedding layers for user, item, category, color, and price range.  
   - Captures collaborative signals and higher-order user–item interactions beyond pure text similarity.

3. **Hybrid Scoring & Ranking**  
   - Computes semantic similarities (cosine) between user-liked item embeddings and all products.  
   - Boosts positives and penalizes negatives, then re-ranks using the deep model for the final recommendation list.

4. **Deployment**  
   - A lightweight **Flask** API serves real-time recommendations through endpoints such as `/initialize`, `/recommend`, `/similar-items`, and `/feedback`.  
   - Item embeddings are cached for fast lookup.

---

### 🔢 Key Formulas (LaTeX)
Cosine similarity for semantic matching:
\[
\text{Similarity}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}
\]

Hybrid semantic score:
\[
\text{score}_i = 1.5\sum_{j \in L} \cos(e_j, e_i) - 1.0\sum_{k \in D} \cos(e_k, e_i)
\]
where \(L\) = liked item indices, \(D\) = disliked item indices, and \(e\) are Sentence-BERT embeddings.

Neural collaborative prediction:
\[
\hat{y} = \mathrm{MLP}\big([u_{emb}; v_{emb}; c_{emb}; col_{emb}; p_{emb}; n_{\text{colors}}; \text{price}]\big)
\]

---

### 💡 What We Learned
- How **Sentence-BERT** enables semantic understanding of product text for cold-start recommendations.  
- Combining **content-based** and **collaborative** signals produces more accurate and diverse recommendations.  
- Importance of **data cleaning and feature engineering** for extracting style keywords, price buckets, and color attributes.  
- Building and deploying a **real-time API** that balances accuracy with low latency.

---

### ⚡ Challenges
- **Cold Start** – Generating meaningful suggestions for brand-new users or products.  
- **Scalability** – Ensuring low-latency inference when dealing with thousands of product embeddings.  
- **Model Alignment** – Harmonizing semantic and collaborative scores so recommendations feel consistent.  
- **Data Quality** – Handling inconsistent product descriptions across different data sources.

---

### 🚀 Future Improvements
- Integrate **FAISS** or **Annoy** for approximate nearest-neighbor search to speed up similarity queries.  
- Fine-tune the Sentence-BERT model on fashion-specific text for improved embeddings.  
- Incorporate **user feedback loops** to continuously update the collaborative model in production.  
- Develop a mobile-friendly UI to showcase FitGenie as a plug-and-play solution for e-commerce platforms.

---

**FitGenie** blends the precision of deep learning with the creativity of fashion, making personalized shopping faster, smarter, and more enjoyable.
