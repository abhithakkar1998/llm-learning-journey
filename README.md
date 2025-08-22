# LLM Learning Journey

This repository documents my hands-on journey into **Large Language Models (LLMs)**.  
The goal is to build practical understanding through small, day-wise experiments covering model inference, fine-tuning, RAG pipelines, and multi-agent frameworks.

---

## 📅 Progress

### **Day 1 – Getting Started with Hugging Face Pipelines**
- Set up a simple text generation pipeline using Hugging Face (`transformers`).
- Explored decoding strategies:
  - **Greedy search**
  - **Sampling with temperature**
  - **Nucleus sampling (top-p)**
- Compared outputs of the same query under different decoding parameters.

### **Day 2 – Tokenization, Embeddings & PCA Visualization**
- Learned how **tokenization** works in LLMs:
  - BERT (WordPiece) vs GPT-2 (Byte-Pair Encoding).
  - Key tokenizer outputs: `input_ids`, `attention_mask`, and `token_type_ids`.
- Extracted **embeddings** using:
  - `last_hidden_state` for token-level vectors.
  - `pooler_output` for sentence-level representation.
  - Explored **mean pooling** vs **CLS pooling** for different tasks.
- Visualized embeddings using **PCA (Principal Component Analysis)**:
  - Reduced 768-dimensional embeddings to 2D.
  - Observed how semantically related sentences cluster together.
- Key modules:  
  - `transformers`: `AutoTokenizer`, `AutoModel`, `pipeline()`  
  - `torch`: for inference & reproducibility (`torch.manual_seed`)  
  - `sklearn.decomposition.PCA`: dimensionality reduction

### **Day 3 – Semantic Similarity, Clustering & Search**
- Computed **cosine similarity** to measure semantic closeness between sentences.
- Performed **KMeans clustering** on sentence embeddings.
- Determined optimal clusters using:
  - **Elbow Method** (inertia drop-off)
  - **Silhouette Score** (cluster separation)
- Visualized clusters using **PCA and t-SNE**.
- Implemented a **mini semantic search demo**: retrieved the most similar sentence to a query.
- Learned when to use cosine similarity, clustering, and dimensionality reduction in real-world tasks.

---

## 🔧 Tech Stack
- Python
- Hugging Face Transformers
- PyTorch
- Jupyter/Colab
- scikit-learn (PCA, clustering, cosine similarity))
- Matplotlib / Seaborn (visualization)
---

## 🚀 Roadmap
- [x] **Day 1:** Introduction to Hugging Face pipelines & decoding strategies  
- [x] **Day 2:** Tokenization, embeddings extraction & visualization  
- [x] **Day 3:** Semantic similarity & sentence clustering using embeddings  
- [ ] **Day 4:** Building a simple semantic search system  
- [ ] **Day 5:** Introduction to fine-tuning (lightweight methods like LoRA/PEFT)  
- [ ] **Day 6:** Retrieval-Augmented Generation (RAG) basics  
- [ ] **Day 7:** Deploy a mini-project (Gradio web app or Flask API) using learned concepts

---

## 📖 References
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)  
- [Text Generation Strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)  
- [BERT Paper](https://arxiv.org/abs/1810.04805)
  
---

## 🤝 Contributing
This repo is mainly for personal learning, but suggestions and improvements are welcome.  
Feel free to open issues or PRs!
