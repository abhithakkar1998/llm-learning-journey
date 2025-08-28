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

## Day 4: Building a Simple Semantic Search System
- Built a **semantic search engine** using Wikipedia topics on AI/ML/Data Science.
- Converted each topic's summary into **sentence embeddings** using a Transformer model.
- Implemented a **search function** to retrieve most relevant topics based on cosine similarity.
- Added **visualization** (bar chart) to interpret results.
- Created a **minimal Gradio UI** for interactive semantic search.

## Day 5: Introduction to LoRA Fine-Tuning (Parameter-Efficient)

- Learned **Low-Rank Adaptation (LoRA)** for parameter-efficient fine-tuning of large language models.
- Used **DistilBERT (distilbert-base-uncased)** for a simple text classification task.
- Configured LoRA adapters targeting attention layers (`q_lin`, `v_lin`) while freezing most of the base model.
- Fine-tuned using Hugging Face **Trainer API** on a toy dataset with minimal compute.
- Evaluated the fine-tuned model on unseen text samples to verify performance.
- Achieved near full fine-tuning performance while training only ~1–2% of parameters.

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
- [x] **Day 4:** Building a simple semantic search system  
- [x] **Day 5:** Introduction to fine-tuning (lightweight methods like LoRA/PEFT)  
- [ ] **Day 6:** Retrieval-Augmented Generation (RAG) basics  
- [ ] **Day 7:** Deploy a mini-project (Gradio web app or Flask API) using learned concepts

---

## 📖 References
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)  
- [Text Generation Strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)  
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Gradio](https://www.gradio.app/guides/quickstart)
- [Lora](https://arxiv.org/abs/2106.09685) | YT Videos: https://www.youtube.com/watch?v=KEv-F5UkhxU, https://www.youtube.com/watch?v=t1caDsMzWBk
  
---

## 🤝 Contributing
This repo is mainly for personal learning, but suggestions and improvements are welcome.  
Feel free to open issues or PRs!
