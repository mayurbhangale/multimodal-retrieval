# Multimodal Semantic Retrieval for Product Search ([arXiv:2501.07365](https://arxiv.org/abs/2501.07365))


> This repository contains an _unofficial implementation_ of the approach described in ["Multimodal Semantic Retrieval for Product Search"](https://arxiv.org/abs/2501.07365) by Dong Liu & Esther Lopez Ramos (2024) / Amazon.  
> Repo contains end-to-end prototype for semantic product search leveraging both natural language and visual features, supporting text-to-product and image-to-product retrieval via a shared semantic embedding space.

---

## Model Architecture

The paper proposes several multimodal retrieval architectures including 3-tower and 4-tower designs, where independent encoders process product images and text before fusion.

<div align="center">
  <img src="https://arxiv.org/html/2501.07365v3/extracted/6209627/figures/4tMMv2.png" alt="Tower Architecture" width="200"/>
  <img src="https://arxiv.org/html/2501.07365v3/extracted/6209627/figures/3tMMv2.png" alt="Tower Architecture" width="200"/>
  <br>
  <i>Figure: Example of the 3-tower and 4-tower architectures for multimodal semantic retrieval (source: arXiv:2501.07365).</i>
</div>

- **Text Tower:** Encodes product title/description with BERT.
- **Image Tower:** Encodes product image with CLIP ViT.
- **Fusion Tower:** Combines text and image embeddings (e.g., concatenation, MLP).
- **Query Tower:** Encodes user query (text) for retrieval.

---

## Core Concept

The core idea is to map both product queries and catalog items into a shared multimodal embedding space. This is achieved by fusing deep representations of text and images, enabling semantic search and retrieval across modalities.

**Key components:**
- **Text Encoder:** Transformer (BERT) for product text.
- **Image Encoder:** Vision Transformer (CLIP) for product images.
- **Fusion Module:** Projects and fuses the two modalities into a single vector.
- **Retrieval:** Uses cosine similarity in embedding space for ranking.

---

## Mathematical Formulation

Given:
- Query $q$
- Product text $p_\mathrm{text}$
- Product image $p_\mathrm{img}$

We define embedding functions:
- $f_\mathrm{text}(q)$: maps the query to the embedding space
- $f_\mathrm{mm}(p_\mathrm{text}, p_\mathrm{img})$: maps the product (with fused modalities) to the embedding space

The objective is to maximize similarity for true query-product pairs and minimize it for negatives, using the InfoNCE contrastive loss:

```math
\mathcal{L}_\mathrm{InfoNCE}
=
-\frac{1}{N} \sum_{i=1}^N \log
\frac{
    \exp\bigl(
      \mathrm{sim}\bigl(f_\mathrm{text}(q_i),\,f_\mathrm{mm}(p^i_\mathrm{text},\,p^i_\mathrm{img})\bigr)\,/\,\tau
    \bigr)
}{
    \sum_{j=1}^N \exp\bigl(
      \mathrm{sim}\bigl(f_\mathrm{text}(q_i),\,f_\mathrm{mm}(p^j_\mathrm{text},\,p^j_\mathrm{img})\bigr)\,/\,\tau
    \bigr)
}
```

Where:
- $\mathrm{sim}(\cdot, \cdot)$ is cosine similarity,
- $\tau$ is the temperature parameter,
- $N$ is the batch size.

---

## Quickstart

   ```
   git clone https://github.com/mayurbhangale/multimodal-retrieval.git
   ```

   ```
   cd multimodal-retrieval
   ```

   ```
   pip install -r requirements.txt
   ```

2. Train
```
python train.py \
  --csv https://raw.githubusercontent.com/luminati-io/eCommerce-dataset-samples/refs/heads/main/amazon-products.csv \
  --epochs 2 --batch_size 8 --max_samples 2000
```

3. Demo
```
python demo.py
```
---

## Reference

> **Dong Liu & Esther Lopez Ramos (2024) / Amazon: [Multimodal Semantic Retrieval for Product Search](https://arxiv.org/abs/2501.07365)**

---
