import gradio as gr
import torch
from data_loader import AmazonProductDataset
from models import FourTowerModel
import torch.nn.functional as F

csv_path = "https://raw.githubusercontent.com/luminati-io/eCommerce-dataset-samples/refs/heads/main/amazon-products.csv"
ds = AmazonProductDataset(csv_path, max_samples=300)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FourTowerModel(device).to(device)
model.load_state_dict(torch.load("multimodal_model.pt", map_location=device))
model.eval()

# Precompute product embeddings once for demo speed
product_embeddings = []
product_meta = []

with torch.no_grad():
    # Process in small batches
    batch_size = 32
    for i in range(0, len(ds), batch_size):
        slice_items = ds[i : i + batch_size]
        titles = [item['title'] for item in slice_items]
        descs  = [item['desc']  for item in slice_items]
        images = [item['image'] for item in slice_items]

        prod_embs = model.encode_product(titles, descs, images)  # (b,512)
        product_embeddings.append(prod_embs)
        product_meta.extend(slice_items)

    product_embeddings = torch.cat(product_embeddings, dim=0)  # (N,512)

def search(query, top_k=5):
    """
    Given a text query, encode it (BiBERT), 
    compute similarity to all product embeddings, return top_k.
    """
    with torch.no_grad():
        # Encode query: just BiBERT CLS to project/dummy
        q_text = [query]
        q_emb_768 = model.query_tower(q_text, device)  # (1,768)

        # To compare in same 512-dim space, we need to fuse query with placeholders.
        # For simplicity: we’ll encode products via full encode_product; queries via forward()
        # so: build a fake “batch” reusing a random product’s title/desc/image for shape
        # But the paper recommends using query_fused. We can do:
        #   - Retrieve a random sample of size 1 from ds, get its title/desc/image,
        #   - duplicate it to match query batch, then call model.forward to get query_fused
        # A simpler hack: duplicate first product as “dummy” just to fuse.
        dummy = ds[0]
        dummy_title = dummy['title']
        dummy_desc  = dummy['desc']
        dummy_img   = dummy['image']

        # Build a one-element batch
        batch_q = {
            'query_texts': [query],
            'titles':      [dummy_title],
            'descs':       [dummy_desc],
            'images':      [dummy_img],
        }
        q_fused, _ = model(batch_q)  # q_fused is (1,512)

        # Compute cosine similarity to all product embeddings
        q_norm = F.normalize(q_fused, dim=1)           # (1,512)
        p_norm = F.normalize(product_embeddings, dim=1) # (N,512)
        sims = torch.mm(q_norm, p_norm.t()).squeeze(0)  # (N,)

        # Top-k
        topk_vals, topk_inds = torch.topk(sims, top_k)
        results = []
        for score, idx in zip(topk_vals.tolist(), topk_inds.tolist()):
            item = product_meta[idx]
            results.append({
                "title": item['title'],
                "asin": item['asin'],
                "score": f"{score:.3f}",
                "image": item['image']
            })
        return results

iface = gr.Interface(
    fn=search,
    inputs=[
        gr.Textbox(label="Enter search query"),
        gr.Slider(minimum=1, maximum=10, step=1, label="Top K")
    ],
    outputs=gr.JSON(label="Top-K Results"),
    title="Multimodal Product Search Demo",
    description="Type a product query; the model returns top-K matching Amazon products (using BiBERT + CLIP)."
)

if __name__ == "__main__":
    iface.launch()