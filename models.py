import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import clip

BIBERT_NAME = "microsoft/BiBERT-base-uncased"
CLIP_MODEL_NAME = "ViT-B/32"

class TextTower(nn.Module):
    """
    BiBERT-based text encoder (for query, title, description).
    """
    def __init__(self, model_name=BIBERT_NAME, out_dim=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.out_dim = out_dim

    def forward(self, texts, device):
        # texts: list of strings
        tokens = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)
        outputs = self.encoder(**tokens)
        # Use [CLS] token embedding
        return outputs.last_hidden_state[:, 0, :]  # shape: (batch, out_dim)

class CLIPImageTower(nn.Module):
    """
    CLIP ViT-B/32 image encoder (visual tower only).
    """
    def __init__(self, device, out_dim=512):
        super().__init__()
        self.device = device
        # Load full CLIP but keep only the visual branch
        self.clip_model, self.preprocess = clip.load(CLIP_MODEL_NAME, device=device, jit=False)
        self.visual = self.clip_model.visual
        self.out_dim = out_dim

    def forward(self, pil_images, device):
        # pil_images: list of PIL.Image or None
        images = []
        for img in pil_images:
            if img is not None:
                images.append(self.preprocess(img))
            else:
                # substitute a zero‐tensor for missing images
                images.append(torch.zeros(3, 224, 224))
        images = torch.stack(images).to(device)
        with torch.no_grad():
            img_features = self.visual(images)
        return img_features  # shape: (batch, out_dim)

class FourTowerModel(nn.Module):
    """
    4-tower multimodal model:
      • Query tower (BiBERT)
      • Title tower (BiBERT)
      • Description tower (BiBERT)
      • Image tower (CLIP ViT-B/32)
    Fusion by concatenation & linear projection → 512-dim joint space.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.text_dim = 768   # BiBERT output dim
        self.img_dim = 512    # CLIP ViT-B/32 output dim
        
        # Text towers
        self.query_tower = TextTower(model_name=BIBERT_NAME, out_dim=self.text_dim)
        self.title_tower = TextTower(model_name=BIBERT_NAME, out_dim=self.text_dim)
        self.desc_tower  = TextTower(model_name=BIBERT_NAME, out_dim=self.text_dim)
        
        # Image tower
        self.image_tower = CLIPImageTower(device=device, out_dim=self.img_dim)
        
        # Fusion: concat(title, desc, image) project to 512
        fusion_in  = 2 * self.text_dim + self.img_dim  # title + desc + image
        fusion_in_q = 3 * self.text_dim + self.img_dim  # query + title + desc + image
        fusion_out = 512
        
        self.fusion_prod = nn.Linear(fusion_in, fusion_out)
        self.fusion_query = nn.Linear(fusion_in_q, fusion_out)
        self.norm = nn.LayerNorm(fusion_out)

    def forward(self, batch):
        """
        batch should be a dict containing:
          • 'query_texts': list[str] (batch_size)
          • 'titles':      list[str] (batch_size)
          • 'descs':       list[str] (batch_size)
          • 'images':      list[PIL.Image or None] (batch_size)
        Returns:
          • query_embs:   Tensor (batch_size, 512)
          • prod_embs:    Tensor (batch_size, 512)
        """
        # Encode text
        q_emb = self.query_tower(batch['query_texts'], self.device)   # (B,768)
        t_emb = self.title_tower(batch['titles'], self.device)        # (B,768)
        d_emb = self.desc_tower(batch['descs'], self.device)          # (B,768)
        # Encode image
        i_emb = self.image_tower(batch['images'], self.device)        # (B,512)
        
        # Product fusion: concat(title, desc, image) to 512
        prod_cat = torch.cat([t_emb, d_emb, i_emb], dim=1)            # (B, 768+768+512)
        prod_proj = self.fusion_prod(prod_cat)                        # (B,512)
        prod_norm = self.norm(prod_proj)
        
        # Query fusion: concat(query, title, desc, image) to 512
        query_cat = torch.cat([q_emb, t_emb, d_emb, i_emb], dim=1)    # (B, 768+768+768+512)
        query_proj = self.fusion_query(query_cat)                     # (B,512)
        query_norm = self.norm(query_proj)
        
        return query_norm, prod_norm

    def encode_query(self, queries):
        """
        Only encode query text → 768-dim BiBERT embed.
        (For retrieval, you can optionally fuse with dummy product towers,
        but here we return just BiBERT CLS.)
        """
        return self.query_tower(queries, self.device)  # (batch,768)

    def encode_product(self, titles, descs, images):
        """
        Encode product towers (title+desc+image) → 512-dim fused embed.
        """
        t_emb = self.title_tower(titles, self.device)
        d_emb = self.desc_tower(descs, self.device)
        i_emb = self.image_tower(images, self.device)
        prod_cat = torch.cat([t_emb, d_emb, i_emb], dim=1)
        prod_proj = self.fusion_prod(prod_cat)
        return self.norm(prod_proj)  # (batch,512)

def nt_xent_loss(query_embs, prod_embs, temperature=0.07):
    """
    NT-Xent loss (SimCLR-style), as in Section 3.3 of the paper.
    - query_embs: (N, 512)
    - prod_embs:  (N, 512)
    """
    batch_size = query_embs.size(0)
    device = query_embs.device

    # L2 normalize
    q_norm = F.normalize(query_embs, dim=1)   # (N,512)
    p_norm = F.normalize(prod_embs, dim=1)    # (N,512)

    # Similarity matrix: (N, N)
    logits = torch.mm(q_norm, p_norm.t()) / temperature  # (N,N)

    labels = torch.arange(batch_size, device=device)     # (N,)
    loss_q2p = F.cross_entropy(logits, labels)
    loss_p2q = F.cross_entropy(logits.t(), labels)
    return (loss_q2p + loss_p2q) / 2