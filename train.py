import torch
from torch.utils.data import DataLoader
from data_loader import AmazonProductDataset
from models import FourTowerModel, nt_xent_loss

def train(csv_path, device, epochs=2, batch_size=8, max_samples=2000):
    """
    Train the 4-tower multimodal model using NT-Xent loss.
    - csv_path: path or URL to CSV
    - device: 'cuda' or 'cpu'
    """
    ds = AmazonProductDataset(csv_path, max_samples=max_samples)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=lambda x: x)

    model = FourTowerModel(device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_items in dl:
            # Build a minimal batch dict
            batch = {
                'query_texts': [item['title'] for item in batch_items],  # use title as proxy for query
                'titles':      [item['title'] for item in batch_items],
                'descs':       [item['desc'] for item in batch_items],
                'images':      [item['image'] for item in batch_items],
            }

            optimizer.zero_grad()
            query_embs, prod_embs = model(batch)  # both are (B,512)
            loss = nt_xent_loss(query_embs, prod_embs, temperature=0.07)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dl)
        print(f"Epoch {epoch+1}/{epochs} — Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "multimodal_model.pt")
    print("Training complete. Model saved to multimodal_model.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="https://raw.githubusercontent.com/luminati-io/eCommerce-dataset-samples/refs/heads/main/amazon-products.csv")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=2000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args.csv, device, args.epochs, args.batch_size, args.max_samples)