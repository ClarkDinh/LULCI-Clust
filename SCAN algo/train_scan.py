import torch
from torch.nn import functional as F
from tqdm import tqdm

def scan_loss(preds, neighbors, entropy_weight=5.0):
    consistency_loss = 0
    for i, neighs in enumerate(neighbors):
        for j in neighs:
            consistency_loss += -torch.log(torch.sum(preds[i] * preds[j] + 1e-8))
    consistency_loss /= len(neighbors) * len(neighbors[0])

    avg_preds = torch.mean(preds, dim=0)
    entropy_loss = torch.sum(avg_preds * torch.log(avg_preds + 1e-8))
    return consistency_loss + entropy_weight * entropy_loss

def train_scan(features, neighbors, cluster_head, epochs=100, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cluster_head.to(device)
    cluster_head.train()
    optimizer = torch.optim.Adam(cluster_head.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for i in tqdm(range(0, len(features), 256), desc=f"Epoch {epoch+1}/{epochs}"):
            batch_feat = features[i:i+256].to(device)  # ✅ use .to(device)
            batch_neighbors = neighbors[i:i+256]
            preds = cluster_head(batch_feat)

            loss = scan_loss(preds, batch_neighbors)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[SCAN] Epoch {epoch+1}: Loss = {total_loss:.4f}")