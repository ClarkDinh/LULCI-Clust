
import torch
import torch.nn.functional as F

class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5, device=None):
        super().__init__()
        self.temperature = temperature
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, z1, z2):
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)

        similarity = torch.matmul(z, z.T) / self.temperature
        batch_size = z1.size(0)

        labels = torch.arange(batch_size, device=self.device)
        labels = torch.cat([labels, labels], dim=0)

        mask = torch.eye(batch_size * 2, device=self.device).bool()
        similarity = similarity.masked_fill(mask, -1e9)

        loss = F.cross_entropy(similarity, labels)
        return loss
