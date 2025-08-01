import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKPool2d(nn.Module):
    """Top-k pooling that takes top k values and averages them"""
    def __init__(self, k=3):
        super().__init__()
        self.k = k
    
    def forward(self, x):
        # x: (batch, channels, H, W)
        batch_size, channels, H, W = x.size()
        
        # flatten the spatial dimensions
        x_flat = x.view(batch_size, channels, -1)  # (batch, channels, H*W)

        # take top-k for each channel
        topk_vals, _ = torch.topk(x_flat, self.k, dim=2)  # (batch, channels, k)

        # average pooling to (1,1)
        result = topk_vals.mean(dim=2, keepdim=True)  # (batch, channels, 1)
        result = result.unsqueeze(-1)  # (batch, channels, 1, 1)
        
        return result

class AdaptiveKPool2d(nn.Module):
    """Adaptive-k pooling based on activation distribution"""
    def __init__(self, max_k=10, threshold=0.1):
        super().__init__()
        self.max_k = max_k
        self.threshold = threshold
    
    def forward(self, x):
        batch_size, channels, H, W = x.size()
        x_flat = x.view(batch_size, channels, -1)
        
        results = []
        for b in range(batch_size):
            batch_results = []
            for c in range(channels):
                channel_vals = x_flat[b, c, :]
                max_val = channel_vals.max()
                
                # dynamically determine k: take all values greater than max_val * threshold
                mask = channel_vals >= (max_val * self.threshold)
                k = min(mask.sum().item(), self.max_k)
                k = max(k, 1)  # ensure k is at least 1
                
                topk_vals, _ = torch.topk(channel_vals, k)
                batch_results.append(topk_vals.mean())
            
            results.append(torch.stack(batch_results))
        
        result = torch.stack(results).unsqueeze(-1).unsqueeze(-1)
        return result