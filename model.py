import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim, depth=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(depth)
        ])

    def forward(self, x):
        residual = x
        for layer in self.layers:
            x = F.relu(layer(x))
        return x + residual

class Dataset2Vec(nn.Module):
    def __init__(self, input_dim=2, out_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim

        # f: 7 × [Dense(32); ResidualBlock(3, 32); Dense(32)]
        f_layers = []
        f_layers.append(nn.Linear(input_dim, 32))
        for _ in range(7):
            f_layers.append(ResidualBlock(32, depth=3))
            f_layers.append(nn.Linear(32, 32))
            f_layers.append(nn.ReLU())
        self.f = nn.Sequential(*f_layers)

        # g: Dense(32); Dense(16); Dense(8)
        self.g = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        # h: 3 × [Dense(16); ResidualBlock(3, 16); Dense(16)]
        h_layers = []
        h_layers.append(nn.Linear(8, 16))
        for _ in range(3):
            h_layers.append(ResidualBlock(16, depth=3))
            h_layers.append(nn.Linear(16, 16))
            h_layers.append(nn.ReLU())
        self.h = nn.Sequential(*h_layers)

        # Final output layer to map to out_dim
        self.out = nn.Linear(16, out_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Args:
            x: (N, M) - predictors
            y: (N, T) - targets
        Returns:
            meta-feature embedding vector of shape (out_dim,)
        """
        N, M = x.shape
        _, T = y.shape

        # Generate (x_{n,m}, y_{n,t}) pairs
        x_exp = x.unsqueeze(2).expand(N, M, T)  # (N, M, T)
        y_exp = y.unsqueeze(1).expand(N, M, T)  # (N, M, T)
        pair = torch.stack([x_exp, y_exp], dim=-1)  # (N, M, T, 2)

        # Reshape for batch processing through f: (N, M, T, 2) -> (N, M*T, 2)
        """
        pair = pair.view(N, M*T, 2)

        # Apply f to each (x_{n,m}, y_{n,t}) pair
        f_out = self.f(pair)  # (N, M*T, 32)
        """

        pair_flat = pair.view(-1, 2)  # (N*M*T, 2)
        f_out_flat = self.f(pair_flat)  # (N*M*T, 32)
        f_out = f_out_flat.view(N, M*T, 32)  # Reshape back


        # Average over N (samples), keeping each (m,t) pair separate
        f_mean = f_out.mean(dim=0)  # (M*T, 32)

        # Apply g to each (m,t) aggregated f-embedding
        g_out = self.g(f_mean)  # (M*T, 8)

        # Average over M*T (feature-target pairs)
        g_mean = g_out.mean(dim=0, keepdim=True)  # (1, 8)

        # Apply h and final output projection
        h_out = self.h(g_mean)  # (1, 16)
        meta_feat = self.out(h_out).squeeze(0)  # (out_dim,)

        meta_feat = F.normalize(meta_feat, p=2, dim=-1)

        return meta_feat




if __name__ == "__main__":
    N = 20
    M = 10
    T = 1

    x = torch.rand(N, M)*20  # predictors
    y = torch.rand(N, T)*20  # targets

    print(x)
    print(y)

    model = Dataset2Vec(input_dim=2, out_dim=32)
    embedding = model(x, y)

    print(embedding)

    print("Meta-feature embedding shape:", embedding.shape)



"""

. ~/.bashrc; conda activate torchenv; python model.py

"""