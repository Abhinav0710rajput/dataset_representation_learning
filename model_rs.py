from re import A
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *



class d2v_rs(nn.Module):
    def __init__(self, embedder):
        super().__init__()
        #if embedder is None
        if embedder is None:
            self.embedder = Dataset2Vec(input_dim=2, out_dim=32)
        else:
            self.embedder = embedder

        self.linear1 = nn.Linear(32, 16)
        self.linear2 = nn.Linear(16, 5)


    def forward(self, x: torch.Tensor, y: torch.Tensor):

        embedding =  self.embedder(x, y)
        a = self.linear1(embedding)
        a = F.relu(a)
        a = self.linear2(a)
        return embedding, a



if __name__ == "__main__":

    N = 20
    M = 10
    T = 1

    x = torch.rand(N, M)*20  # predictors
    y = torch.rand(N, T)*20  # targets

    print(x)
    print(y)

    model = d2v_rs(None)
    _, result = model(x, y)

    print(result)

    print("result shape:", result.shape)



"""

. ~/.bashrc; conda activate torchenv; python model_rs.py

"""