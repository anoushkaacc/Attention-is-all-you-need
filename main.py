import torch 
import torch.nn as nn

class selfattention(nn.module):
  def __init__(self, embed_size, heads):
    super(selfattention, self).__init__()
    self.embed_size = embed_size
    self.heads = heads
    self.head_dim = embed_size // heads

    assert (self.head_dim * heads == embed_size), "embed size needs to be div by heads"
    self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

  def forward(self, values, keys, query, mask): 
    N = query.shape[0]
    value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

    # split into heads
    values = values.reshape(N, value_len, self.heads, self.head_dim)
    keys = keys.reshape(N, key_len, self.heads, self.head_dim)
    queries = query.reshape(N, key_len, self.heads, self.head_dim)

    energy = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])
    # queries shape: (N, querylen , heads , heads_dim)
    # keys shape: (N, key_len, heads, heads_dim)
    # energy shape : (N, heads, query_len, key_len)




