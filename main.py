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
    
    if mask is not None:
      energy = energy.masked_fill(mask == 0, float("-1e20")) #this float means minus infinity

      attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3) #normalizing the keylength
      out = torch.einsum("nhql,nlhd -> nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
      # attention shape : N, heads n querylen ,keylen
      # value shape : N, vale_len , heads , heads_dim
      # after einsum (N, query_len , heads , head_Dim) then flatten the last two dimensions

      out = self.fc_out(out)
      return out

#create the transformer block

class transformerblock(nn.Module):
  def __init__(self, embed_size, heads , dropout, forward_expansion):
    super(transformerblock, self).__init__()
    self.attention = selfattention(embed_size, heads)
    self.norm1 = nn.LayerNorm(embed_size)
    self.norm2 = nn.LayerNorm(embed_size)

    self.feed_forward = nn.Sequential(
      nn.Linear(embed_size, forward_expansion*embed_size),
      nn.ReLU(),
      nn.Linear(forward_expansion*embed_size, embed_size)
    )
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, value, key , query, mask):
    attention = self.attention(value, key , query, mask)

    x= self.dropout(self.norm(attention+query))
    forward = self.feed_forward(x)
    out = self.dropout(self.norm2(forward + x))
    return out
  
class encoder(nn.Module):
  def __init__(
      self,
      src_vocab_size,
      embed_size,
      num_layers,
      heads,
      device,
      forward_expansion,
      dropout,
      max_length,
  ):
    super(encoder, self).__init__()
    self.embed_size = embed_size
    self.device = device
    self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
    self.position_embedding = nn.Embedding(max_length, embed_size)

    self.layers = nn.ModuleList(
      [
        transformerblock(
          embed_size,
          heads,
          dropout=dropout,
          forward_expansion=forward_expansion,
        )
      ]
    )
    self.dropout = nn.Dropout(dropout)
  def forward(self, x, mask):
    N, seq_length = x.shape
    positions = torch.arrange(0, seq_length).expand(N, seq_length).to(self.device)

    out= self.dropout(self.word_embedding(x) + self.position_embedding(positions))
    
    for layer in self.layers:
      out = layer(out , out , out, mask)
    return out


#only decoder is yet to be made
