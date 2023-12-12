import torch 
import torch.nn as nn
from torch.nn import functional as F


# GET INPUT TRAINING FILE
# TODO: Write custom input file for training
f = open("input.txt", "r")


# READ IT IN AS STRING
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# GET ALL THE UNIQUE CHARACTERS
chars = sorted(list(set(text)))
vocab_size = len(chars)

# TOKENIZE INPUT TEXT
# TODO: implement TikToken
sToI = {ch:i for i,ch in enumerate(chars) }
iToS = {i:ch for i,ch in enumerate(chars) }
encode = lambda s: [sToI[c] for c in s]
decode = lambda l: ''.join([iToS[i] for i in l])

# ENCODE ENTIRE DATA SET AND STORE IN TENSOR
data = torch.tensor(encode(text), dtype=torch.long)

# SPLIT INTO TRAIN AND TEST 
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

# BUILD CONTEXT WITH BLOCKS
block_size = 8
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]

# BUILD BATCHES TO TRAIN SEVERAL BLOCKS AT THE SAME TIME
torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    #generate small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

xb, yb = get_batch('train')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t + 1]
        target = yb[b, t]

#TRAIN THE MODEL
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        #each token reads off logits for next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        #idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss=None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss  = m(xb, yb)

print(decode(m.generate(idx=torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
