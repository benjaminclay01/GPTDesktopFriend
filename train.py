# GET INPUT TRAINING FILE
# TODO: Write custom input file for training
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# READ IT IN AS STRING
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# GET ALL THE UNIQUE CHARACTERS
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# TOKENIZE INPUT TEXT
# TODO: implement TikToken
sToI = {ch:i for i,ch in enumerate(chars) }
iIoS = i:ch for i,ch in enumerate(chars) }
encode = lambda s: [sToI[c] for c in s]
decode = lambda l: ''.join([iToS[i] for i in l])

# ENCODE ENTIRE DATA SET AND STORE IN TENSOR
import torch 
data = torch.tensor(encode(text), dtype=torch.long)

# SPLIT INTO TRAIN AND TEST 
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]


