import torch
import torch.nn as nn
import torch.nn.functional as F

# Mini vocabulario
vocab = ["hola","mundo","como","estas","adios","todo","bien"]
word2idx = {w:i for i,w in enumerate(vocab)}
idx2word = {i:w for i,w in enumerate(vocab)}
vocab_size = len(vocab)
embed_size = 16
hidden_size = 32

# Modelo mini Transformer
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=2)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)            # Convertir tokens a vectores
        x = x.unsqueeze(1)              # Forma (seq_len, batch, embed)
        x, _ = self.attention(x, x, x)  # Atención
        x = x.squeeze(1)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

# Instanciar modelo
model = MiniGPT()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.NLLLoss()

# Datos de entrenamiento (ejemplo)
frases = [["hola","mundo"], ["como","estas"], ["todo","bien"], ["adios"]]
sequences = []
targets = []

for frase in frases:
    for i in range(len(frase)-1):
        sequences.append([word2idx[frase[i]]])
        targets.append(word2idx[frase[i+1]])

sequences = torch.tensor(sequences)
targets = torch.tensor(targets)

# Entrenamiento mini
for epoch in range(500):
    optimizer.zero_grad()
    output = model(sequences)
    loss = loss_fn(output, targets)
    loss.backward()
    optimizer.step()

print("Mini GPT entrenado!")
