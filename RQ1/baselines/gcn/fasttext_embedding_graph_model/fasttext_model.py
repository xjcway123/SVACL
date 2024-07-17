
import torch
import torch.nn as nn
import torch.nn.functional as F

class FastTextEmbedding(nn.Module):
    def __init__(self , vocab_size = 10000 , n_gram_vocab_size = 250499 , embed_size = 600 , hidden_size = 1024 , dropout = 0.2  ):
        super(FastTextEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=1)
        self.embedding_ngram2 = nn.Embedding(n_gram_vocab_size, embed_size)
        self.embedding_ngram3 = nn.Embedding(n_gram_vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_size * 3, hidden_size)

    def forward(self, x):
        """
            input   1-gram 2-gram 3-gram
            return  [batch_size,hidden_size]
        """
        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[1])
        out_trigram = self.embedding_ngram3(x[2])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)    # [128, 32, 900]

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)   # [batch_size,hidden_size]
        return out
