import torch

from flair.data import Sentence
from flair.embeddings import (
    DocumentPoolEmbeddings,
    FlairEmbeddings, WordEmbeddings,
    BertEmbeddings
)

embeddings = DocumentPoolEmbeddings([
    WordEmbeddings("glove"),
    FlairEmbeddings("news-forward"),
    FlairEmbeddings("news-backward"),
    BertEmbeddings(),
])

query = Sentence("I love Berlin")

paragraph_1 = Sentence("Paris is an interesting city.")
paragraph_2 = Sentence("The computer is new.")

embeddings.embed([query, paragraph_1, paragraph_2])

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

similartity_to_paragraph_1 = cos(query.embedding, paragraph_1.embedding)
print(similartity_to_paragraph_1)
