import pandas as pd

import torch

from flair.data import Sentence
from flair.embeddings import (
    DocumentPoolEmbeddings,
    FlairEmbeddings,
    ELMoEmbeddings, BertEmbeddings
)


def preproc(text, limit=512):
    return (
        text.replace("[", "")
            .replace("]", "")
            .lower()[:limit]
    )

def try_sentence(text):
    try:
        return Sentence(text)
    except:
        pass


def main():
    print("Instantiate embeddings")
    embeddings = DocumentPoolEmbeddings([
        FlairEmbeddings("pubmed-forward"),
        FlairEmbeddings("pubmed-backward"),
        ELMoEmbeddings("pubmed"),
        BertEmbeddings("bert-large-uncased"),
    ])

    print("Load pubmed_data")
    pubmed_data = pd.concat([
        pd.read_json(f"data/medline/{medline_file}") for medline_file in
        ["medline_2016.json", "medline_2017.json", "medline_2018.json"]
    ])

    print("pubmed_corpus")
    pubmed_corpus = [
        try_sentence(text) for text in
        pubmed_data.title.apply(preproc).head(10_000)
    ]
    pubmed_corpus = [
        text for text in pubmed_corpus if text
    ]
    print(pubmed_corpus[0:5])

    print("query")
    query = [
        Sentence(text) for text in
        [
            "Searching for the causal effects of body mass index in over 300 000 participants in UK Biobank, using Mendelian randomization.",
            "Prioritizing putative influential genes in cardiovascular disease susceptibility by applying tissue-specific Mendelian randomization.",
            "Longitudinal analysis strategies for modelling epigenetic trajectories",
            "FATHMM-XF: accurate prediction of pathogenic point mutations via extended features",
            "PhenoSpD: an integrated toolkit for phenotypic correlation estimation and multiple testing correction using GWAS summary statistics.",
            "LD Hub: a centralized database and web interface to perform LD score regression that maximizes the potential of summary level GWAS data for SNP heritability and genetic correlation analysis.",
            "MELODI: Mining Enriched Literature Objects to Derive Intermediates",
            "The MR-Base platform supports systematic causal inference across the human phenome",
        ]
    ]

    print("Embed")
    for text in query + pubmed_corpus:
        embeddings.embed(text)

    print("Calculate scores")
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-5)
    cos_scores = []
    for query_id, query_text in enumerate(query):
        cos_res = [
            {
                "query_id": query_id,
                "target_id": target_id,
                "score": cos(query_text.embedding,
                             target_text.embedding).item()
            }
            for target_id, target_text in enumerate(pubmed_corpus)
        ]
        cos_scores.append(cos_res)

    cos_scores = pd.concat(pd.DataFrame(x) for x in cos_scores)
    print(cos_scores)

    n = 5
    for query_id, query_text in enumerate(query):
        print(f"# Query {query_id}")
        print(query_text)

        top_n = (
            cos_scores
              .query(f"query_id == {query_id}")
              .sort_values("score", ascending=False)
              .head(n)
        )
        for target_id, target_score in zip(top_n.target_id, top_n.score):
            print(f"  ## Candidate {target_id}, score {target_score}")
            print(f"  {pubmed_corpus[target_id]}\n")
        print("\n\n")

if __name__ == "__main__":
    main()
