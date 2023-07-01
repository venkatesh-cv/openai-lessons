import openai
import os
import re
import requests
import sys
from num2words import num2words
import os
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
from openai_config import settings, configure_openai
import tiktoken

# This encodes the reference data as BPE tokens for GPT models. 
# This is not the vectorized values. But a preprocessing step
def generate_bpe_tokens(df_to_embed) -> pd.core.frame.DataFrame:
    tokenizer = tiktoken.encoding_for_model(settings().bpe_encoding_for_model)
    df_to_embed['n_tokens'] = df_to_embed["context"].apply(lambda x: len(tokenizer.encode(x)))
    df_to_embed = df_to_embed[df_to_embed.n_tokens<8192]
    df_to_embed = df_to_embed.drop(columns = ["n_tokens"])
    return df_to_embed


def generate_adav2_embeddings(df_to_embed: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    df_to_embed = generate_bpe_tokens(df_to_embed)
    df_to_embed['values'] = df_to_embed["context"].apply(lambda x : get_embedding(x, engine = settings().embedding_engine)) # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    return df_to_embed


def load_data() ->  pd.core.frame.DataFrame:
    df_statements=pd.read_csv(os.path.join(os.getcwd(),'data_to_embed.csv')) # This assumes that you have placed the bill_sum_data.csv in the same directory you are running Jupyter Notebooks
    pd.options.mode.chained_assignment = None
    return df_statements

if(__name__ == "__main__"):
    configure_openai(openai)
    df_statements = load_data()
    df_statements = generate_adav2_embeddings(df_statements)
    print(df_statements)