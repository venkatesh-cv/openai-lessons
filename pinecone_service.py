import openai
import pinecone
from openai_utils import get_embeddings, chat
from embedding_service import generate_adav2_embeddings, load_data
import pandas as pd
from pinecone_config import pinecone_settings
from openai_config import openai_settings

index = None

def _get_embedded_data():
    df_statements = load_data()
    df_statements = generate_adav2_embeddings(df_statements)
    return df_statements;


def _init_pinecone_index() -> pinecone.GRPCIndex:
    pinecone.init(api_key=pinecone_settings.api_key, environment=pinecone_settings.env)
    # check if index already exists (it shouldn't if this is first time)
    if pinecone_settings.index_name not in pinecone.list_indexes():
        # if does not exist, create index
        pinecone.create_index(
            pinecone_settings.index_name,
            dimension=openai_settings.embeddings_length,
            metric='cosine',
            metadata_config={'indexed': ['channel_id', 'published']}
        )
    # connect to index
    index = pinecone.GRPCIndex(pinecone_settings.index_name)
    # view index stats
    # print(index.describe_index_stats())
    return index

def _insert_into_vectordb(df_statements, index :pinecone.GRPCIndex):
    index.upsert_from_dataframe(df_statements)

def search(text:str, limit = 5):
    if(index is None):
       print ("Initializing index")
       _init_pinecone_index()
    else:
        print("index is available")
    encoded_query = get_embeddings(text)
    result = index.query(encoded_query,top_k=limit, include_metadata=True)
    print(result)
    return result

def _prime_database(index:pinecone.GRPCIndex) -> None:
    df_statements = _get_embedded_data()
    df_statements["id"] = df_statements["id"].apply(lambda x: str(x))
    df_statements["metadata"] = df_statements["context"].apply(lambda x: {"text":x})
    df_statements = df_statements.drop(columns = ["context"])
    print(df_statements)
    _insert_into_vectordb(df_statements, index)

def get_context(search_query:str) -> None:
    results = search(search_query)
    matches = list(results["matches"])
    context_results = ' '.join(list(match["metadata"]["text"] for match in matches))
    return context_results
    

if __name__ == "__main__":
    index = _init_pinecone_index()
    #index.delete(delete_all=True)
    #_prime_database(index)
    #search("what do you know about fruits",3)
    context_results = get_context("what do you know about fruits")
    print(chat(context=context_results, query="what do you know about fruits"))