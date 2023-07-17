import openai
import pinecone
from openai_utils import get_embeddings, chat
from embedding_service import generate_adav2_embeddings, load_data
import pandas as pd
import pandas.core.frame as df
from pinecone_config import pinecone_settings
from openai_config import openai_settings
import pg_embedding_dao

index = None

def _get_embedded_data() -> pd.core.frame.DataFrame:
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
            metadata_config={'indexed': ['tags']}
        )
    # connect to index
    index = pinecone.GRPCIndex(pinecone_settings.index_name)
    # view index stats
    return index

def _insert_into_vectordb(df_statements: df.DataFrame, index :pinecone.GRPCIndex):
    df_statements["id"] = df_statements["id"].apply(lambda x: str(x))
    df_statements["metadata"] = df_statements.apply(lambda row: {"text":row["context"], "tags":str(row["tags"]).split(",")}, axis = 1)
    df_statements = df_statements.drop(columns = ["context","tags"])
    index.upsert_from_dataframe(df_statements)

def _search(index:pinecone.GRPCIndex, text:str, limit = 5, tags :list[str]= []):
    if(index is None):
       print ("Initializing index")
       _init_pinecone_index()
    else:
        print("index is available")
    encoded_query = get_embeddings(text)
    result = index.query(encoded_query,top_k=limit, include_metadata=True, filter={"tags":{"$in":tags}})
    return result

def _prime_database(index:pinecone.GRPCIndex) -> None:
    df_statements = _get_embedded_data()
    _insert_into_vectordb(df_statements, index)

def _get_context(index:pinecone.GRPCIndex, search_query:str, tags:list[str] = []) -> str:
    results = _search(index, search_query,limit=5, tags= tags)
    matches = list(results["matches"])
    context_results = ' '.join(list(match["metadata"]["text"] for match in matches))
    return context_results

def chat_using_pinecone(query:str, tags:list[str] = []) -> None:
    index = _init_pinecone_index()
    print("initialized pinecone")
    index.delete(delete_all=True)
    _prime_database(index)
    print("primed pinecone vector database")
    context_results = _get_context(index,query, tags)
    print(chat(context=context_results, query=query))

if __name__ == "__main__":
    query = "write a hold rule that checks if a patient is an adult before approving cosmetic procedures"
    tags = ["rules"]
    chat_using_pinecone(query,tags)
    