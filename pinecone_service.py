import openai
import pinecone
from openai_utils import get_embeddings, chat
from embedding_service import generate_adav2_embeddings, load_data
import pandas as pd
import pandas.core.frame as df
from pinecone_config import pinecone_settings
from openai_config import openai_settings

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
            metadata_config={'indexed': ['channel_id', 'published']}
        )
    # connect to index
    index = pinecone.GRPCIndex(pinecone_settings.index_name)
    # view index stats
    # print(index.describe_index_stats())
    return index

def _insert_into_vectordb(df_statements: df.DataFrame, index :pinecone.GRPCIndex):
    #df_as_dict = df_statements.to_dict('records')
    #index.upsert(vectors=df_as_dict)
    index.upsert_from_dataframe(df_statements)

def search(text:str, limit = 5, tags :list[str]= []):
    if(index is None):
       print ("Initializing index")
       _init_pinecone_index()
    else:
        print("index is available")
    encoded_query = get_embeddings(text)
    #result = index.query(encoded_query,top_k=limit, include_metadata=True, filter={"tags":{"$eq":"fruits"}})
    result = index.query(encoded_query,top_k=limit, include_metadata=True)
    print (result)
    return result

def _prime_database(index:pinecone.GRPCIndex) -> None:
    df_statements = _get_embedded_data()
    df_statements["id"] = df_statements["id"].apply(lambda x: str(x))
    df_statements["metadata"] = df_statements.apply(lambda row: {"text":row["context"], "tags":row["tags"]}, axis = 1)
    df_statements = df_statements.drop(columns = ["context","tags"])
    print(df_statements["metadata"])
    _insert_into_vectordb(df_statements, index)

def get_context(search_query:str, tags:list[str] = []) -> str:
    results = search(search_query,limit=4, tags= tags)
    matches = list(results["matches"])
    context_results = ' '.join(list(match["metadata"]["text"] for match in matches))
    return context_results
    

if __name__ == "__main__":
    index = _init_pinecone_index()
    index.delete(delete_all=True)
    _prime_database(index)
    # search("what do you know about fruits",3)
    query = "write a hold rule that checks if a patient is an adult for any cosmetic surgery related procedure. Print just the rule"
    tags = ["rules"]
    context_results = get_context(query, tags)
    #print(context_results)
    print(chat(context=context_results, query=query))