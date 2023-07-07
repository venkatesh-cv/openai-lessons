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


def _insert_into_pgvectordb(df_statements:df.DataFrame):
    #print(df_statements)
    pg_embedding_dao.insert_embeddings(df_statements)

def _search(text:str, limit = 5, tags :list[str]= []):
    encoded_query = get_embeddings(text)
    result = pg_embedding_dao.query_embeddings(str(encoded_query), limit, tags)
    #print (result)
    return result

def _prime_database() -> None:
    df_statements = _get_embedded_data()
    _insert_into_pgvectordb(df_statements)

def _get_context(search_query:str, tags:list[str] = []) -> str:
    results = _search(search_query,limit=5, tags= tags)
    context_results = ' '.join(list(row[1] for row in results))
    return context_results

def chat_using_pgvector(query:str, tags:list[str]) -> None:
    _prime_database()
    # search("what do you know about fruits",3)
    context_results = _get_context(query, tags)
    #print(context_results)
    print(chat(context=context_results, query=query))
if __name__ == "__main__":
    query = "write a hold rule that checks if a patient is an adult before approving a cosmetic procedure"
    tags = ["rules"]
    chat_using_pgvector(query, tags)