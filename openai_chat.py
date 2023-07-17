import openai
import pinecone
from openai_utils import get_embeddings, chat
from embedding_service import generate_adav2_embeddings, load_data
import pandas as pd
import pandas.core.frame as df
from pinecone_config import pinecone_settings
from openai_config import openai_settings
from pg_vector_service import prime_database, get_context

def chat_using_pgvector(query:str, tags:list[str]) -> None:
    #prime_database()
    # print("primed postgres database")
    # search("what do you know about fruits",3)
    context_results = get_context(query, tags)
    #print(context_results)
    print(chat(context=context_results, query=query))
if __name__ == "__main__":
    query = "write a hold rule that checks if a patient is an adult before approving a cosmetic procedure"
    tags = ["rules"]
    chat_using_pgvector(query, tags)