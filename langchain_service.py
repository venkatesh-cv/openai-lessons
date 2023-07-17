from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.schema import Document
from openai_config import openai_settings
from langchain.schema import SystemMessage, HumanMessage
from pg_vector_service import get_context, prime_database
from langchain.vectorstores import PGVector
from urllib.parse import quote_plus
import pg_vector_config
from langchain.vectorstores._pgvector_data_models import EmbeddingStore
from embedding_service import load_data
from pandas import DataFrame

def _init_chat_openai() -> ChatOpenAI:
    return ChatOpenAI(model_kwargs={"openai_api_type" : "azure", "engine":openai_settings.chat_engine, "deployment_name":openai_settings.chat_engine},
                       openai_api_base = openai_settings.base_url,
                       openai_api_key = openai_settings.api_key,
                       model="gpt-35-turbo",
                       temperature=0)

def _init_azure_openai() -> AzureOpenAI:
    return AzureOpenAI(openai_api_type = openai_settings.api_type, 
                 openai_api_base = openai_settings.base_url,
                 openai_api_version = openai_settings.api_version,
                 openai_api_key = openai_settings.api_key,
                 model="gpt-35-turbo",
                 max_tokens=100,
                 temperature=openai_settings.temperature,
                 deployment_name=openai_settings.chat_engine)

def _init_openai_embeddings() -> OpenAIEmbeddings:
    embeddings =  OpenAIEmbeddings(openai_api_base=openai_settings.base_url,
                            openai_api_type = openai_settings.api_type, 
                            openai_api_version = openai_settings.api_version,
                            openai_api_key = openai_settings.api_key,
                            model=openai_settings.embedding_engine,
                            embedding_ctx_length=1536, chunk_size=1                
                            )
    return embeddings


def zero_shot_prompt(querystr:str = "") -> None:
    prompt = PromptTemplate.from_template("Suggest a good name for a company that makes {attribute}")
    query = prompt.format(attribute = querystr)
    llm = _init_azure_openai()
    #print("-------------------P R E D I C T ----------------")
    #print(zero_shot_llm_predict(llm, query))
    #print("-------------------P R E D I C T   M E S S A G E S----------------")
    #result = zero_shot_llm_predict_messages(llm,"", query)
    #print(result)
    print("-------------------C H A I N   M E S S A G E S----------------")
    result = zero_shot_with_chain(llm=llm, prompt=prompt, query=querystr)
    print(result)

def zero_shot_llm_predict(llm:AzureOpenAI, query:str = "") -> str:
    return llm.predict(query)
    

def zero_shot_llm_predict_messages(llm:AzureOpenAI, context:str = "you are a healthcare industry and perl expert. you will suggest business friendly names for the attributes given in the prompt", query:str = "") -> str:
    return llm.predict_messages ([SystemMessage(content=context), HumanMessage(content=query)])

def zero_shot_with_chain(llm:AzureOpenAI, prompt:PromptTemplate, query:str = "") -> None:
    print(query)
    print(prompt)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(query)

def _connection_string() -> str:
    cstr =  PGVector.connection_string_from_db_params(
        driver="psycopg2",
        host = pg_vector_config.db_host_localhost,
        port = pg_vector_config.db_port,
        database=pg_vector_config.db_name,
        user = pg_vector_config.db_user,
        password=quote_plus(pg_vector_config.db_password))
    print(cstr)
    return cstr

def conversational_chat(llm:AzureOpenAI, query:str = "")  ->  any:
    db = PGVector(connection_string=_connection_string(), embedding_function=_init_openai_embeddings())
    df = load_data()
    list_of_documents = []
    df.apply(lambda row: list_of_documents.append(Document(page_content=row["context"], metadata={"tags":str(row["tags"]).split(",")})), axis = 1)
    db.add_documents(list_of_documents)
    result = db.similarity_search_with_score(query="who is Kamala Harris", k=3)
    return result

    

def example():
    model = _init_chat_openai()
    return model.predict(
    "Translate this sentence from English to French. I love programming."
    )

if __name__ == "__main__":
    print('lang chain ')
    #ßprint(zero_shot_prompt("socks"))
    print(conversational_chat(_init_chat_openai(), "Hello there"))
    #print(chat("Peter parker loves Mary jane \n Peter works for the morning herald as a photographer\n they married in 2010 \n they have a son knowles","what do you know about Peter parker"))