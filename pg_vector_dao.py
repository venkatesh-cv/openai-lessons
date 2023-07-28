import psycopg2
from psycopg2.extras import execute_values
import pg_vector_config
import pandas as pd


def connection():
    conn = psycopg2.connect(database=pg_vector_config.db_name,
                        host=pg_vector_config.db_host_localhost,
                        user=pg_vector_config.db_user,
                        password=pg_vector_config.db_password,
                        port=pg_vector_config.db_port)
    return conn

def insert_embeddings(embeddings_to_insert:pd.core.frame.DataFrame) -> None:
    rows = []
    #print(embeddings_to_insert)
    embeddings_to_insert.apply(lambda item: rows.append((str(item["values"]), item["context"],item["response"], item["tags"])), axis = 1)
    conn = connection()
    cleanup = "truncate table bca"
    command = "insert into bca(embedding,text,response,tags) values %s"
    success = False
    try:
        #print(rows)
        conn.cursor().execute(cleanup)
        execute_values(cur=conn.cursor(), sql=command, argslist=rows)
        conn.commit()
        success = True
        #print(rows)
    finally:
        if(success != True):
           print("Rollback as operation failed")
           conn.rollback()
        conn.close()
        
#pg_embedding_dao.query_embeddings(encoded_query, limit, True, tags)
def query_embeddings(query:str, limit:int = 5, tags:list[str] = []):
    conn = connection()
    try:
        cursor = conn.cursor()
        #SELECT * FROM bca ORDER BY embedding <-> '[3,1,2]' LIMIT 5;
        cursor.execute("select id,text,response,tags, embedding from bca order by embedding <-> %s limit %s ", (query, str(limit)))
        result = cursor.fetchall() 
        # print(result)
        return result
    finally:
        conn.close()