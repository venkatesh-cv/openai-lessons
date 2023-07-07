import pinecone_service
import pg_vector_service
if __name__ == "__main__":
    query = "who are peter and mary"
    tags = ["marvel"]
    print("************** PINE CONE SEARCH ******************")
    pinecone_service.chat_using_pinecone(query,tags)
    print("************** PG VECTOR SEARCH ******************")
    pg_vector_service.chat_using_pgvector(query=query, tags=tags)
