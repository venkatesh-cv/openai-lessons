import os

class Config():
    _index_name = None
    _api_key = None
    _env = None

    def __init__(self) -> None:
        self._index_name = "gen-qa-openai"
        self._api_key = os.getenv("PINECONE_API_KEY") or "PINECONE_API_KEY"
        self._env = os.getenv("PINECONE_ENVIRONMENT") or "PINECONE_ENVIRONMENT"


    @property
    def index_name(self) -> str:
        return self._index_name
    
    @property
    def api_key(self) -> str:
        return self._api_key
    
    @property
    def env(self) -> str:
        return self._env

pinecone_settings = Config()