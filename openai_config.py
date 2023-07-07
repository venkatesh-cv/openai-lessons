import os
import openai
class OpenAI_Config:
    _endpoint = "https://oainpusegaistudygroup01.openai.azure.com/"
    _embedding_engine = None
    _chat_engine = None
    _api_key = None
    _api_type = None
    _api_version = None
    _deployment_endpoint = None
    _bpe_encoding_for_model = None
    _embeddings_length = None
    _temperature = None

    def __init__(self) -> None:
        self._api_key = os.getenv("OPENAI_API_KEY")
        self._api_type = "azure"
        self._api_version = "2023-03-15-preview"
        self._chat_engine = "gpt-35-turbo-0301"
        self._embedding_engine = "text-embedding-ada-002"
        self._deployment_endpoint = self._endpoint+"/openai/deployments?api-version=" + self._api_version 
        self._bpe_encoding_for_model = "gpt-4"
        self._embeddings_length = 1536
        self._temperature = 0.2

    @property
    def endpoint(self) -> str:
        return self._endpoint
    
    @property
    def embedding_engine(self) -> str:
        return self._embedding_engine
    
    @property
    def chat_engine(self) -> str:
        return self._chat_engine
    
    @property
    def api_key(self) -> str:
        return self._api_key
    
    @property
    def api_type(self) -> str:
        return self._api_type
    
    @property
    def api_version(self) -> str:
        return self._api_version
    
    @property
    def bpe_encoding_for_model(self) -> str:
        return self._bpe_encoding_for_model
    
    @property
    def embeddings_length(self) -> int:
        return self._embeddings_length
    
    @property
    def temperature(self) -> float:
        return self._temperature

openai_settings = OpenAI_Config()
def settings() -> OpenAI_Config:
    return openai_settings


def configure_openai(openai: openai) -> openai:
    openai.api_type = openai_settings.api_type
    openai.api_base = openai_settings.endpoint
    openai.api_version = openai_settings.api_version
    openai.api_key = openai_settings.api_key
    return openai

if __name__ == "__main__" :
    print (openai_settings.chat_engine)
    print (openai_settings.embedding_engine)
    print (openai_settings.endpoint)
    print (openai_settings.api_key)