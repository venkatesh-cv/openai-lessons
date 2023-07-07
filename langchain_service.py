from langchain.llms import AzureOpenAI
from openai_config import openai_settings
from langchain.schema import SystemMessage, HumanMessage, AIMessage

def _init_azure_openai() -> AzureOpenAI:
    return AzureOpenAI(openai_api_type = openai_settings.api_type, 
                 openai_api_base = openai_settings.endpoint,
                 openai_api_version = openai_settings.api_version,
                 openai_api_key = openai_settings.api_key,
                 model="gpt-35-turbo",
                 max_tokens=1000,
                 temperature=openai_settings.temperature,
                 deployment_name=openai_settings.chat_engine)

def chat(context:str = "", query:str = "") -> str:
    llm = _init_azure_openai()
    print(llm)
    return llm.predict_messages ([SystemMessage(content=context), HumanMessage(content=query)])
    

if __name__ == "__main__":
    print('lang chain ')
    print(chat("Peter parker loves Mary jane \n Peter works for the morning herald as a photographer\n they married in 2010 \n they have a son knowles","what do you know about Peter parker"))