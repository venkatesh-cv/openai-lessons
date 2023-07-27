from simpleaichat import AIChat
from openai_config import settings
from pg_vector_service import get_context, prime_database
from pydantic import BaseModel, Field
from simpleaichat.utils import wikipedia_search_lookup
from serpapi import GoogleSearch
from googlesearch import search

def init_azure_openai(response_instructions:str = "you are an AI assistant. You will provide helpful answers"):
    ai = AIChat(
        api_key=settings().api_key, 
        api_version=settings().api_version, 
        model=settings().chat_engine,
        api_type=settings().api_type,
        api_endpoint=settings().completions_endpoint,
        system=response_instructions)
    return ai


def q_and_a(response_instructions:str = "you are an AI assistant. You will provide helpful answers", queries:list[str] = []) -> None:
    ai = init_azure_openai(response_instructions)
    for query in queries: 
        response = ai(query)
        print(response)


def q_and_a_stream(response_instructions:str = "you are an AI assistant. You will provide helpful answers", queries:list[str] = []) -> None:
    ai = init_azure_openai(response_instructions)
    for query in queries: 
        for chunk in ai.stream(prompt=query, params={"max_tokens": 100}):
            response_td = chunk["response"]  # dict contains "delta" for the new token and "response"
            print(response_td)

def q_and_a_with_context(response_instructions:str = "you are an AI assistant. You will provide helpful answers", queries:list[str] = [], tags:list[str] = []) -> None:
    ai = init_azure_openai(response_instructions)
    for query in queries: 
        context = get_context(query, tags)
        response = ai(system= response_instructions+","+context, prompt= query)
        print(response)


def q_and_a_with_output_schema(queries:list[str]=[], output_schema:any = None) -> None:
    ai = init_azure_openai()
    for query in queries:
        response = ai(prompt=query, output_schema=output_schema)
        print(response)


class get_event_metadata(BaseModel):
    """Event information"""
    description: str = Field(description="Description of event")
    city: str = Field(description="City where event occured")
    year: int = Field(description="Year when event occured")
    month: str = Field(description="Month when event occured")

def google_search_serpapi(query:str):
    """search Google using API"""
    #print("----------Searching Google for: ", query)
    results = GoogleSearch({"q": query, "api_key":"83b2adb6d395aa2047437da4328db82923a9301f01f6464f0cd1655cbbe24985"}).get_dict()
    # print(results["search_information"])
    # print("------X----")
    # print(results["organic_results"][0]['snippet'])
    # print("----------")
    return results["organic_results"][0]['snippet']

def google_search(query:str):
    """search the internet using google"""
    results = search(query, advanced=True)
    return ",".join(list(map(lambda result: result.description, results)))

def wiki_lookup(query:str):
    """Lookup a topic in wikipedia"""
    #print("----------Searching Wikipedia for: ", query)
    page = wikipedia_search_lookup(query, sentences=3)
    #print("------------------")
    return page

def q_and_a_with_tools(response_instructions:str = "You are an AI assistant. You will answer queries in a helpful manner", queries:list[str]=[]) -> None:
    ai = init_azure_openai()
    for query in queries:
        response = ai(system= response_instructions, prompt=query, tools=[google_search, wiki_lookup], save_messages=True)
        print(query)
        print(response['response'])
        print("tool used %s" % response['tool'])


# write a main function that calls demo
if __name__ == "__main__":
    # q_and_a(response_instructions= "You  are an AI assistant. You will answer queries in one sentence", 
    #          queries=["what is 100 crores in u.s.dollars","who is kamala Harris", "where was she born", "when was she born", "What are her educational qualifications"])
    # q_and_a_stream(response_instructions= "You  are an AI assistant. You will help write emails", 
    #         queries=["write a 100 word email to all employees on the importance of respectfulness to colleagues"])
    # prime_database()
    # q_and_a_with_context(response_instructions= "you are an AI assistant. You will answer queries in one sentence.", queries=["where does peter knowles work","does he have children"], tags=["marvel"])
    # q_and_a(response_instructions="{'country':'India', 'headOfState':'Narendra Modi', 'population': '1.3 billion'}", queries=["norway", "iceland"])
    # q_and_a_with_output_schema(output_schema=get_event_metadata, queries=["when the Berlin wall fell", "when did the USSR dissolve","when did communist government fall in Poland"])
    q_and_a_with_tools(response_instructions= "You  are an AI assistant. You will answer queries in one sentence", queries=["what is the maximum temperature in Chennai, India today","who is kamala Harris", "where was she, Kamala Harris, born", "when was she born", "What are her educational qualifications"])