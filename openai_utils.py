import os
import json
import openai
from openai_config import configure_openai, openai_settings

configure_openai(openai=openai)

def chat(context:str, query:str):    
  #print("context:%s \n\nquery:%s \n" % (context, query))
  response = openai.ChatCompletion.create(
    engine =openai_settings.chat_engine,
    messages = [{"role":"system","content":context},{"role":"user","content":query}],
    temperature=0.2,
    max_tokens=800,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None)
  return response["choices"][0]["message"]["content"]
  # print("Type is %s" % response.openai_id)

def get_embeddings(text:str) -> str:
  response = openai.Embedding.create(
  engine=openai_settings.embedding_engine,
  input = text,
  temperature=0.2,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)
  # print("%s" % response["data"][0]["embedding"])
  # print(response.keys())
  return response["data"][0]["embedding"]


if __name__ == "__main__":
  # print("hello world")
  # chat()
  print(get_embeddings("world is a funny place"))
  print(chat("reply on with wikipedia links in json format", "who is mike tyson"))