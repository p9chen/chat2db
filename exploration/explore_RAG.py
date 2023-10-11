"""
Similarity and RAG exploration

Reference: https://blog.gopenai.com/rag-for-everyone-a-beginners-guide-to-embedding-similarity-search-and-vector-db-423946475c90

"""

from langchain.embeddings import HuggingFaceEmbeddings

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

import pandas as pd
import numpy as np
from numpy.linalg import norm

import time
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline


# %% Embedding Similarity

def get_embedding(text, model="sentence-transformers/all-MiniLM-L6-v2", device="cuda"):
    embeddings = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device},
    )

    text = text.replace("\n", " ")

    return embeddings.embed_query(text)

# get the embedding of context
contexts = ["I have a dog. My dog's name is Jimmy",
            "I have a cat. My cat's name is biscuit",
            "My dog who likes to listen to music",
            "My cat likes to play with cricket balls "]

df_context = pd.DataFrame({'context': contexts})
df_context['embedding'] = df_context.context.apply(lambda x: get_embedding(x))
print(df_context)

# get the embedding of question
question = "What is my cat's name?"
question_embedding = get_embedding(text=question)
df_embedding = pd.DataFrame({'embed': question_embedding})
print(df_embedding)

# calculate similarity between context and question
def similarity(x, y):
    return np.dot(x, y) / (norm(x) * norm(y))

cos_sim = []
for index, row in df_context.iterrows():
   x = row.embedding
   y = question_embedding
   # calculate the cosine similiarity
   cosine = similarity(x, y)

   cos_sim.append(cosine)

df_context["cos_sim"] = cos_sim


# %% LangChain RAG

# connect to vector store
PERSIST_DIR = "vector_db/db_chroma"
embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"},
    )

vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

# Retrieve using top-k similarity search
question = "What are the approaches to LLM quantization"
docs = vectorstore.similarity_search_with_score(question, k=3)
print(docs)


docs = vectorstore.max_marginal_relevance_search(question, k=3)
print(docs)



