"""
Similarity and RAG exploration for the following vector DB
 - Chroma
 - Pgvector

Note: You need to run ingest for Chroma or Pgvector for this demo to work

Reference: https://blog.gopenai.com/rag-for-everyone-a-beginners-guide-to-embedding-similarity-search-and-vector-db-423946475c90

"""

import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from numpy.linalg import norm

from langchain.vectorstores import Chroma

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector, DistanceStrategy

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


# %% RAG - Chroma

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


# %% RAG - Pgvector

# load env variable
if load_dotenv("ingest/pgvector/.env"):
    print("Successfully loaded .env")
else:
    print("Failed to load .env")

# setup config
DATA_DIR = "data"  # dir or file
COLLECTION_NAME = 'langchain_pgvector'

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"},
)

connection_string = PGVector.connection_string_from_db_params(
    driver=os.environ.get("DB_DRIVER"),
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT'),
    database=os.getenv('DB_DATABASE'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)

# connect to pgvector db
store = PGVector(
    connection_string=connection_string,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    distance_strategy=DistanceStrategy.COSINE
)

# retrieve context
retriever = store.as_retriever(search_kwargs={"k": 2})

docs = retriever.get_relevant_documents(query='What is GPTQ')
print(docs)

