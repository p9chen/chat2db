import os
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector, DistanceStrategy

import gradio as gr

# %% Config TODO: move yaml config

INSTRUCTION = "Given the context that has been provided. \n {context}, Answer the following question - \n{question}"

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """

bnb_config = BitsAndBytesConfig(load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False)

model_id = "/home/will/Documents/models/Llama-2-13b-chat-hf"  # "meta-llama/Llama-2-13b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = bnb_config, device_map={"": 0})


# %%
def get_prompt(instruction=INSTRUCTION, new_system_prompt=DEFAULT_SYSTEM_PROMPT,
               b_inst="[INST]", e_inst="[/INST]", b_sys="<<SYS>>\n", e_sys="\n<</SYS>>\n\n"):
    """
    create prompt template to be feed into LLM based on user prompt and optional user system prompt
    :param instruction: user instruction
    :param new_system_prompt: user defined system prompt
    :param b_inst: token indicates the beginning of prompt
    :param e_inst: token indicates the end of prompt
    :param b_sys: token indicates the beginning of system prompt
    :param e_sys: token indicates the end of system prompt
    :return: full prompt template
    """
    system_prompt = b_sys + new_system_prompt + e_sys
    prompt_template = b_inst + system_prompt + instruction + e_inst
    return prompt_template


# %% Create pipeline

def create_pipeline(max_new_tokens=512):
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=1
    )
    return pipe

# %% Setup Chatbot requirements

# load env variable
if load_dotenv("ingest/pgvector/.env"):
    print("Successfully loaded .env")
else:
    print("Failed to load .env")

# setup config
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

db = PGVector(
    connection_string=connection_string,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    distance_strategy=DistanceStrategy.COSINE
)

# setup retriever
retriever = db.as_retriever(search_type="mmr",
                            search_kwargs={'k': 3,
                                           'score_threshold': 0.25,
                                           'fetch_k': 30,
                                           'lambda_mult': 0.3  # Diversity of results from MMR; 1 for min and 0 for max
                                           }
                            )

# setup prompt
template = get_prompt()
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# setup chat memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5,
    return_messages=True
)

# %% Create Chat bot

class ChatBot:
    def __init__(self, memory, prompt, task: str="text-generation", retriever=retriever):
        self.memory = memory
        self.prompt = prompt
        self.retriever = retriever
        self.bot = None

    def create_chat_bot(self, max_new_tokens=512):
        hf_pipe = create_pipeline(max_new_tokens)
        llm = HuggingFacePipeline(pipeline =hf_pipe)
        self.bot = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt}
        )

    def clear_llm_memory(self):
        if not self.bot:
            self.bot.memory.clear()

    def update_prompt(self, sys_prompt):
        if not self.bot:
            template = get_prompt(new_system_prompt=sys_prompt)

            prompt = PromptTemplate(template=template, input_variables=["context", "question"])

            self.bot.combine_docs_chain.llm_chain.prompt = prompt


# %% POC UI

chat_bot = ChatBot(memory=memory, prompt=prompt)
chat_bot.create_chat_bot()

with gr.Blocks() as demo:
    update_sys_prompt = gr.Textbox(label="Update System Prompt")
    chatbot = gr.Chatbot(label="Chat2DB Bot", height=300)
    msg = gr.Textbox(label="Question")
    clear = gr.ClearButton([msg, chatbot])
    clear_memory = gr.Button(value="Clear LLM Memory")


    def respond(message, chat_history):
        bot_message = chat_bot.bot({"question": message})['answer']
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear_memory.click(chat_bot.clear_llm_memory)
    update_sys_prompt.submit(chat_bot.update_prompt, inputs=update_sys_prompt)

demo.launch(share=False, debug=True)