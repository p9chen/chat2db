import torch
import json
import textwrap
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# DB Ingestion
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, TextLoader

# Setting up with LangChain
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

import gradio as gr

# %%

# ============================================================ #
#                 Ingest Data into VectorDB                    #
# ============================================================ #

loader = PyPDFLoader("data/2023_GPTQ_Accurate Post-Training Quantization For Generative Pre-Trained Transformers.pdf")
# loader = TextLoader("data/state_of_the_union.txt")

text_splitter = RecursiveCharacterTextSplitter(
    # Set a small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=20,
    length_function=len,
)

pages = loader.load_and_split(text_splitter)

db = Chroma.from_documents(pages, HuggingFaceEmbeddings(), persist_directory='vector_db')


# %%
# ============================================================ #
#                   Set up quantized model                     #
# ============================================================ #

bnb_config = BitsAndBytesConfig(load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False)

model_id = "/home/will/Documents/models/Llama-2-7b-chat-hf"  # "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = bnb_config,device_map={"": 0})

# %%

# ============================================================ #
#                    Set up prompt template                    #
# ============================================================ #

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

# %%

instruction = "Given the context that has been provided. \n {context}, Answer the following question - \n{question}"

system_prompt = """You are an expert in chess.
You will be given a context to answer from. Be precise in your answers wherever possible.
In case you are sure you don't know the answer then you say that based on the context you don't know the answer.
In all other instances you provide an answer to the best of your capability. Cite urls when you can access them related to the context."""

get_prompt(instruction, system_prompt)

template = get_prompt(instruction, system_prompt)
print(template)

prompt = PromptTemplate(template=template, input_variables=["context", "question"])


memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5,
    return_messages=True
)


retriever = db.as_retriever()


def create_pipeline(max_new_tokens=512):
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=1)
    return pipe


class ChatBot:
    def __init__(self, memory, prompt, task:str = "text-generation", retriever = retriever):
        self.memory = memory
        self.prompt = prompt
        self.retriever = retriever

    def create_chat_bot(self, max_new_tokens = 512):
        hf_pipe = create_pipeline(max_new_tokens)
        llm = HuggingFacePipeline(pipeline =hf_pipe)
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt}
        )
        return qa


chat_bot = ChatBot(memory=memory, prompt=prompt)

bot = chat_bot.create_chat_bot()

def clear_llm_memory():
    bot.memory.clear()

def update_prompt(sys_prompt):
    if sys_prompt == "":
        sys_prompt = system_prompt
    template = get_prompt(instruction, sys_prompt)

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    bot.combine_docs_chain.llm_chain.prompt = prompt

# %% POC

with gr.Blocks() as demo:
    update_sys_prompt = gr.Textbox(label = "Update System Prompt")
    chatbot = gr.Chatbot(label="Chess Bot", height = 300)
    msg = gr.Textbox(label="Question")
    clear = gr.ClearButton([msg, chatbot])
    clear_memory = gr.Button(value = "Clear LLM Memory")


    def respond(message, chat_history):
        bot_message = bot({"question": message})['answer']
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear_memory.click(clear_llm_memory)
    update_sys_prompt.submit(update_prompt, inputs=update_sys_prompt)

demo.launch(share=False, debug=True)

