import torch
# pip install git+https://github.com/huggingface/transformers@cae78c46d
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


# %% Config TODO: move yaml config

INSTRUCTION = "Given the context that has been provided. \n {context}, Answer the following question - \n{question}"

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

PERSIST_DIR = "vector_db/db_chroma"

bnb_config = BitsAndBytesConfig(load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False)

model_id = "/home/will/Documents/models/Llama-2-7b-chat-hf"  # "meta-llama/Llama-2-7b-chat-hf"

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


# %% Setup DB retriever





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


# connect to existing vector DB
embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"},
    )

db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

# setup retriever
retriever = db.as_retriever(search_type="mmr",
                            search_kwargs={'k': 5,
                                           'fetch_k': 50,
                                           'lambda_mult': 0.5
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

# %%
import gradio as gr
import random
import time

def clear_llm_memory():
    bot.memory.clear()

def update_prompt(sys_prompt):
    template = get_prompt(instruction, sys_prompt)

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    bot.combine_docs_chain.llm_chain.prompt = prompt


# %% POC

with gr.Blocks() as demo:
    update_sys_prompt = gr.Textbox(label = "Update System Prompt")
    chatbot = gr.Chatbot(label="Chat2DB Bot", height = 300)
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