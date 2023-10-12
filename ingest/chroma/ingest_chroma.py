import os
from pathlib import Path

from chromadb.config import Settings
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredMarkdownLoader

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# path for persist vector DB
DB_CHROMA_PATH = "vector_db/db_chroma"
# path for input data
DATA_DIR = "data"  # dir or file



# Create vector database
def create_vector_db_chroma(data_path, db_path, chunk_size=1000, chunk_overlap=100, length_function=len):
    """
    Creates a vector database using document loaders and embeddings.

    This function loads data from PDF, markdown and text files in the 'data/' directory,
    splits the loaded documents into chunks, transforms them into embeddings using HuggingFace,
    and finally persists the embeddings into a Chroma vector database.

    """
    # Initialize loaders for different file types
    text_loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    markdown_loader = DirectoryLoader(data_path, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)

    # Use proper way to load the files
    if os.path.isdir(Path(data_path)):
        all_loaders = [text_loader, pdf_loader, markdown_loader]
        # Load documents from all loaders
        loaded_documents = []
        for loader in all_loaders:
            loaded_documents.extend(loader.load())
    elif os.path.isfile(Path(data_path)):
        extension = Path(data_path).suffix.lower()  # get file extension and convert to lowercase
        if extension == '.txt':
            loader = TextLoader(file_path=data_path)
        elif extension == '.pdf':
            loader = PyPDFLoader(file_path=data_path)
        elif extension == '.md':
            loader = UnstructuredMarkdownLoader(file_path=data_path)
        else:
            raise ValueError(f'Unsupported file extension: {extension}')
        loaded_documents = loader.load()
    else:
        raise ValueError(f'Invalid path: {data_path} is neither a file nor a directory')

    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   length_function=length_function,
                                                   )

    chunked_documents = text_splitter.split_documents(loaded_documents)

    # Initialize HuggingFace embeddings
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"},
    )

    # Create and persist a Chroma vector database from the chunked documents
    vector_database = Chroma.from_documents(
        documents=chunked_documents,
        embedding=huggingface_embeddings,
        persist_directory=db_path,
    )

    vector_database.persist()


if __name__ == "__main__":
    create_vector_db_chroma(DATA_DIR, DB_CHROMA_PATH)