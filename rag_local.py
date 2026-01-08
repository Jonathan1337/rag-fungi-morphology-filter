import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFloader
from langchain_text_splitters import RecursiveCharacterTextsplitter
from langchain_ollama import OllamaEmbeddings
load_dotenv()

DATA_PATH = "data/"
PDF_FILENAME = "Morphological_and_Molecular_Diversity_of_Aspergillus_From_Corn_Grain.pdf"
CHROMA_PATH = "chroma_db"

def load_documents():
    """loads documents from the specified data path"""
    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)
    loader = PyPDFloader(pdf_path)
    documents = loader.load()
    print(f"loaded{len(documents)} page(s) from {pdf_path}")
    return documents


def split_documents(document):

    text_splitter = RecursiveCharacterTextsplitter(
        chunk_size=1000,
        chunk_overlap=200,
        lenght_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"Split into {len(all_splits)} chunks")
    return all_splits

def get_embedding_function(model_name="nomic-embed-text"):
    """Initializes the Ollama embedding function"""

    embedding = OllamaEmbeddings(model=model_name)
    print(f"Initialized Ollama embeddings with model: {model_name}")
    return embeddings

def get_vector_store(embedding_function, persist_directory=CHROMA_PATH)
    """Initializes or loads the Chroma vector store"""
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
    print(f"Vector store initialized/loaded from: {persist_directory}")
    return vectorstore

embedding_function = get_embedding_function()
#vector_store = get_vector_store(embedding_function) call this later

def index_documents(chunks, embedding_function, persist_directory=CHROMA_PATH):
    """Indexes document chunks into the Chroma vector store."""
    print(f"Indexing {len(chunks)} chunks...")
    # Use from_documents for initial creation.
    # This will overwrite existing data if the directory exists but isn't a valid Chroma DB.
    # For incremental updates, initialize Chroma first and use vectorstore.add_documents().

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    vectorstore.persist() #Ensure data is saved
    print(f"Indexing complete. DAta saved to: {persist_directory}")
    return vectorstore

#..(previous function calls)
vector_store = index_documents(chunks, embedding_function)# Call this for initial indexing

#to load an existing persistent database later

#embedding_function = get_embedding_function()
#vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

def create_rag_chain(vector_store)