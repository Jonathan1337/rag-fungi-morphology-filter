"""
rag_local.py
------------
Pipeline RAG (Retrieval-Augmented Generation) para análise de artigos
científicos sobre morfologia fúngica.

Suporta dois modos de operação:
  - Texto: Processa apenas o texto extraído do PDF (padrão).
  - Multimodal: Extrai imagens do PDF, gera descrições textuais via modelo
    de visão (LLaVA), e indexa texto + descrições de imagem no FAISS.

Uso:
  python rag_local.py                  # Modo texto (padrão)
  python rag_local.py --multimodal     # Modo multimodal
"""

import os
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- Constantes Globais ---
DATA_PATH = "data/"
PDF_FILENAME = "Morphological_and_Molecular_Diversity_of_Aspergillus_From_Corn_Grain.pdf"
FAISS_PATH = "faiss_index"
EXTRACTED_IMAGES_PATH = "data/extracted_images/"


# =====================================================================
#  FUNÇÕES DO PIPELINE DE TEXTO
# =====================================================================

def load_documents():
    """Carrega documentos do arquivo PDF especificado."""
    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} page(s) from {pdf_path}")
    return documents


def split_documents(documents):
    """Divide documentos em chunks otimizados para busca semântica."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"Split into {len(all_splits)} chunks")
    return all_splits


# =====================================================================
#  FUNÇÕES DO PIPELINE MULTIMODAL
# =====================================================================

def load_multimodal_documents(vision_model="llava"):
    """
    Carrega texto do PDF e extrai + textualiza imagens para criar uma
    base de conhecimento multimodal.

    Pipeline melhorado (v2):
        1. Extração de texto + split em chunks
        2. Extração de imagens (com classificação de tipo)
        3. OCR pre-pass para extrair rótulos das imagens
        4. Textualização contextualizada via modelo de visão
        5. Validação das descrições contra o texto do documento

    Args:
        vision_model (str): Nome do modelo de visão no Ollama.

    Returns:
        list[Document]: Lista combinada de chunks de texto + descrições de imagem.
    """
    from image_extractor import extract_images_from_pdf
    from image_textualizer import textualize_all_images
    from ocr_extractor import extract_text_batch
    from description_validator import extract_species_from_text, validate_all_descriptions

    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)

    # 1. Carregar e dividir texto normalmente
    print("\n--- Phase 1: Text Extraction ---")
    text_docs = load_documents()
    text_chunks = split_documents(text_docs)

    # Extrair contexto do documento (espécies mencionadas no texto)
    full_text = " ".join([doc.page_content for doc in text_docs])
    species_list = extract_species_from_text(full_text)
    print(f"  Species found in document: {', '.join(species_list)}")

    # 2. Extrair imagens do PDF (com classificação de tipo)
    print("\n--- Phase 2: Image Extraction + Classification ---")
    extracted_images = extract_images_from_pdf(pdf_path, EXTRACTED_IMAGES_PATH)

    if not extracted_images:
        print("No images found in PDF. Proceeding with text-only mode.")
        return text_chunks

    # 3. OCR pre-pass — extrair texto das imagens
    print("\n--- Phase 3: OCR Pre-Pass ---")
    ocr_results = extract_text_batch(extracted_images)

    # Anexar resultados OCR às imagens
    for img, ocr in zip(extracted_images, ocr_results):
        img["ocr_text"] = ocr

    # 4. Gerar descrições contextualizadas via modelo de visão
    print("\n--- Phase 4: Context-Aware Textualization ---")
    doc_context = {
        "title": PDF_FILENAME.replace("_", " ").replace(".pdf", ""),
        "species": species_list,
    }
    image_docs = textualize_all_images(
        extracted_images,
        model_name=vision_model,
        document_context=doc_context,
    )

    # 5. Validar descrições contra o documento
    print("\n--- Phase 5: Description Validation ---")
    validated_docs = validate_all_descriptions(image_docs, species_list)

    # Combinar chunks de texto + documentos de imagem validados
    combined_docs = text_chunks + validated_docs
    print(f"\n--- Combined: {len(text_chunks)} text chunks + "
          f"{len(validated_docs)} image descriptions = "
          f"{len(combined_docs)} total documents ---\n")

    return combined_docs


# =====================================================================
#  FUNÇÕES DE EMBEDDING E INDEXAÇÃO
# =====================================================================

def get_embedding_function(model_name="nomic-embed-text"):
    """Inicializa a função de embedding via Ollama."""
    embedding = OllamaEmbeddings(model=model_name)
    print(f"Initialized Ollama embeddings with model: {model_name}")
    return embedding


def get_vector_store(embedding_function, persist_directory=FAISS_PATH):
    """Carrega o banco vetorial FAISS do disco."""
    vectorstore = FAISS.load_local(
        persist_directory,
        embedding_function,
        allow_dangerous_deserialization=True
    )
    print(f"Vector store loaded from: {persist_directory}")
    return vectorstore


def index_documents(chunks, embedding_function, persist_directory=FAISS_PATH):
    """Indexa chunks de documentos no FAISS."""
    print(f"Indexing {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_function,
    )
    vectorstore.save_local(persist_directory)
    print(f"Indexing complete. Data saved to: {persist_directory}")
    return vectorstore


# =====================================================================
#  FUNÇÕES DA CADEIA RAG
# =====================================================================

def create_rag_chain(vector_store, llm_model_name="qwen3:8b", context_window=8192):
    """Constrói a cadeia RAG completa usando LCEL."""
    # Inicializar o LLM
    llm = ChatOllama(
        model=llm_model_name,
        temperature=0,
        num_ctx=context_window
    )
    print(f"Initialized ChatOllama with model: {llm_model_name}, "
          f"context window: {context_window}")

    # Criar o retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 3}
    )
    print("Retriever initialized.")

    # Template do prompt — inclui menção a conteúdo multimodal
    template = """Answer the question based ONLY on the following context.
The context may include both text excerpts and textual descriptions of images
extracted from the document. Use all available information to provide a
comprehensive answer.

Context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    print("Prompt template created.")

    # Cadeia RAG via LCEL
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain created.")
    return rag_chain


def query_rag(chain, question):
    """Envia uma pergunta para a cadeia RAG e exibe a resposta."""
    print(f"\n{'─'*60}")
    print(f"Question: {question}")
    print(f"{'─'*60}")
    response = chain.invoke(question)
    print(f"\nResponse:\n{response}")
    print(f"{'─'*60}\n")


# =====================================================================
#  EXECUÇÃO PRINCIPAL
# =====================================================================

def parse_args():
    """Configura e parseia os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="RAG Fungi Morphology Filter — Pipeline de busca semântica"
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Ativa o modo multimodal: extrai e textualiza imagens do PDF "
             "usando modelo de visão (LLaVA)."
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        default="llava",
        help="Modelo de visão a usar no modo multimodal (default: llava)."
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="qwen3:8b",
        help="Modelo LLM para geração de respostas (default: qwen3:8b)."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Pergunta customizada para enviar ao RAG."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    mode_label = "MULTIMODAL" if args.multimodal else "TEXT-ONLY"
    print(f"\n{'='*60}")
    print(f"  RAG Fungi Morphology Filter — Mode: {mode_label}")
    print(f"{'='*60}\n")

    # 1. Carregar e processar documentos
    if args.multimodal:
        chunks = load_multimodal_documents(vision_model=args.vision_model)
    else:
        docs = load_documents()
        chunks = split_documents(docs)

    # 2. Inicializar embeddings
    embedding_function = get_embedding_function()

    # 3. Indexar documentos no FAISS
    print("\nAttempting to index documents...")
    vector_store = index_documents(chunks, embedding_function)

    # 4. Criar cadeia RAG
    rag_chain = create_rag_chain(vector_store, llm_model_name=args.llm_model)

    # 5. Consultas
    if args.query:
        query_rag(rag_chain, args.query)
    else:
        # Perguntas de exemplo padrão
        query_rag(rag_chain, "What is the main topic of the document?")
        query_rag(rag_chain, "Summarize the introduction section.")

        if args.multimodal:
            # Pergunta que pode se beneficiar do contexto visual
            query_rag(
                rag_chain,
                "Describe the morphological features of Aspergillus species "
                "shown in the images of the document."
            )