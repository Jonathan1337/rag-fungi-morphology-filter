# 🍄 RAG Fungi Morphology Filter

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama">
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00?style=for-the-badge" alt="ChromaDB">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

Sistema **RAG (Retrieval-Augmented Generation)** para filtragem e análise de dados micológicos em artigos acadêmicos. Desenvolvido com **Ollama**, **LangChain** e **ChromaDB** para busca semântica em morfologia fúngica.

O sistema permite realizar perguntas em linguagem natural sobre o conteúdo de artigos científicos em PDF, retornando respostas contextualizadas e fundamentadas exclusivamente no conteúdo do documento.

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Arquitetura](#-arquitetura)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação](#-instalação)
- [Uso](#-uso)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Configuração](#-configuração)
- [Tecnologias](#-tecnologias)
- [Licença](#-licença)

---

## 🔍 Visão Geral

Este projeto implementa um pipeline RAG completo que:

1. **Carrega** documentos PDF de artigos científicos sobre morfologia fúngica.
2. **Divide** o conteúdo em chunks (fragmentos) otimizados para busca semântica.
3. **Gera embeddings** vetoriais utilizando o modelo `nomic-embed-text` via Ollama.
4. **Armazena** os vetores em um banco de dados ChromaDB persistente.
5. **Recupera** os trechos mais relevantes a partir de uma pergunta do usuário.
6. **Gera respostas** contextualizadas usando o LLM `qwen3:8b` via Ollama.

O artigo utilizado como base de conhecimento é:
> *"Morphological and Molecular Diversity of Aspergillus From Corn Grain"*

---

## 🏗 Arquitetura

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  PDF Input   │───▶│  Document Loader  │───▶│  Text Splitter   │
│  (data/)     │    │  (PyPDFLoader)    │    │  (Recursive)     │
└─────────────┘    └──────────────────┘    └────────┬─────────┘
                                                     │
                                                     ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  LLM Response    │◀───│  RAG Chain       │◀───│  ChromaDB        │
│  (StrOutput)     │    │  (LCEL Pipeline) │    │  (Vector Store)  │
└──────────────────┘    └────────┬─────────┘    └────────┬─────────┘
                                 │                       │
                                 ▼                       ▼
                        ┌──────────────────┐    ┌──────────────────┐
                        │  ChatOllama      │    │  OllamaEmbeddings│
                        │  (qwen3:8b)      │    │  (nomic-embed)   │
                        └──────────────────┘    └──────────────────┘
```

---

## ⚙ Pré-requisitos

Antes de iniciar, certifique-se de ter instalados:

| Requisito | Versão Mínima | Descrição |
|-----------|:------------:|-----------|
| [Python](https://www.python.org/) | 3.10+ | Linguagem principal do projeto |
| [Ollama](https://ollama.com/) | Última | Runtime local para modelos LLM |
| [Git](https://git-scm.com/) | — | Para clonar o repositório |

### Modelos Ollama necessários

Após instalar o Ollama, baixe os modelos executando:

```bash
# Modelo de embeddings (representação vetorial do texto)
ollama pull nomic-embed-text

# Modelo LLM para geração de respostas
ollama pull qwen3:8b
```

> **💡 Nota:** O modelo `qwen3:8b` requer aproximadamente **5 GB** de espaço em disco e pelo menos **8 GB de RAM** para execução confortável.

---

## 🚀 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/Jonathan1337/rag-fungi-morphology-filter.git
cd rag-fungi-morphology-filter
```

### 2. Crie e ative um ambiente virtual

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar (Windows)
venv\Scripts\activate

# Ativar (Linux/macOS)
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install langchain langchain-community langchain-ollama langchain-text-splitters chromadb pypdf python-dotenv
```

### 4. Configure as variáveis de ambiente (opcional)

Crie um arquivo `.env` na raiz do projeto se desejar customizar configurações:

```env
# Exemplo de variáveis (o projeto funciona sem este arquivo)
OLLAMA_HOST=http://localhost:11434
```

### 5. Verifique se o Ollama está em execução

```bash
ollama list
```

---

## 💻 Uso

### Execução padrão

```bash
python rag_local.py
```

Na primeira execução, o script irá:
1. Carregar o PDF da pasta `data/`.
2. Dividir o texto em chunks de 1000 caracteres.
3. Criar embeddings e indexar no ChromaDB.
4. Realizar duas perguntas de exemplo sobre o documento.

### Exemplo de saída

```
loaded 15 page(s) from data/Morphological_and_Molecular_Diversity_of_Aspergillus_From_Corn_Grain.pdf
Split into 42 chunks
Initialized Ollama embeddings with model: nomic-embed-text
Indexing 42 chunks...
Indexing complete. Data saved to: chroma_db

Querying RAG chain...
Question: What is the main topic of the document?
Response: The document discusses the morphological and molecular diversity of Aspergillus species isolated from corn grain...
```

### Personalizando perguntas

Edite a seção `__main__` em `rag_local.py` para customizar suas consultas:

```python
query_question = "Quais espécies de Aspergillus foram identificadas?"
query_rag(rag_chain, query_question)
```

---

## 📁 Estrutura do Projeto

```
rag-fungi-morphology-filter/
│
├── 📄 rag_local.py          # Script principal do pipeline RAG
├── 📂 data/                  # Diretório dos documentos PDF
│   └── 📑 Morphological_and_Molecular_Diversity_of_Aspergillus_From_Corn_Grain.pdf
├── 📂 chroma_db/             # Banco vetorial ChromaDB (gerado na execução)
├── 📄 .env                   # Variáveis de ambiente (opcional)
├── 📄 .gitignore             # Arquivos ignorados pelo Git
├── 📄 LICENSE                # Licença MIT
├── 📄 README.md              # Este arquivo
└── 📄 SCRIPT_DETAILS.md      # Documentação detalhada do script
```

---

## 🔧 Configuração

Os parâmetros principais podem ser ajustados diretamente no script:

| Parâmetro | Valor Padrão | Descrição |
|-----------|:------------:|-----------|
| `DATA_PATH` | `data/` | Diretório contendo os PDFs |
| `PDF_FILENAME` | `Morphological_and_...pdf` | Nome do arquivo PDF de entrada |
| `CHROMA_PATH` | `chroma_db` | Diretório do banco vetorial persistente |
| `chunk_size` | `1000` | Tamanho máximo de cada chunk (em caracteres) |
| `chunk_overlap` | `200` | Sobreposição entre chunks consecutivos |
| `model_name` (embedding) | `nomic-embed-text` | Modelo Ollama para embeddings |
| `llm_model_name` | `qwen3:8b` | Modelo Ollama para geração de respostas |
| `temperature` | `0` | Criatividade do LLM (0 = mais factual) |
| `context_window` | `8192` | Tamanho da janela de contexto do LLM |
| `search_kwargs['k']` | `3` | Número de chunks recuperados por consulta |

---

## 🛠 Tecnologias

| Tecnologia | Função |
|-----------|--------|
| **[LangChain](https://www.langchain.com/)** | Framework de orquestração para pipelines LLM |
| **[Ollama](https://ollama.com/)** | Runtime local para execução de modelos de IA |
| **[ChromaDB](https://www.trychroma.com/)** | Banco de dados vetorial para busca semântica |
| **[PyPDF](https://pypdf.readthedocs.io/)** | Carregamento e parsing de arquivos PDF |
| **[Qwen3 8B](https://ollama.com/library/qwen3)** | Modelo LLM para geração de respostas |
| **[Nomic Embed Text](https://ollama.com/library/nomic-embed-text)** | Modelo de embeddings de texto |

---

## 📄 Licença

Este projeto está licenciado sob a **Licença MIT** — veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

<p align="center">
  Desenvolvido por <a href="https://github.com/Jonathan1337">Jonathan1337</a> 🍄
</p>
