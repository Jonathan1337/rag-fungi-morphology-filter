# 🍄 RAG Fungi Morphology Filter

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama">
  <img src="https://img.shields.io/badge/FAISS-Vector_Store-3B5998?style=for-the-badge" alt="FAISS">
  <img src="https://img.shields.io/badge/EasyOCR-Text_Extraction-FF6F00?style=for-the-badge" alt="EasyOCR">
  <img src="https://img.shields.io/badge/Version-1.2.1-blue?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

Sistema **RAG (Retrieval-Augmented Generation)** para filtragem e análise de dados micológicos em artigos acadêmicos. Desenvolvido com **Ollama**, **LangChain** e **FAISS** para busca semântica em morfologia fúngica.

O sistema permite realizar perguntas em linguagem natural sobre o conteúdo de artigos científicos em PDF, retornando respostas contextualizadas e fundamentadas exclusivamente no conteúdo do documento.

> 🆕 **Modo Multimodal v2**: Pipeline multimodal com **5 fases** — extração de imagens, **classificação automática**, **OCR pre-pass** (EasyOCR), **textualização contextualizada** (LLaVA) e **validação anti-alucinação** contra o texto do documento.

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
2. **Extrai imagens** embutidas no PDF com **classificação automática** de tipo (placas, microscopia, diagrama, etc.).
3. **Pré-processa com OCR** (EasyOCR) para extrair rótulos, espécies e textos das imagens.
4. **Gera descrições contextualizadas** via LLaVA com prompts específicos por tipo de imagem.
5. **Valida descrições** contra o texto do documento, detectando alucinações e espécies fabricadas.
6. **Divide** o conteúdo em chunks otimizados para busca semântica.
7. **Gera embeddings** vetoriais utilizando o modelo `nomic-embed-text` via Ollama.
8. **Armazena** os vetores em um índice FAISS local.
9. **Recupera** os trechos mais relevantes a partir de uma pergunta do usuário.
10. **Gera respostas** contextualizadas usando o LLM `qwen3:8b` via Ollama.

O artigo utilizado como base de conhecimento é:
> *"Morphological and Molecular Diversity of Aspergillus From Corn Grain"*

---

## 🏗 Arquitetura

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  PDF Input   │───▶│  Document Loader  │───▶│  Text Splitter   │
│  (data/)     │    │  (PyPDFLoader)    │    │  (Recursive)     │
└──────┬──────┘    └──────────────────┘    └────────┬─────────┘
       │                                            │
       │  ┌──────────────────┐                      │
       └─▶│  Image Extractor │                      │
          │  + Classifier    │                      │
          └────────┬─────────┘                      │
                   ▼                                │
          ┌──────────────────┐                      │
          │  OCR Pre-Pass    │                      │
          │  (EasyOCR)       │                      │
          └────────┬─────────┘                      │
                   ▼                                │
          ┌──────────────────┐                      │
          │ Image Textualizer│                      │
          │ (LLaVA + Context)│                      │
          └────────┬─────────┘                      │
                   ▼                                │
          ┌──────────────────┐                      │
          │  Desc. Validator │                      │
          │  (Anti-Halluc.)  │                      │
          └────────┬─────────┘                      │
                   │                                │
                   ▼                                ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  LLM Response    │◀───│  RAG Chain       │◀───│  FAISS Index     │
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


# Criar ambiente virtual
python -m venv venv

# Ativar (Windows)
venv\Scripts\activate

# Ativar (Linux/macOS)
source venv/bin/activate
```

### 3. Instale as dependências

```bash
# Dependências core
pip install langchain langchain-community langchain-ollama langchain-text-splitters faiss-cpu pypdf python-dotenv PyMuPDF Pillow requests

# Dependências para modo multimodal v2 (OCR + validação)
pip install easyocr opencv-python-headless
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

### Modo Texto (padrão)

```bash
python rag_local.py
```

Na primeira execução, o script irá:
1. Carregar o PDF da pasta `data/`.
2. Dividir o texto em chunks de 1000 caracteres.
3. Criar embeddings e indexar no FAISS.
4. Realizar duas perguntas de exemplo sobre o documento.

### 🆕 Modo Multimodal (Pipeline v2)

```bash
python rag_local.py --multimodal
```

Executa o pipeline multimodal completo em **5 fases**:

| Fase | Descrição | Módulo |
|:----:|-----------|--------|
| 1 | **Extração de texto** — carrega e divide o PDF em chunks | `rag_local.py` |
| 2 | **Extração + Classificação de imagens** — extrai imagens e classifica tipo | `image_extractor.py` |
| 3 | **OCR Pre-Pass** — extrai rótulos, espécies e textos das imagens | `ocr_extractor.py` |
| 4 | **Textualização Contextualizada** — gera descrições com prompts por tipo | `image_textualizer.py` |
| 5 | **Validação Anti-Alucinação** — verifica espécies contra o documento | `description_validator.py` |

#### Tipos de imagem classificados

| Tipo | Heurística | Prompt usado |
|------|-----------|-------------|
| `icon` | < 200×200px | Breve, sem contexto científico |
| `plate_grid_row` | ratio > 2.5, largura > 800 | Placas de Petri macroscópicas |
| `plate_grid_matrix` | ratio < 0.6, alta, largura > 1000 | Matriz de culturas |
| `microscopy_panel` | ratio < 0.8, > 800px | Estruturas microscópicas |
| `diagram` | > 800×800px, ~quadrado | Diagramas científicos |
| `single_image` | fallback | Prompt genérico com guardrails |

### Opções de CLI

| Flag | Descrição | Default |
|------|-----------|:-------:|
| `--multimodal` | Ativa o modo multimodal | `False` |
| `--vision-model` | Modelo de visão para textualização | `llava` |
| `--llm-model` | Modelo LLM para respostas | `qwen3:8b` |
| `--query` | Pergunta customizada | - |

### Exemplos de uso

```bash
# Modo texto com pergunta customizada
python rag_local.py --query "Quais espécies de Aspergillus foram identificadas?"

# Modo multimodal com modelo de visão específico
python rag_local.py --multimodal --vision-model llava

# Pergunta sobre conteúdo visual
python rag_local.py --multimodal --query "Describe the morphological features shown in the microscopy images."
```

---

## 📁 Estrutura do Projeto

```
rag-fungi-morphology-filter/
│
├── 📄 rag_local.py              # Script principal — pipeline RAG (5 fases)
├── 📄 image_extractor.py         # Extração + classificação de tipo de imagem
├── 📄 image_textualizer.py       # Textualização contextualizada via LLaVA
├── 📄 ocr_extractor.py           # OCR pre-pass via EasyOCR
├── 📄 description_validator.py   # Validação anti-alucinação de descrições
├── 📂 data/                      # Diretório dos documentos PDF
│   ├── 📑 Morphological_and_Molecular_Diversity_of_Aspergillus_From_Corn_Grain.pdf
│   └── 📂 extracted_images/      # Imagens extraídas (gerado na execução)
├── 📂 faiss_index/               # Índice vetorial FAISS (gerado na execução)
├── 📄 changelog.md               # Histórico de alterações
├── 📄 .env                       # Variáveis de ambiente (opcional)
├── 📄 .gitignore                 # Arquivos ignorados pelo Git
├── 📄 LICENSE                    # Licença MIT
├── 📄 README.md                  # Este arquivo
└── 📄 SCRIPT_DETAILS.md          # Documentação detalhada do script
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
| **[FAISS](https://github.com/facebookresearch/faiss)** | Índice vetorial para busca semântica (Meta AI) |
| **[EasyOCR](https://github.com/JaidedAI/EasyOCR)** | OCR deep learning para extração de texto de imagens |
| **[OpenCV](https://opencv.org/)** | Pré-processamento de imagens para OCR |
| **[PyPDF](https://pypdf.readthedocs.io/)** | Carregamento e parsing de arquivos PDF |
| **[Qwen3 8B](https://ollama.com/library/qwen3)** | Modelo LLM para geração de respostas |
| **[Nomic Embed Text](https://ollama.com/library/nomic-embed-text)** | Modelo de embeddings de texto |
| **[LLaVA](https://ollama.com/library/llava)** | Modelo de visão para textualização multimodal |
| **[PyMuPDF](https://pymupdf.readthedocs.io/)** | Extração de imagens de PDFs |
| **[Pillow](https://pillow.readthedocs.io/)** | Manipulação e processamento de imagens |

---

## 📄 Licença

Este projeto está licenciado sob a **Licença MIT** — veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

<p align="center">
  Desenvolvido por <a href="https://github.com/Jonathan1337">Jonathan1337</a> 🍄
</p>
