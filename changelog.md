# Changelog

Todas as alterações notáveis deste projeto serão documentadas neste arquivo.

---

## [1.2.1] — 2026-02-16

### 🧪 Teste Final Documentado

- `test_procedure.md` [NOVO] — Documenta a metodologia de teste em 3 etapas
- `test_final_comparison.py` [NOVO] — Script de comparação automática contra ground truth
- `test_comparison_v2.md` [GERADO] — Relatório final: score médio 57%, ✅ 2/7, ⚠️ 4/7, ❌ 1/7
- **Zero alucinações** de espécies falsas em todas as 7 imagens
- **5/7 tipos de imagem** identificados corretamente
- **Rótulos OCR** detectados com sucesso em todas as imagens com texto

---

## [1.2.0] — 2026-02-16

### ✨ Pipeline v2 — Correção de Qualidade da Textualização

- **Estratégia 1 — OCR Pre-Pass:**
  - `ocr_extractor.py` [NOVO] — Extrai texto/rótulos das imagens via EasyOCR antes da textualização
  - Pré-processamento: grayscale → threshold adaptativo → denoising
  - Filtragem por confiança (≥0.3) e deduplicação de detecções
- **Estratégia 2 — Classificação de Imagem:**
  - `image_extractor.py` — Adicionada função `classify_image()` com heurísticas de dimensão/aspect ratio
  - Tipos: `icon`, `plate_grid_row`, `plate_grid_matrix`, `microscopy_panel`, `diagram`, `single_image`
- **Estratégia 3 — Prompts Contextualizados:**
  - `image_textualizer.py` — Reescrito com prompts específicos por tipo de imagem
  - Guardrails anti-alucinação: "Do NOT invent", "ONLY mention species from OCR/document"
  - Contexto do documento (título, lista de espécies) injetado no prompt
- **Estratégia 4 — Validação Pós-Processamento:**
  - `description_validator.py` [NOVO] — Valida descrições contra espécies do documento
  - Detecta espécies alucinadas, texto fabricado (francês), magnificações inventadas
  - Atribui confidence score (0.0-1.0) e marca espécies não verificadas

### 🔧 Alterado

- `rag_local.py` — Pipeline multimodal agora executa 5 fases: texto → classificação → OCR → textualização contextualizada → validação

### 📦 Novas Dependências

- `easyocr` — OCR deep learning para extração de texto de imagens
- `opencv-python-headless` — Pré-processamento de imagens para OCR

---

## [1.1.1] — 2026-02-16

### 🧪 Teste de Qualidade da Textualização

- `test_manual_descriptions.md` — Descrições manuais (ground truth) das 7 imagens extraídas
- `test_llava_descriptions.md` — Descrições geradas automaticamente pelo LLaVA (v1)
- `test_comparison.md` — Análise comparativa completa com avaliação por imagem
- `test_textualization.py` — Script de teste para gerar as descrições LLaVA
- Resultado: LLaVA acerta tipo geral em 5/7 imagens, mas apresenta alucinações em detalhes (espécies, rótulos, layouts)

---

## [1.1.0] — 2026-02-14

### ✨ Adicionado

- **Textualização Multimodal** — Pipeline de extração e descrição textual de imagens do PDF
  - `image_extractor.py` — Extrai imagens embutidas no PDF usando PyMuPDF, com filtragem automática de ícones/decorações (< 100x100px)
  - `image_textualizer.py` — Gera descrições textuais detalhadas das imagens via modelo de visão LLaVA (Ollama), retornando objetos `Document` do LangChain
  - Flag `--multimodal` no CLI para ativar o modo multimodal
  - Flag `--vision-model` para selecionar o modelo de visão (default: `llava`)
  - Flag `--query` para perguntas customizadas via linha de comando
  - Flag `--llm-model` para selecionar o modelo LLM
- Diretório `data/extracted_images/` para armazenamento das imagens extraídas

### 🔧 Alterado

- `rag_local.py` — Refatoração completa:
  - Removidas chamadas de função no escopo global (linhas 57, 121-122)
  - Adicionado `argparse` para CLI estruturado
  - Template do prompt atualizado para contexto multimodal
  - Código organizado em seções com headers
  - Nova função `load_multimodal_documents()` para orquestrar o fluxo multimodal
  - Vector store migrado de ChromaDB para FAISS (compatibilidade Python 3.14)
- `.gitignore` — Adicionado `data/extracted_images/` e `faiss_index/`
- `README.md` — Atualizado com informações do modo multimodal
- `SCRIPT_DETAILS.md` — Documentação dos novos módulos

### 📦 Novas Dependências

- `PyMuPDF` (`fitz`) — Extração de imagens de PDFs
- `Pillow` — Manipulação de imagens
- `requests` — Comunicação com a API do Ollama (modelo de visão)
- `faiss-cpu` — Banco de dados vetorial FAISS (substitui ChromaDB)

### 🤖 Modelo Ollama Adicional

- `llava` — Modelo de visão para textualização de imagens (~4.7 GB)

---

## [1.0.0] — Versão Inicial

### ✨ Adicionado

- Pipeline RAG completo com LangChain, Ollama e ChromaDB
- Carregamento e parsing de PDFs via `PyPDFLoader`
- Divisão de texto em chunks com `RecursiveCharacterTextSplitter`
- Embeddings via `nomic-embed-text` (Ollama)
- Indexação e busca vetorial com ChromaDB
- Cadeia RAG com `ChatOllama` (qwen3:8b) e LCEL
- Documentação completa (`README.md`, `SCRIPT_DETAILS.md`)
