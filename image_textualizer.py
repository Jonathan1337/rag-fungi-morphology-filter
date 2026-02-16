"""
image_textualizer.py
--------------------
Módulo responsável por gerar descrições textuais de imagens usando um modelo
de visão (LLaVA) via Ollama. As descrições são encapsuladas como objetos
Document do LangChain para integração direta no pipeline RAG.

v2: Prompts contextualizados por tipo de imagem + OCR + anti-alucinação.
"""

import base64
import os
import requests
from langchain_core.documents import Document


# Configurações padrão
DEFAULT_VISION_MODEL = "llava"
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# =====================================================================
#  PROMPTS POR TIPO DE IMAGEM
# =====================================================================

PROMPT_PLATE_GRID = """You are analyzing a MACROSCOPIC photograph of fungal colonies growing on agar plates (Petri dishes).
This is NOT a microscopy image. These are visible-to-the-naked-eye colonies photographed from above.

Document context: "{doc_title}"
Species discussed in the document: {species_list}
Text detected in the image via OCR: {ocr_text}

STRICT RULES:
- This shows PETRI DISH CULTURE PLATES, not microscope slides
- ONLY mention species or media names that appear in the OCR text or species list above
- Count the exact number of plates visible (rows × columns)
- Do NOT invent magnification values — these are photographs
- If you cannot read a label, say "label not legible" rather than guessing

Describe: the grid layout (rows × columns), colony colors and textures, and any visible media labels."""

PROMPT_MICROSCOPY = """You are analyzing microscopy images of fungal structures.

Document context: "{doc_title}"
Species discussed in the document: {species_list}
Text detected in the image via OCR: {ocr_text}

STRICT RULES:
- ONLY identify species that appear in the OCR text or species list above
- Describe visible structures: conidiophores, conidia, hyphae, vesicles, stipes
- Note microscopy type (optical/light microscopy vs SEM) based on visual appearance
- If you cannot identify a species precisely, describe the structure generically
- Do NOT fabricate labels, annotations, or text not visible in the image
- Do NOT invent text in languages other than what OCR detected

Describe the morphological features visible in each panel of the image."""

PROMPT_DIAGRAM = """You are analyzing a scientific diagram from a research document.

Document context: "{doc_title}"
Species discussed in the document: {species_list}
Text detected in the image via OCR: {ocr_text}

STRICT RULES:
- ONLY list species names, codes, or labels that were detected by OCR above
- Do NOT invent species names or isolate codes not present in the OCR text
- Describe the overall structure (tree, chart, table, graph)
- If you cannot read text clearly, state that explicitly
- Use the OCR text as your primary source of factual content

Describe the diagram structure and the relationships it shows."""

PROMPT_ICON = """This is a small icon or logo image.

Briefly describe what it appears to be (logo, seal, badge, etc.).
State that it contains no scientific content relevant to fungal morphology.
Keep the description to 1-2 sentences maximum."""

PROMPT_GENERIC = """You are analyzing a scientific image from the document: "{doc_title}"
Species discussed: {species_list}
Text detected via OCR: {ocr_text}

STRICT RULES:
- Do NOT invent or guess species names, labels, or data not in the OCR text
- Only describe what is clearly visible in the image
- If uncertain about any content, state your uncertainty explicitly
- Use the OCR-detected text as factual reference

Provide a factual, accurate description of this image."""

# Mapeamento tipo → prompt
PROMPT_MAP = {
    "plate_grid_row": PROMPT_PLATE_GRID,
    "plate_grid_matrix": PROMPT_PLATE_GRID,
    "microscopy_panel": PROMPT_MICROSCOPY,
    "diagram": PROMPT_DIAGRAM,
    "icon": PROMPT_ICON,
    "single_image": PROMPT_GENERIC,
}


# =====================================================================
#  FUNÇÕES CORE
# =====================================================================

def encode_image_to_base64(image_path):
    """
    Codifica uma imagem em base64 para envio à API do Ollama.

    Args:
        image_path (str): Caminho para o arquivo de imagem.

    Returns:
        str: String base64 da imagem.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_prompt(image_type, ocr_text="", species_list="", doc_title=""):
    """
    Constrói o prompt contextualizado para o modelo de visão.

    Args:
        image_type (str): Tipo da imagem (de classify_image).
        ocr_text (str): Texto extraído via OCR.
        species_list (str): Lista de espécies do documento.
        doc_title (str): Título do documento.

    Returns:
        str: Prompt formatado.
    """
    template = PROMPT_MAP.get(image_type, PROMPT_GENERIC)

    # Ícones não precisam de contexto
    if image_type == "icon":
        return template

    return template.format(
        doc_title=doc_title or "Unknown",
        species_list=species_list or "Not available",
        ocr_text=ocr_text or "No text detected",
    )


def textualize_image(image_path, model_name=DEFAULT_VISION_MODEL, prompt=None):
    """
    Gera uma descrição textual de uma imagem usando um modelo de visão via Ollama.

    Args:
        image_path (str): Caminho para o arquivo de imagem.
        model_name (str): Nome do modelo de visão no Ollama (default: 'llava').
        prompt (str): Prompt customizado para a descrição (opcional).

    Returns:
        str: Descrição textual gerada pelo modelo de visão.

    Raises:
        ConnectionError: Se não conseguir conectar ao Ollama.
        RuntimeError: Se o modelo retornar erro.
    """
    if prompt is None:
        prompt = PROMPT_GENERIC.format(
            doc_title="Unknown", species_list="Not available", ocr_text="No text detected"
        )

    image_base64 = encode_image_to_base64(image_path)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        result = response.json()
        description = result.get("response", "").strip()

        if not description:
            raise RuntimeError(
                f"Empty response from model '{model_name}' for image: {image_path}"
            )

        return description

    except requests.ConnectionError:
        raise ConnectionError(
            f"Could not connect to Ollama at {OLLAMA_BASE_URL}. "
            "Ensure Ollama is running with: 'ollama serve'"
        )
    except requests.HTTPError as e:
        raise RuntimeError(
            f"Ollama API error for model '{model_name}': {e}"
        )


def textualize_all_images(image_list, model_name=DEFAULT_VISION_MODEL,
                          document_context=None):
    """
    Processa uma lista de imagens extraídas e retorna Documents do LangChain
    com as descrições textuais contextualizadas.

    Args:
        image_list (list[dict]): Lista de dicionários com info das imagens
            (conforme retornado por image_extractor.extract_images_from_pdf).
            Deve incluir 'ocr_text' se OCR foi executado.
        model_name (str): Nome do modelo de visão no Ollama.
        document_context (dict): Contexto do documento com chaves:
            - 'title' (str): Título/nome do documento
            - 'species' (list[str]): Lista de espécies mencionadas

    Returns:
        list[Document]: Lista de Documents do LangChain.
    """
    documents = []
    total = len(image_list)

    # Preparar contexto
    doc_title = ""
    species_str = "Not available"
    if document_context:
        doc_title = document_context.get("title", "")
        species = document_context.get("species", [])
        if species:
            species_str = ", ".join(species)

    print(f"\n{'='*60}")
    print(f"  IMAGE TEXTUALIZER — Generating descriptions via {model_name}")
    print(f"  Total images to process: {total}")
    if document_context:
        print(f"  Document context: {doc_title}")
        print(f"  Known species: {species_str}")
    print(f"{'='*60}")

    for i, img_info in enumerate(image_list, 1):
        image_path = img_info["path"]
        page_number = img_info["page_number"]
        image_type = img_info.get("image_type", "single_image")
        ocr_text = img_info.get("ocr_text", {}).get("full_text", "") if isinstance(
            img_info.get("ocr_text"), dict) else ""

        print(f"\n  [{i}/{total}] Processing: {os.path.basename(image_path)} "
              f"(page {page_number}, type: {image_type})...")

        try:
            # Construir prompt contextualizado
            prompt = build_prompt(
                image_type=image_type,
                ocr_text=ocr_text,
                species_list=species_str,
                doc_title=doc_title,
            )

            description = textualize_image(image_path, model_name, prompt)

            # Criar o conteúdo enriquecido com contexto
            content = (
                f"[IMAGE FROM PAGE {page_number} | TYPE: {image_type}] "
                f"The following is a textual description of a scientific image "
                f"extracted from the document:\n\n{description}"
            )

            # Adicionar OCR como referência
            if ocr_text:
                content += f"\n\n[OCR TEXT DETECTED]: {ocr_text}"

            doc = Document(
                page_content=content,
                metadata={
                    "source": image_path,
                    "page": page_number,
                    "type": "image_description",
                    "image_type": image_type,
                    "image_index": img_info["image_index"],
                    "dimensions": f"{img_info['width']}x{img_info['height']}",
                    "ocr_text": ocr_text,
                },
            )
            documents.append(doc)

            preview = description[:120].replace("\n", " ")
            print(f"  ✓ Description generated ({len(description)} chars): "
                  f"{preview}...")

        except (ConnectionError, RuntimeError) as e:
            print(f"  ✗ Error: {e}")
        except Exception as e:
            print(f"  ✗ Unexpected error processing {image_path}: {e}")

    print(f"\n  Summary: {len(documents)}/{total} images successfully textualized")
    print(f"{'='*60}\n")

    return documents


if __name__ == "__main__":
    from image_extractor import extract_images_from_pdf
    from rag_local import DATA_PATH, PDF_FILENAME

    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)

    images = extract_images_from_pdf(pdf_path)

    if images:
        print("\nTesting textualization with first image only...")
        test_docs = textualize_all_images(images[:1])

        if test_docs:
            print(f"\n--- Generated Document ---")
            print(f"Content:\n{test_docs[0].page_content}")
            print(f"\nMetadata: {test_docs[0].metadata}")
    else:
        print("No images found in PDF.")
