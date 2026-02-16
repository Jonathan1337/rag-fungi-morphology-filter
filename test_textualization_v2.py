"""
test_textualization_v2.py
--------------------------
Script de teste do pipeline v2 de textualização multimodal.
Executa: extração → classificação → OCR → textualização contextualizada → validação.
Salva os resultados em test_llava_descriptions_v2.md para comparação.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_extractor import extract_images_from_pdf
from ocr_extractor import extract_text_batch
from image_textualizer import textualize_image, build_prompt
from description_validator import extract_species_from_text, validate_description
from langchain_community.document_loaders import PyPDFLoader

DATA_PATH = "data/"
PDF_FILENAME = "Morphological_and_Molecular_Diversity_of_Aspergillus_From_Corn_Grain.pdf"
EXTRACTED_IMAGES_PATH = "data/extracted_images/"
OUTPUT_FILE = "test_llava_descriptions_v2.md"


def main():
    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)

    # 1. Carregar texto do PDF para extrair contexto
    print("\n=== Phase 1: Loading document text ===")
    loader = PyPDFLoader(pdf_path)
    text_docs = loader.load()
    full_text = " ".join([doc.page_content for doc in text_docs])
    species_list = extract_species_from_text(full_text)
    species_str = ", ".join(species_list)
    doc_title = PDF_FILENAME.replace("_", " ").replace(".pdf", "")
    print(f"  Document: {doc_title}")
    print(f"  Species found: {species_str}")

    # 2. Extrair imagens (com classificação)
    print("\n=== Phase 2: Image extraction + classification ===")
    images = extract_images_from_pdf(pdf_path, EXTRACTED_IMAGES_PATH)

    if not images:
        print("Nenhuma imagem encontrada.")
        return

    # 3. OCR pre-pass
    print("\n=== Phase 3: OCR pre-pass ===")
    ocr_results = extract_text_batch(images)

    # Anexar OCR às imagens
    for img, ocr in zip(images, ocr_results):
        img["ocr_text"] = ocr

    # 4. Textualização contextualizada + validação
    print("\n=== Phase 4: Context-aware textualization + validation ===")
    output_lines = []
    output_lines.append("# Descrições LLaVA v2 — Textualização Contextualizada\n")
    output_lines.append(f"Pipeline v2 com OCR pre-pass, prompts por tipo de imagem,")
    output_lines.append(f"e validação contra o texto do documento.\n")
    output_lines.append(f"**Espécies no documento:** {species_str}\n")
    output_lines.append("---\n")

    for i, img_info in enumerate(images, 1):
        image_path = img_info["path"]
        page = img_info["page_number"]
        width = img_info["width"]
        height = img_info["height"]
        image_type = img_info.get("image_type", "single_image")
        ocr = img_info.get("ocr_text", {})
        ocr_text = ocr.get("full_text", "") if isinstance(ocr, dict) else ""
        filename = os.path.basename(image_path)

        print(f"\n[{i}/{len(images)}] Textualizando: {filename} "
              f"(type: {image_type}, OCR: {ocr.get('num_detections', 0)} texts)...")

        # Construir prompt contextualizado
        prompt = build_prompt(
            image_type=image_type,
            ocr_text=ocr_text,
            species_list=species_str,
            doc_title=doc_title,
        )

        try:
            description = textualize_image(image_path, prompt=prompt)

            # Validar
            validation = validate_description(description, species_list, ocr_text)

            output_lines.append(f"## {filename} (Página {page} — {width}×{height}px)\n")
            output_lines.append(f"**Tipo classificado:** `{image_type}`\n")
            output_lines.append(f"**OCR detectado:** {ocr_text if ocr_text else '(nenhum)'}\n")
            output_lines.append(f"**Confiança:** {validation['confidence_score']:.2f}\n")

            if validation["verified_species"]:
                output_lines.append(f"**Espécies verificadas:** {', '.join(validation['verified_species'])}\n")
            if validation["unverified_species"]:
                output_lines.append(f"**⚠️ Espécies NÃO verificadas:** {', '.join(validation['unverified_species'])}\n")
            if validation["issues"]:
                output_lines.append(f"**Issues:** {'; '.join(validation['issues'])}\n")

            output_lines.append(f"\n### Descrição\n")
            output_lines.append(f"{validation['cleaned_description']}\n")
            output_lines.append("---\n")

            print(f"  ✓ OK ({len(description)} chars, conf: {validation['confidence_score']:.2f})")
            if validation["unverified_species"]:
                print(f"  ⚠ Unverified: {validation['unverified_species']}")

        except Exception as e:
            output_lines.append(f"## {filename} (Página {page} — {width}×{height}px)\n")
            output_lines.append(f"**ERRO:** {e}\n")
            output_lines.append("---\n")
            print(f"  ✗ Erro: {e}")

    # 5. Salvar
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\n✓ Descrições v2 salvas em: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
