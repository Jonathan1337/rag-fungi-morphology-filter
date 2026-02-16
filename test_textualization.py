"""
test_textualization.py
-----------------------
Script de teste que executa a textualização LLaVA em todas as imagens extraídas
e salva as descrições em um arquivo markdown para comparação.
"""

import os
import sys

# Adicionar o diretório do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_extractor import extract_images_from_pdf
from image_textualizer import textualize_image

DATA_PATH = "data/"
PDF_FILENAME = "Morphological_and_Molecular_Diversity_of_Aspergillus_From_Corn_Grain.pdf"
EXTRACTED_IMAGES_PATH = "data/extracted_images/"
OUTPUT_FILE = "test_llava_descriptions.md"


def main():
    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)

    # 1. Extrair imagens (caso ainda não existam)
    images = extract_images_from_pdf(pdf_path, EXTRACTED_IMAGES_PATH)

    if not images:
        print("Nenhuma imagem encontrada.")
        return

    # 2. Gerar descrições via LLaVA
    output_lines = []
    output_lines.append("# Descrições LLaVA — Textualização Automática\n")
    output_lines.append("Descrições geradas automaticamente pelo modelo de visão **LLaVA** via Ollama")
    output_lines.append(f"para as imagens extraídas do PDF *\"{PDF_FILENAME}\"*.\n")
    output_lines.append("---\n")

    for i, img_info in enumerate(images, 1):
        image_path = img_info["path"]
        page = img_info["page_number"]
        width = img_info["width"]
        height = img_info["height"]
        filename = os.path.basename(image_path)

        print(f"\n[{i}/{len(images)}] Textualizando: {filename}...")

        try:
            description = textualize_image(image_path)

            output_lines.append(f"## {filename} (Página {page} — {width}×{height}px)\n")
            output_lines.append(f"{description}\n")
            output_lines.append("---\n")

            print(f"  ✓ OK ({len(description)} chars)")
        except Exception as e:
            output_lines.append(f"## {filename} (Página {page} — {width}×{height}px)\n")
            output_lines.append(f"**ERRO:** {e}\n")
            output_lines.append("---\n")
            print(f"  ✗ Erro: {e}")

    # 3. Salvar arquivo markdown
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\n✓ Descrições salvas em: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
