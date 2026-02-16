"""
image_extractor.py
------------------
Módulo responsável por extrair imagens embutidas em arquivos PDF usando PyMuPDF (fitz).
Parte do pipeline de textualização multimodal do RAG Fungi Morphology Filter.
"""

import os
import fitz  # PyMuPDF
from PIL import Image
import io


# Dimensão mínima para filtrar ícones e elementos decorativos
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100


def extract_images_from_pdf(pdf_path, output_dir="data/extracted_images/"):
    """
    Extrai todas as imagens de um arquivo PDF e salva como PNG.

    Args:
        pdf_path (str): Caminho para o arquivo PDF.
        output_dir (str): Diretório onde as imagens serão salvas.

    Returns:
        list[dict]: Lista de dicionários com informações de cada imagem extraída:
            - 'path': caminho do arquivo de imagem salvo
            - 'page_number': número da página do PDF (1-indexed)
            - 'image_index': índice da imagem na página
            - 'width': largura em pixels
            - 'height': altura em pixels
    """
    # Criar o diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    extracted_images = []
    total_skipped = 0

    print(f"\n{'='*60}")
    print(f"  IMAGE EXTRACTOR — Extracting images from PDF")
    print(f"  Source: {pdf_path}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]  # cross-reference number da imagem

            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Converter bytes para objeto PIL Image para verificar dimensões
                pil_image = Image.open(io.BytesIO(image_bytes))
                width, height = pil_image.size

                # Filtrar imagens muito pequenas (ícones, decorações)
                if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                    total_skipped += 1
                    continue

                # Converter para RGB se necessário (CMYK, P, etc.)
                if pil_image.mode not in ("RGB", "RGBA"):
                    pil_image = pil_image.convert("RGB")

                # Gerar nome do arquivo: page{N}_img{M}.png
                filename = f"page{page_num + 1}_img{img_index + 1}.png"
                filepath = os.path.join(output_dir, filename)

                # Salvar como PNG
                pil_image.save(filepath, "PNG")

                image_info = {
                    "path": filepath,
                    "page_number": page_num + 1,
                    "image_index": img_index + 1,
                    "width": width,
                    "height": height,
                    "image_type": classify_image(width, height),
                }
                extracted_images.append(image_info)

                print(f"  ✓ Page {page_num + 1}, Image {img_index + 1}: "
                      f"{width}x{height}px → {filename} "
                      f"[{image_info['image_type']}]")

            except Exception as e:
                print(f"  ✗ Page {page_num + 1}, Image {img_index + 1}: "
                      f"Error extracting — {e}")

    doc.close()

    print(f"\n  Summary: {len(extracted_images)} images extracted, "
          f"{total_skipped} skipped (too small)")
    print(f"{'='*60}\n")

    return extracted_images


def classify_image(width, height):
    """
    Classifica o tipo de imagem com base nas dimensões e aspect ratio.

    Heurística usada para selecionar o prompt adequado na textualização.

    Args:
        width (int): Largura em pixels.
        height (int): Altura em pixels.

    Returns:
        str: Tipo da imagem ('icon', 'plate_grid_row', 'plate_grid_matrix',
             'microscopy_panel', 'diagram', 'single_image').
    """
    # Ícones e logos
    if width < 200 and height < 200:
        return "icon"

    ratio = width / height if height > 0 else 1

    # Imagem muito larga → provavelmente linha de placas de Petri
    if ratio > 2.5 and width > 800:
        return "plate_grid_row"

    # Imagem muito alta → painel de microscopia ou matriz de placas
    if ratio < 0.6 and height > 1500:
        # Se for muito alta e larga, provavelmente é uma matriz de placas
        if width > 1000:
            return "plate_grid_matrix"
        return "microscopy_panel"

    # Imagem grande e ~quadrada → diagrama ou imagem única
    if width > 800 and height > 800:
        if ratio < 0.8:
            return "microscopy_panel"
        return "diagram"

    return "single_image"


if __name__ == "__main__":
    # Teste standalone
    from rag_local import DATA_PATH, PDF_FILENAME

    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)
    images = extract_images_from_pdf(pdf_path)

    print(f"\nTotal images extracted: {len(images)}")
    for img in images:
        print(f"  - {img['path']} "
              f"(page {img['page_number']}, {img['width']}x{img['height']}px)")
