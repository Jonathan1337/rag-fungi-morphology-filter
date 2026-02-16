"""
ocr_extractor.py
-----------------
Módulo responsável por extrair texto de imagens usando EasyOCR.
O texto extraído é fornecido como contexto factual ao modelo de visão (LLaVA),
reduzindo alucinações e melhorando a precisão da textualização.
"""

import easyocr
import cv2
import numpy as np
from PIL import Image

# Inicialização lazy do reader (evita carregar modelo na importação)
_reader = None


def _get_reader():
    """Inicializa o reader EasyOCR sob demanda (singleton)."""
    global _reader
    if _reader is None:
        print("  Initializing EasyOCR reader (first run may download models)...")
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def preprocess_image(image_path):
    """
    Pré-processa a imagem para melhorar a qualidade do OCR.

    Aplica: grayscale → resize → threshold adaptativo → denoising.

    Args:
        image_path (str): Caminho para a imagem.

    Returns:
        numpy.ndarray: Imagem pré-processada.
    """
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Converter para grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Redimensionar se muito pequeno (mínimo 300px de lado menor)
    h, w = gray.shape
    min_side = min(h, w)
    if min_side < 300:
        scale = 300 / min_side
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Denoising suave
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Threshold adaptativo para melhorar contraste do texto
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return binary


def extract_text_from_image(image_path, min_confidence=0.3, preprocess=True):
    """
    Extrai texto de uma imagem usando EasyOCR.

    Args:
        image_path (str): Caminho para a imagem.
        min_confidence (float): Confiança mínima para aceitar uma detecção (0.0-1.0).
        preprocess (bool): Se True, aplica pré-processamento de imagem.

    Returns:
        dict: Dicionário contendo:
            - 'raw_texts': lista de strings detectadas
            - 'full_text': texto concatenado
            - 'detections': lista de (texto, confiança) para cada detecção
            - 'num_detections': total de detecções aceitas
    """
    reader = _get_reader()

    try:
        # Tentar com imagem pré-processada e original
        if preprocess:
            try:
                processed = preprocess_image(image_path)
                results_processed = reader.readtext(processed)
            except Exception:
                results_processed = []

            results_original = reader.readtext(image_path)

            # Usar o resultado com mais detecções
            if len(results_processed) >= len(results_original):
                results = results_processed
            else:
                results = results_original
        else:
            results = reader.readtext(image_path)

    except Exception as e:
        print(f"  ⚠ OCR error for {image_path}: {e}")
        return {
            "raw_texts": [],
            "full_text": "",
            "detections": [],
            "num_detections": 0,
        }

    # Filtrar por confiança mínima
    filtered = []
    for bbox, text, confidence in results:
        if confidence >= min_confidence and text.strip():
            filtered.append({
                "text": text.strip(),
                "confidence": round(confidence, 3),
            })

    # Remover duplicatas mantendo a maior confiança
    seen = {}
    for det in filtered:
        t = det["text"].lower()
        if t not in seen or det["confidence"] > seen[t]["confidence"]:
            seen[t] = det
    unique = list(seen.values())

    raw_texts = [d["text"] for d in unique]
    full_text = " | ".join(raw_texts)

    return {
        "raw_texts": raw_texts,
        "full_text": full_text,
        "detections": [(d["text"], d["confidence"]) for d in unique],
        "num_detections": len(unique),
    }


def extract_text_batch(image_list):
    """
    Extrai texto de múltiplas imagens.

    Args:
        image_list (list[dict]): Lista de dicionários com info das imagens
            (com chave 'path').

    Returns:
        list[dict]: Lista de resultados OCR, um por imagem.
    """
    results = []
    total = len(image_list)

    print(f"\n{'='*60}")
    print(f"  OCR EXTRACTOR — Extracting text labels from images")
    print(f"  Total images: {total}")
    print(f"{'='*60}")

    for i, img_info in enumerate(image_list, 1):
        image_path = img_info["path"]
        filename = image_path.split("/")[-1].split("\\")[-1]
        print(f"\n  [{i}/{total}] OCR: {filename}...", end="")

        ocr_result = extract_text_from_image(image_path)
        results.append(ocr_result)

        n = ocr_result["num_detections"]
        if n > 0:
            preview = ocr_result["full_text"][:80]
            print(f" ✓ {n} texts: {preview}...")
        else:
            print(f" (no text detected)")

    print(f"\n  Summary: OCR complete for {total} images")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    import os
    from image_extractor import extract_images_from_pdf

    DATA_PATH = "data/"
    PDF_FILENAME = "Morphological_and_Molecular_Diversity_of_Aspergillus_From_Corn_Grain.pdf"

    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)
    images = extract_images_from_pdf(pdf_path)

    if images:
        ocr_results = extract_text_batch(images)
        for img, ocr in zip(images, ocr_results):
            filename = os.path.basename(img["path"])
            print(f"\n{filename}:")
            for text, conf in ocr["detections"]:
                print(f"  [{conf:.2f}] {text}")
