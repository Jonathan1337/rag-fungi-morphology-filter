"""
description_validator.py
------------------------
Módulo responsável por validar descrições geradas pelo modelo de visão (LLaVA)
contra o texto do documento original. Detecta alucinações, valida espécies
mencionadas, e atribui score de confiança.
"""

import re
from langchain_core.documents import Document

# Padrão regex para detectar nomes de espécies no formato científico
# Captura: "A. niger", "Aspergillus niger", "A. tubingensis", etc.
SPECIES_PATTERN = re.compile(
    r'\b(?:'
    r'(?:Aspergillus|A\.)\s+(?:niger|flavus|fumigatus|oryzae|terreus|clavatus|tubingensis|parasiticus|nidulans|carbonarius|luchuensis|welwitschiae|japonicus|aculeatus|brasiliensis|uvarum|foetidus)'
    r')',
    re.IGNORECASE
)

# Padrão para nomes genéricos de espécies em formato binomial
BINOMIAL_PATTERN = re.compile(
    r'\b([A-Z][a-z]+\.?\s+[a-z]+)\b'
)


def extract_species_from_text(text):
    """
    Extrai nomes de espécies mencionadas em um texto (normalmente o PDF completo).

    Args:
        text (str): Texto do documento.

    Returns:
        list[str]: Lista de nomes de espécies únicos encontrados.
    """
    matches = SPECIES_PATTERN.findall(text)

    # Normalizar: expandir abreviações
    normalized = set()
    for match in matches:
        # Normalizar "A." → "Aspergillus"
        cleaned = re.sub(r'^A\.\s*', 'Aspergillus ', match, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        normalized.add(cleaned)

    return sorted(normalized)


def _find_species_in_description(description):
    """
    Encontra todas as menções a espécies na descrição gerada pelo LLaVA.

    Returns:
        set[str]: Nomes de espécies mencionados na descrição.
    """
    # Primeiro, buscar padrões conhecidos de Aspergillus
    aspergillus_matches = SPECIES_PATTERN.findall(description)

    # Também capturar padrões binomiais genéricos (podem ser alucinações)
    binomial_matches = BINOMIAL_PATTERN.findall(description)

    all_species = set()

    for m in aspergillus_matches:
        cleaned = re.sub(r'^A\.\s*', 'Aspergillus ', m, flags=re.IGNORECASE)
        all_species.add(cleaned.strip())

    for m in binomial_matches:
        # Filtrar palavras comuns que parecem binomial mas não são
        words = m.split()
        first_word = words[0].rstrip(".")
        second_word = words[1] if len(words) > 1 else ""

        # Skip list expandida para evitar falsos positivos
        skip_words = {
            # Pronomes e artigos
            "The", "This", "These", "That", "Those", "Each", "Some",
            "It", "Its", "They", "Them", "Their", "There",
            "We", "Our", "You", "Your",
            # Conjunções e preposições
            "In", "On", "At", "By", "For", "From", "Into", "With",
            "Without", "Within", "Between", "Below", "Above", "Under",
            # Verbos comuns
            "If", "Are", "Is", "Was", "Were", "Has", "Have", "Had",
            "Can", "May", "Will", "Could", "Would", "Should", "Must",
            # Adjetivos e advérbos
            "Black", "White", "Light", "Dark", "Small", "Large",
            "More", "Most", "Very", "Also", "However", "Although",
            "Possible", "Specific", "Different", "Similar", "Various",
            # Termos de documento/imagem
            "Image", "Figure", "Table", "Page", "Scale", "Group",
            "Type", "Panel", "Section", "Part", "Row", "Column",
            "Note", "Based", "Overall",
            # Termos científicos genéricos (não são espécies)
            "Petri", "Morphological", "Molecular", "Diversity",
            "Scanning", "Electron", "Colony",
        }

        # Palavras no segundo termo que indicam que não é espécie
        skip_second_words = {
            "the", "is", "are", "and", "or", "not", "may", "can",
            "will", "be", "of", "in", "on", "at", "to", "for",
            "from", "with", "has", "have", "was", "were", "more",
            "you", "we", "it", "its",
        }

        if first_word in skip_words:
            continue
        if second_word.lower() in skip_second_words:
            continue
        # Requer que o segundo termo tenha pelo menos 4 caracteres
        if len(second_word) < 4:
            continue

        all_species.add(m.strip())

    return all_species


def validate_description(description, valid_species, ocr_text=""):
    """
    Valida uma descrição gerada pelo LLaVA contra espécies conhecidas e texto OCR.

    Args:
        description (str): Descrição gerada pelo modelo de visão.
        valid_species (list[str]): Lista de espécies válidas do documento.
        ocr_text (str): Texto extraído via OCR da imagem.

    Returns:
        dict: Resultado da validação contendo:
            - 'confidence_score': float 0.0-1.0
            - 'unverified_species': list de espécies não verificadas
            - 'verified_species': list de espécies confirmadas
            - 'issues': list de problemas encontrados
            - 'cleaned_description': descrição com marcações de não-verificado
    """
    valid_set = {s.lower() for s in valid_species}
    ocr_lower = ocr_text.lower()

    mentioned = _find_species_in_description(description)

    verified = []
    unverified = []
    issues = []

    for species in mentioned:
        species_lower = species.lower()
        # Verificar se a espécie está no documento
        in_doc = any(v in species_lower or species_lower in v for v in valid_set)
        # Verificar se aparece no OCR
        in_ocr = species_lower in ocr_lower or any(
            part in ocr_lower for part in species_lower.split()
        )

        if in_doc or in_ocr:
            verified.append(species)
        else:
            unverified.append(species)

    # Detectar fabricação de idiomas (texto em francês, etc.)
    french_indicators = [
        "du spore", "de la hyphae", "croissance", "germination du",
        "avec conidia", "condidiophore"
    ]
    for indicator in french_indicators:
        if indicator.lower() in description.lower():
            issues.append(f"Possible fabricated French text detected: '{indicator}'")

    # Detectar magnificações inventadas (quando a imagem é de placas)
    magnification_pattern = re.compile(r'\b\d+[xX]\b')
    if magnification_pattern.search(description):
        # Verificar se magnificação aparece no OCR
        if not magnification_pattern.search(ocr_text):
            issues.append("Magnification values mentioned but not found in OCR text")

    # Calcular score de confiança
    total_claims = len(mentioned)
    if total_claims == 0:
        confidence = 0.8  # Sem espécies mencionadas, confiança neutra
    else:
        verified_ratio = len(verified) / total_claims
        confidence = verified_ratio * 0.9  # Max 0.9 se tudo verificado

    # Penalizar por issues
    confidence -= len(issues) * 0.1
    confidence = max(0.0, min(1.0, confidence))

    # Se não há unverified nem issues, confiança alta
    if not unverified and not issues:
        confidence = max(confidence, 0.8)

    # Marcar espécies não verificadas na descrição
    cleaned = description
    for species in unverified:
        cleaned = cleaned.replace(species, f"[UNVERIFIED: {species}]")

    return {
        "confidence_score": round(confidence, 2),
        "unverified_species": unverified,
        "verified_species": verified,
        "issues": issues,
        "cleaned_description": cleaned,
    }


def validate_all_descriptions(documents, valid_species):
    """
    Valida todas as descrições de imagem e atualiza o metadata dos Documents.

    Args:
        documents (list[Document]): Lista de Documents de descrição de imagem.
        valid_species (list[str]): Lista de espécies válidas do documento.

    Returns:
        list[Document]: Documents atualizados com metadata de validação.
    """
    validated = []

    print(f"\n{'='*60}")
    print(f"  DESCRIPTION VALIDATOR — Checking {len(documents)} descriptions")
    print(f"  Valid species: {', '.join(valid_species[:10])}")
    print(f"{'='*60}")

    for i, doc in enumerate(documents, 1):
        ocr_text = doc.metadata.get("ocr_text", "")

        result = validate_description(
            doc.page_content, valid_species, ocr_text
        )

        # Atualizar metadata com validação
        doc.metadata["confidence_score"] = result["confidence_score"]
        doc.metadata["unverified_species"] = result["unverified_species"]
        doc.metadata["verified_species"] = result["verified_species"]
        doc.metadata["validation_issues"] = result["issues"]

        # Usar descrição limpa se houver espécies não verificadas
        if result["unverified_species"]:
            doc.page_content = result["cleaned_description"]

        validated.append(doc)

        # Log
        status = "✓" if result["confidence_score"] >= 0.7 else "⚠"
        log_msg = f"  [{i}/{len(documents)}] {status} Confidence: {result['confidence_score']:.2f}"
        if result["unverified_species"]:
            unv = result["unverified_species"]
            log_msg += f" | Unverified: {unv}"
        if result["issues"]:
            log_msg += f" | Issues: {len(result['issues'])}"
        print(log_msg)

    print(f"\n  Summary: {len(validated)} descriptions validated")
    avg_confidence = sum(d.metadata.get("confidence_score", 0) for d in validated) / max(len(validated), 1)
    print(f"  Average confidence: {avg_confidence:.2f}")
    print(f"{'='*60}\n")

    return validated


if __name__ == "__main__":
    # Teste standalone
    test_species = [
        "Aspergillus niger", "Aspergillus flavus", "Aspergillus fumigatus",
        "Aspergillus oryzae", "Aspergillus terreus", "Aspergillus clavatus",
        "Aspergillus tubingensis",
    ]

    # Simular uma descrição com alucinação
    test_desc = (
        "This image shows Saccharomyces cerevisiae and Aspergillus niger "
        "colonies growing on agar. Candida albicans is also visible."
    )

    result = validate_description(test_desc, test_species)
    print(f"Confidence: {result['confidence_score']}")
    print(f"Verified: {result['verified_species']}")
    print(f"Unverified: {result['unverified_species']}")
    print(f"Issues: {result['issues']}")
    print(f"\nCleaned:\n{result['cleaned_description']}")
