"""
test_final_comparison.py
-------------------------
Script de teste final: compara as descrições manuais (ground truth) com as
descrições geradas pelo pipeline v2 e gera um relatório comparativo detalhado.

Uso:
    python test_final_comparison.py
"""

import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_extractor import extract_images_from_pdf
from ocr_extractor import extract_text_batch
from image_textualizer import textualize_image, build_prompt
from description_validator import extract_species_from_text, validate_description
from langchain_community.document_loaders import PyPDFLoader

DATA_PATH = "data/"
PDF_FILENAME = "Morphological_and_Molecular_Diversity_of_Aspergillus_From_Corn_Grain.pdf"
EXTRACTED_IMAGES_PATH = "data/extracted_images/"

MANUAL_FILE = "test_manual_descriptions.md"
OUTPUT_ALGO = "test_llava_descriptions_v2.md"
OUTPUT_COMPARISON = "test_comparison_v2.md"


# =================================================================
#  CRITÉRIOS DE AVALIAÇÃO POR IMAGEM (ground truth)
# =================================================================

GROUND_TRUTH = {
    "page1_img1.png": {
        "type": "icon",
        "layout": "single icon",
        "key_facts": [
            "logo ou selo de periódico",
            "bookmark/marcador vermelho",
            "sem conteúdo científico sobre fungos",
        ],
        "species": [],
        "labels": [],
    },
    "page1_img2.png": {
        "type": "capa de periódico",
        "layout": "single image",
        "key_facts": [
            "HAYATI Journal of Biosciences",
            "aves aquáticas voando",
            "ScienceDirect",
            "capa institucional do periódico",
        ],
        "species": [],
        "labels": ["HAYATI", "Journal of Biosciences"],
    },
    "page3_img1.png": {
        "type": "placas de Petri",
        "layout": "grade 2×5 (10 placas)",
        "key_facts": [
            "placas de Petri com colônias",
            "2 linhas (A. niger Group I e II)",
            "5 meios de cultura por linha",
            "fotografia macroscópica, NÃO microscopia",
        ],
        "species": ["A. niger", "A. tubingensis"],
        "labels": ["MEA", "CYA", "CYA37", "DG-18", "CREA"],
    },
    "page3_img2.png": {
        "type": "microscopia óptica e SEM",
        "layout": "painel 2×4 (8 imagens)",
        "key_facts": [
            "conidióforos em microscopia óptica",
            "conídios em SEM",
            "comparação entre dois grupos de A. niger",
            "barras de escala presentes",
        ],
        "species": ["A. niger", "A. tubingensis"],
        "labels": ["conidiophore/stipe", "conidia"],
    },
    "page4_img1.png": {
        "type": "microscopia óptica e SEM",
        "layout": "painel 5×4 (20 imagens)",
        "key_facts": [
            "5 espécies de Aspergillus",
            "conidióforos e conídios",
            "microscopia óptica e SEM alternados",
            "barras de escala presentes",
        ],
        "species": ["A. flavus", "A. oryzae", "A. fumigatus", "A. clavatus", "A. terreus"],
        "labels": ["conidiophore/stipe", "conidia"],
    },
    "page5_img1.png": {
        "type": "placas de Petri (matriz ampla)",
        "layout": "matriz 8×5 (40 placas)",
        "key_facts": [
            "5 espécies em 8 condições de cultivo",
            "anverso (Observe) e reverso (Reverse)",
            "fotografia macroscópica, NÃO microscopia",
            "40 placas de Petri",
        ],
        "species": ["A. flavus", "A. oryzae", "A. fumigatus", "A. clavatus", "A. terreus"],
        "labels": ["MEA", "CYA", "CYA37", "DG-18", "CREA", "Observe", "Reverse"],
    },
    "page7_img1.png": {
        "type": "árvore filogenética",
        "layout": "dendrograma com clados",
        "key_facts": [
            "árvore filogenética molecular",
            "isolados de Aspergillus de grãos de milho",
            "códigos de isolados (USMP18, USMN20, etc.)",
            "referências NRRL/ICPA",
            "NÃO é microscopia",
        ],
        "species": ["A. niger", "A. tubingensis", "A. flavus", "A. oryzae",
                     "A. fumigatus", "A. clavatus", "A. terreus"],
        "labels": ["NRRL", "ICPA"],
    },
}


def load_algo_descriptions(filepath):
    """Carrega as descrições do algoritmo do arquivo md."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Extrair blocos por imagem
    pattern = r'## (page\d+_img\d+\.png) \(.*?\)\n(.*?)(?=\n## page|\n---\s*$|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)

    descriptions = {}
    for name, block in matches:
        descriptions[name] = block.strip()

    return descriptions


def evaluate_image(image_name, algo_text, ground_truth_info):
    """Avalia a descrição do algoritmo contra o ground truth."""
    algo_lower = algo_text.lower()
    results = {
        "image": image_name,
        "scores": {},
        "details": [],
        "overall": "❌",
    }

    # 1. Tipo de imagem
    gt_type = ground_truth_info["type"].lower()
    type_keywords = {
        "icon": ["icon", "logo", "badge", "seal", "ícone"],
        "capa de periódico": ["cover", "journal", "capa", "periódico"],
        "placas de petri": ["petri", "agar", "plate", "colony", "colonies", "colônia"],
        "microscopia": ["microscop", "conidiophore", "conidia", "hyphae", "sem ", "optical"],
        "árvore filogenética": ["phylogenetic", "tree", "cladogram", "dendro", "evolutionary"],
    }

    type_correct = False
    for type_key, keywords in type_keywords.items():
        if type_key in gt_type:
            type_correct = any(kw in algo_lower for kw in keywords)
            break

    results["scores"]["tipo_imagem"] = "✅" if type_correct else "❌"
    if type_correct:
        results["details"].append(f"✅ Tipo correto: identificou como '{gt_type}'")
    else:
        results["details"].append(f"❌ Tipo incorreto: esperado '{gt_type}'")

    # 2. Rótulos/Labels OCR
    gt_labels = ground_truth_info["labels"]
    if gt_labels:
        found = sum(1 for label in gt_labels if label.lower() in algo_lower)
        ratio = found / len(gt_labels)
        if ratio >= 0.7:
            results["scores"]["rotulos"] = "✅"
            results["details"].append(f"✅ Rótulos: {found}/{len(gt_labels)} detectados")
        elif ratio >= 0.3:
            results["scores"]["rotulos"] = "⚠️"
            results["details"].append(f"⚠️ Rótulos: {found}/{len(gt_labels)} detectados (parcial)")
        else:
            results["scores"]["rotulos"] = "❌"
            results["details"].append(f"❌ Rótulos: apenas {found}/{len(gt_labels)} detectados")
    else:
        results["scores"]["rotulos"] = "—"

    # 3. Espécies
    gt_species = ground_truth_info["species"]
    if gt_species:
        found = sum(1 for sp in gt_species if sp.lower().replace("a. ", "aspergillus ") in algo_lower
                     or sp.lower() in algo_lower)
        ratio = found / len(gt_species)
        if ratio >= 0.7:
            results["scores"]["especies"] = "✅"
            results["details"].append(f"✅ Espécies: {found}/{len(gt_species)} corretas")
        elif ratio >= 0.3:
            results["scores"]["especies"] = "⚠️"
            results["details"].append(f"⚠️ Espécies: {found}/{len(gt_species)} corretas (parcial)")
        else:
            results["scores"]["especies"] = "❌"
            results["details"].append(f"❌ Espécies: apenas {found}/{len(gt_species)} corretas")
    else:
        results["scores"]["especies"] = "—"

    # 4. Alucinações — Verificar espécies inventadas
    hallucinated_species = []
    false_species = [
        "saccharomyces cerevisiae", "candida albicans",
        "penicillium chrysogenum", "botrytis cinerea",
        "neurospora crassa", "fusarium",
    ]
    for fs in false_species:
        if fs in algo_lower:
            hallucinated_species.append(fs)

    if hallucinated_species:
        results["scores"]["alucinacoes"] = "❌"
        results["details"].append(f"❌ Alucinações: espécies inventadas — {', '.join(hallucinated_species)}")
    else:
        results["scores"]["alucinacoes"] = "✅"
        results["details"].append("✅ Sem alucinações de espécies falsas")

    # 5. Fatos-chave
    gt_facts = ground_truth_info["key_facts"]
    fact_checks = []
    for fact in gt_facts:
        fact_lower = fact.lower()
        # Verificar se a ideia principal está presente
        keywords = [w for w in fact_lower.split() if len(w) > 3]
        match_count = sum(1 for kw in keywords if kw in algo_lower)
        fact_checks.append(match_count >= max(1, len(keywords) // 2))

    facts_found = sum(fact_checks)
    facts_ratio = facts_found / len(gt_facts) if gt_facts else 1
    if facts_ratio >= 0.6:
        results["scores"]["fatos"] = "✅"
        results["details"].append(f"✅ Fatos-chave: {facts_found}/{len(gt_facts)} presentes")
    elif facts_ratio >= 0.3:
        results["scores"]["fatos"] = "⚠️"
        results["details"].append(f"⚠️ Fatos-chave: {facts_found}/{len(gt_facts)} presentes (parcial)")
    else:
        results["scores"]["fatos"] = "❌"
        results["details"].append(f"❌ Fatos-chave: apenas {facts_found}/{len(gt_facts)} presentes")

    # 6. Anti-padrão: chamar placas de microscopia
    is_plate_image = "placa" in gt_type.lower() or "petri" in gt_type.lower()
    if is_plate_image:
        if "microscop" in algo_lower and "not" not in algo_lower.split("microscop")[0][-20:]:
            # Verificar se diz "not microscopy" ou "NOT microscopy"
            context_before = algo_lower.split("microscop")[0][-30:]
            if "not" not in context_before and "não" not in context_before:
                results["scores"]["macro_micro"] = "❌"
                results["details"].append("❌ Erro macro/micro: chamou placas de Petri de microscopia")
            else:
                results["scores"]["macro_micro"] = "✅"
                results["details"].append("✅ Corretamente distinguiu placas de microscopia")
        else:
            results["scores"]["macro_micro"] = "✅"
            results["details"].append("✅ Corretamente identificou como fotografia macroscópica")

    # Overall score
    score_values = {"✅": 3, "⚠️": 1, "❌": 0, "—": 0}
    score_max = {"✅": 3, "⚠️": 3, "❌": 3, "—": 0}
    total = sum(score_values.get(v, 0) for v in results["scores"].values())
    maximum = sum(score_max.get(v, 0) for v in results["scores"].values())
    pct = total / maximum if maximum > 0 else 0

    if pct >= 0.7:
        results["overall"] = "✅"
    elif pct >= 0.4:
        results["overall"] = "⚠️"
    else:
        results["overall"] = "❌"

    results["score_pct"] = round(pct * 100)

    return results


def generate_v2_descriptions():
    """Regenera as descrições v2 do algoritmo."""
    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)

    # Carregar contexto do documento
    print("=== Carregando contexto do documento ===")
    loader = PyPDFLoader(pdf_path)
    text_docs = loader.load()
    full_text = " ".join([doc.page_content for doc in text_docs])
    species_list = extract_species_from_text(full_text)
    species_str = ", ".join(species_list)
    doc_title = PDF_FILENAME.replace("_", " ").replace(".pdf", "")
    print(f"  Espécies: {species_str}")

    # Extrair imagens
    print("\n=== Extraindo imagens ===")
    images = extract_images_from_pdf(pdf_path, EXTRACTED_IMAGES_PATH)
    if not images:
        print("Nenhuma imagem encontrada.")
        return

    # OCR
    print("\n=== OCR Pre-Pass ===")
    ocr_results = extract_text_batch(images)
    for img, ocr in zip(images, ocr_results):
        img["ocr_text"] = ocr

    # Textualizar + validar
    print("\n=== Textualização + Validação ===")
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

        print(f"\n  [{i}/{len(images)}] {filename} (type: {image_type})...")

        prompt = build_prompt(
            image_type=image_type,
            ocr_text=ocr_text,
            species_list=species_str,
            doc_title=doc_title,
        )

        try:
            description = textualize_image(image_path, prompt=prompt)
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

        except Exception as e:
            output_lines.append(f"## {filename} (Página {page} — {width}×{height}px)\n")
            output_lines.append(f"**ERRO:** {e}\n")
            output_lines.append("---\n")
            print(f"  ✗ Erro: {e}")

    with open(OUTPUT_ALGO, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\n✓ Descrições v2 salvas em: {OUTPUT_ALGO}")


def generate_comparison_report():
    """Gera o relatório comparativo final."""
    algo_descs = load_algo_descriptions(OUTPUT_ALGO)

    lines = []
    lines.append("# 📊 Relatório Comparativo Final — Pipeline v2\n")
    lines.append(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("**Metodologia:** Comparação das descrições geradas pelo pipeline v2")
    lines.append("(OCR + classificação + prompts contextualizados + validação)")
    lines.append("contra descrições manuais (ground truth).\n")
    lines.append("---\n")

    all_results = []

    for image_name, gt_info in GROUND_TRUTH.items():
        algo_text = algo_descs.get(image_name, "")
        if not algo_text:
            lines.append(f"## {image_name}\n")
            lines.append("**⚠️ Descrição do algoritmo não encontrada.**\n")
            lines.append("---\n")
            continue

        result = evaluate_image(image_name, algo_text, gt_info)
        all_results.append(result)

        lines.append(f"## {image_name}\n")
        lines.append(f"**Ground Truth — Tipo:** {gt_info['type']}\n")
        lines.append(f"**Ground Truth — Layout:** {gt_info['layout']}\n")
        lines.append(f"**Avaliação Geral:** {result['overall']} ({result['score_pct']}%)\n")

        # Tabela de scores
        lines.append("\n| Critério | Resultado |")
        lines.append("|----------|:---------:|")
        for criterion, score in result["scores"].items():
            criterion_name = criterion.replace("_", " ").title()
            lines.append(f"| {criterion_name} | {score} |")

        lines.append("\n**Detalhes:**\n")
        for detail in result["details"]:
            lines.append(f"- {detail}")

        lines.append("\n---\n")

    # Resumo final
    lines.append("## 📋 Resumo Final\n")

    # Tabela resumo
    lines.append("| Imagem | Tipo | Rótulos | Espécies | Alucinações | Fatos | Overall |")
    lines.append("|--------|:----:|:-------:|:--------:|:-----------:|:-----:|:-------:|")
    for r in all_results:
        s = r["scores"]
        lines.append(
            f"| {r['image']} | {s.get('tipo_imagem','—')} | {s.get('rotulos','—')} | "
            f"{s.get('especies','—')} | {s.get('alucinacoes','—')} | "
            f"{s.get('fatos','—')} | **{r['overall']} {r['score_pct']}%** |"
        )

    # Estatísticas
    total_images = len(all_results)
    correct = sum(1 for r in all_results if r["overall"] == "✅")
    partial = sum(1 for r in all_results if r["overall"] == "⚠️")
    incorrect = sum(1 for r in all_results if r["overall"] == "❌")
    avg_pct = sum(r["score_pct"] for r in all_results) / total_images if total_images else 0

    lines.append(f"\n### Estatísticas Gerais\n")
    lines.append(f"- **Total de imagens:** {total_images}")
    lines.append(f"- **✅ Correto:** {correct}/{total_images}")
    lines.append(f"- **⚠️ Parcial:** {partial}/{total_images}")
    lines.append(f"- **❌ Incorreto:** {incorrect}/{total_images}")
    lines.append(f"- **Score médio:** {avg_pct:.0f}%\n")

    # Conclusão
    lines.append("### Conclusão\n")
    if avg_pct >= 70:
        lines.append("O pipeline v2 demonstra **qualidade satisfatória** na textualização multimodal. ")
        lines.append("As estratégias de OCR pre-pass, classificação, prompts contextualizados e ")
        lines.append("validação anti-alucinação contribuíram para uma melhoria significativa.\n")
    elif avg_pct >= 50:
        lines.append("O pipeline v2 apresenta **qualidade moderada**. Há melhorias significativas ")
        lines.append("em relação à v1, mas ainda existem áreas que necessitam de ajustes.\n")
    else:
        lines.append("O pipeline v2 ainda apresenta **problemas de qualidade significativos**. ")
        lines.append("São necessárias melhorias adicionais nas estratégias implementadas.\n")

    with open(OUTPUT_COMPARISON, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n{'='*60}")
    print(f"  RELATÓRIO FINAL GERADO: {OUTPUT_COMPARISON}")
    print(f"  Score médio: {avg_pct:.0f}%")
    print(f"  ✅ {correct} | ⚠️ {partial} | ❌ {incorrect}")
    print(f"{'='*60}\n")


def main():
    print("=" * 60)
    print("  TESTE FINAL — Pipeline v2 de Textualização Multimodal")
    print("=" * 60)

    # Verificar se descrições manuais existem
    if not os.path.exists(MANUAL_FILE):
        print(f"\n❌ Arquivo de descrições manuais não encontrado: {MANUAL_FILE}")
        print("   Execute primeiro a Etapa 1 do procedimento de teste.")
        return

    print(f"\n✓ Descrições manuais: {MANUAL_FILE}")

    # Etapa 2: Gerar descrições v2
    print(f"\n{'='*60}")
    print("  ETAPA 2: Gerando descrições do algoritmo v2...")
    print(f"{'='*60}")
    generate_v2_descriptions()

    # Etapa 3: Comparação
    print(f"\n{'='*60}")
    print("  ETAPA 3: Comparando algoritmo vs ground truth...")
    print(f"{'='*60}")
    generate_comparison_report()

    print("\n✓ TESTE FINAL COMPLETO!")
    print(f"  → Etapa 1: {MANUAL_FILE}")
    print(f"  → Etapa 2: {OUTPUT_ALGO}")
    print(f"  → Etapa 3: {OUTPUT_COMPARISON}")


if __name__ == "__main__":
    main()
