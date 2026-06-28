from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "report" / "final_report_template.md"
OUTPUT_PDF = ROOT / "report" / "academic_report_with_figures.pdf"
FIGURE_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"

FIGURES = [
    ("roc_curve.png", "Sekil 1. ROC egrisi"),
    ("precision_recall_curve.png", "Sekil 2. Precision-Recall egrisi"),
    ("confusion_matrix.png", "Sekil 3. Confusion matrix"),
    ("feature_importance.png", "Sekil 4. Feature importance grafigi"),
    ("calibration_curve.png", "Sekil 5. Calibration curve"),
    ("model_comparison.png", "Sekil 6. Model karsilastirma grafigi"),
]

BONUS_FIGURES = [
    ("shap_summary.png", "Sekil 7. SHAP ozellik etkisi analizi"),
    ("lime_explanation.png", "Sekil 8. LIME lokal aciklanabilirlik analizi"),
    ("nested_cv_results.png", "Sekil 9. Nested cross-validation sonuclari"),
    ("feature_stability.png", "Sekil 10. Feature stability analizi"),
    ("ensemble_optimization.png", "Sekil 11. Ensemble agirlik optimizasyonu"),
    ("threshold_optimization.png", "Sekil 12. Threshold optimization sonuclari"),
]

FIGURE_COMMENTS = {
    "roc_curve.png": (
        "Yorum: ROC egrisi modellerin normal ve papilodem siniflarini genel olarak ayirma gucunu "
        "gostermektedir. Yuksek ROC-AUC degerleri, modellerin siniflari olasilik skorlarina gore "
        "basarili sekilde siralayabildigini desteklemektedir."
    ),
    "precision_recall_curve.png": (
        "Yorum: Precision-Recall egrisi, papilodem sinifinin normal sinifa gore daha az ornek "
        "icermesi nedeniyle ozellikle onemlidir. Yuksek PR-AUC degerleri, pozitif sinifin "
        "yakalanmasinda modellerin guclu performans gosterdigini ortaya koymaktadir."
    ),
    "confusion_matrix.png": (
        "Yorum: Confusion matrix, dogru ve hatali siniflandirmalarin sinif bazinda dagilimini "
        "gostermektedir. Diyagonal hucrelerdeki yogunluk modelin genel dogrulugunu, diyagonal "
        "disindaki hucreler ise yanlis siniflandirilan ornekleri temsil etmektedir."
    ),
    "feature_importance.png": (
        "Yorum: Feature importance grafigi, model kararlarinda en etkili radyomik ozellikleri "
        "siralamaktadir. Feature_0005 ozelliginin belirgin katkisi, sinif ayriminda bu ozelligin "
        "onemli bir ayirt edici bilgi tasidigini gostermektedir."
    ),
    "calibration_curve.png": (
        "Yorum: Calibration curve, model olasilik tahminlerinin gercek pozitiflik oranlariyla "
        "ne kadar uyumlu oldugunu degerlendirmektedir. Egrilerin ideal dogruya yakin seyretmesi "
        "ve dusuk Brier Score degerleri, kalibrasyon sonrasi olasilik tahminlerinin makul "
        "duzeyde guvenilir oldugunu gostermektedir."
    ),
    "model_comparison.png": (
        "Yorum: Model karsilastirma grafigi, farkli algoritmalarin test setindeki performansini "
        "bir arada sunmaktadir. Genel siniflandirma basarisi acisindan KNN modeli one cikarken, "
        "RF ve ensemble modelleri olasilik bazli ayirt edicilik metriklerinde guclu sonuclar vermistir."
    ),
    "shap_summary.png": (
        "Yorum: SHAP analizi, model tahminlerine en fazla katkida bulunan ozellikleri aciklamaktadir. "
        "Bu analiz, feature importance sonuclariyla birlikte yorumlandiginda model kararlarinin "
        "hangi radyomik degiskenlere dayandigini daha anlasilir hale getirmektedir."
    ),
    "lime_explanation.png": (
        "Yorum: LIME analizi, secilen tek bir test ornegi icin model kararini lokal olarak "
        "aciklamaktadir. Bu grafik, ilgili ornekte hangi ozellik araliklarinin papilodem veya "
        "normal sinif tahminini destekledigini gostererek model yorumlanabilirligini artirmaktadir."
    ),
    "nested_cv_results.png": (
        "Yorum: Nested cross-validation sonuclari, model secimi ve performans tahmininin daha "
        "tarafsiz degerlendirilmesini saglamaktadir. Dis fold sonuclarinin birlikte raporlanmasi, "
        "model performansinin farkli hasta gruplari uzerindeki kararliligini incelemeye yardim eder."
    ),
    "feature_stability.png": (
        "Yorum: Feature stability analizi, farkli cross-validation foldlarinda secilen ozelliklerin "
        "ne kadar tutarli oldugunu gostermektedir. Yuksek stabiliteye sahip ozellikler, modelin "
        "yalnizca tek bir bolunmeye bagli olmayan daha guvenilir sinyaller kullandigini destekler."
    ),
    "ensemble_optimization.png": (
        "Yorum: Ensemble optimizasyonu, RF, ET ve GB modellerinin soft voting yapisindaki "
        "agirliklarini validation performansina gore ayarlamaktadir. Bu yaklasim, ensemble "
        "modelin tum uyeleri esit kabul etmek yerine daha etkili modellerden daha fazla "
        "yararlanmasini saglamaktadir."
    ),
    "threshold_optimization.png": (
        "Yorum: Threshold optimization analizi, varsayilan 0.50 karar esigi yerine validation "
        "setinde daha uygun esiklerin denenmesini saglamaktadir. Bu islem, precision ve recall "
        "arasindaki dengeyi klinik onceliklere gore ayarlamak icin kullanilabilir."
    ),
}

SKIP_LINES = {
    "Bu bolume `outputs/tables/model_performance_test.csv` tablosundaki sonuclari ekle.",
    "Bu bolume outputs/tables/model_performance_test.csv tablosundaki sonuclari ekle.",
    "Tablo basliklari:",
    "Bu bolume asagidaki gorselleri ekle:",
    "Friedman testi, Wilcoxon signed-rank testi ve Bonferroni duzeltmesi sonuclari `outputs/tables/statistical_tests.csv` dosyasindan rapora eklenmelidir.",
    "Friedman testi, Wilcoxon signed-rank testi ve Bonferroni duzeltmesi sonuclari outputs/tables/statistical_tests.csv dosyasindan rapora eklenmelidir.",
    "Bu bolumde su sorular cevaplanmalidir:",
}

SKIP_PREFIXES = (
    "- `outputs/figures/",
    "- outputs/figures/",
    "- Accuracy",
    "- Precision",
    "- Recall",
    "- F1-score",
    "- Macro-F1",
    "- ROC-AUC",
    "- PR-AUC",
    "- Balanced Accuracy",
    "- Brier Score",
)


def register_fonts() -> tuple[str, str]:
    candidates = [
        (
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ),
        (
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        ),
    ]
    for regular, bold in candidates:
        if Path(regular).exists() and Path(bold).exists():
            pdfmetrics.registerFont(TTFont("ReportFont", regular))
            pdfmetrics.registerFont(TTFont("ReportFont-Bold", bold))
            return "ReportFont", "ReportFont-Bold"
    return "Helvetica", "Helvetica-Bold"


def build_styles():
    base_font, bold_font = register_fonts()
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName=bold_font,
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=18,
            textColor=colors.HexColor("#1F3A5F"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportH1",
            parent=styles["Heading1"],
            fontName=bold_font,
            fontSize=14,
            leading=17,
            spaceBefore=14,
            spaceAfter=8,
            textColor=colors.HexColor("#2E74B5"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=10.5,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBullet",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=9.8,
            leading=12.5,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontName=bold_font,
            fontSize=10,
            leading=12,
            alignment=TA_CENTER,
            spaceBefore=6,
            spaceAfter=6,
            textColor=colors.HexColor("#1F3A5F"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="FigureComment",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=9.3,
            leading=12,
            alignment=TA_JUSTIFY,
            leftIndent=0.35 * cm,
            rightIndent=0.35 * cm,
            spaceBefore=6,
            spaceAfter=10,
            textColor=colors.HexColor("#333333"),
        )
    )
    return styles


def clean_line(line: str) -> str:
    line = line.strip()
    line = line.replace("`", "")
    line = re.sub(r"^(cevap|yorum)\s*:\s*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"\s+", " ", line)
    return line


def parse_sections(markdown_path: Path) -> tuple[str, list[tuple[str, list[str]]]]:
    title = ""
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_lines: list[str] = []
    previous_line = None

    for raw in markdown_path.read_text(encoding="utf-8").splitlines():
        line = clean_line(raw)
        if not line:
            continue

        if line.startswith("# "):
            title = line[2:].strip()
            continue

        if line.startswith("## "):
            if current_title:
                sections.append((current_title, current_lines))
            current_title = line[3:].strip()
            current_lines = []
            previous_line = None
            continue

        if line in SKIP_LINES or line.startswith(SKIP_PREFIXES):
            continue
        if line == previous_line:
            continue

        current_lines.append(line)
        previous_line = line

    if current_title:
        sections.append((current_title, current_lines))
    return title, sections


def paragraph(text: str, styles, style_name: str = "ReportBody") -> Paragraph:
    return Paragraph(html.escape(text), styles[style_name])


def add_lines(story: list, lines: list[str], styles) -> None:
    bullet_buffer: list[str] = []

    def flush_bullets() -> None:
        if not bullet_buffer:
            return
        items = [ListItem(paragraph(item, styles, "ReportBullet")) for item in bullet_buffer]
        story.append(ListFlowable(items, bulletType="bullet", leftIndent=18, bulletFontSize=8))
        story.append(Spacer(1, 4))
        bullet_buffer.clear()

    for line in lines:
        if is_generated_table_duplicate(line):
            continue
        if line.startswith("- "):
            bullet_buffer.append(line[2:].strip())
            continue
        flush_bullets()

        if re.match(r"^\d+\.\s+", line):
            story.append(paragraph(line, styles, "ReportBody"))
        elif looks_like_plain_table_line(line):
            story.append(Paragraph(f"<font name='Courier'>{html.escape(line)}</font>", styles["ReportBullet"]))
        else:
            story.append(paragraph(line, styles, "ReportBody"))

    flush_bullets()


def looks_like_plain_table_line(line: str) -> bool:
    return bool(re.search(r"\s{2,}", line)) and (
        line.startswith(("Model", "LR", "SVM", "RF", "ET", "GB", "KNN", "MLP", "Ensemble", "WeightedEnsemble", "Feature_", "Ozellik", "Özellik"))
    )


def is_generated_table_duplicate(line: str) -> bool:
    if line.startswith(("Model Accuracy", "Ozellik Importance", "Özellik Importance")):
        return True
    if re.match(r"^(LR|SVM|RF|ET|GB|KNN|MLP|Ensemble|WeightedEnsemble)\s+\d", line):
        return True
    if re.match(r"^Feature_\d{4}\s+\d", line):
        return True
    return False


def dataframe_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> Table:
    if columns:
        present_columns = [column for column in columns if column in df.columns]
        df = df[present_columns] if present_columns else df
    if max_rows:
        df = df.head(max_rows)
    df = df.copy()
    if df.empty and len(df.columns) == 0:
        df = pd.DataFrame({"status": ["no_output"]})
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    data = [list(df.columns)] + df.astype(str).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5")),
                ("FONTNAME", (0, 0), (-1, 0), "ReportFont-Bold" if "ReportFont-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B8B8B8")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def add_result_tables(story: list, styles) -> None:
    performance = TABLE_DIR / "model_performance_test.csv"
    if performance.exists():
        df = pd.read_csv(performance)
        story.append(Spacer(1, 6))
        story.append(paragraph("Tablo 1. Final test seti model performanslari", styles, "Caption"))
        story.append(
            dataframe_table(
                df,
                columns=["model", "accuracy", "macro_f1", "roc_auc", "pr_auc", "brier_score"],
            )
        )
        story.append(Spacer(1, 10))


def add_feature_table(story: list, styles) -> None:
    features = TABLE_DIR / "top_features.csv"
    if features.exists():
        df = pd.read_csv(features)
        story.append(Spacer(1, 6))
        story.append(paragraph("Tablo 2. En onemli 10 radyomik ozellik", styles, "Caption"))
        story.append(dataframe_table(df, columns=["feature", "importance"], max_rows=10))
        story.append(Spacer(1, 10))


def add_statistics_table(story: list, styles) -> None:
    stats = TABLE_DIR / "statistical_tests.csv"
    if stats.exists():
        df = pd.read_csv(stats)
        story.append(Spacer(1, 6))
        story.append(paragraph("Tablo 3. Istatistiksel test sonuclari", styles, "Caption"))
        story.append(dataframe_table(df))
        story.append(Spacer(1, 10))


def add_bonus_tables(story: list, styles) -> None:
    bonus_tables = [
        (
            "deep_learning_mlp_results.csv",
            "Tablo 4. Deep learning MLP model sonucu",
            ["model", "accuracy", "macro_f1", "roc_auc", "pr_auc", "brier_score"],
            None,
        ),
        (
            "shap_summary.csv",
            "Tablo 5. SHAP analizinde en etkili ozellikler",
            ["feature", "mean_abs_shap"],
            10,
        ),
        (
            "lime_explanation.csv",
            "Tablo 6. LIME lokal aciklama agirliklari",
            ["model", "test_index", "feature_rule", "lime_weight"],
            10,
        ),
        (
            "nested_cv_results.csv",
            "Tablo 7. Nested cross-validation dis fold sonuclari",
            ["fold", "selected_model", "inner_best_macro_f1", "outer_macro_f1", "outer_roc_auc", "outer_pr_auc"],
            None,
        ),
        (
            "feature_stability.csv",
            "Tablo 8. Feature stability analizinde en stabil ozellikler",
            ["feature", "selection_count", "n_folds", "stability_percent"],
            10,
        ),
        (
            "ensemble_optimization.csv",
            "Tablo 9. Ensemble agirlik optimizasyonu",
            ["rf_weight", "et_weight", "gb_weight", "validation_macro_f1", "test_macro_f1", "test_roc_auc", "test_pr_auc"],
            None,
        ),
        (
            "threshold_optimization.csv",
            "Tablo 10. Threshold optimization sonuclari",
            ["model", "best_threshold", "validation_macro_f1", "test_macro_f1", "test_recall", "test_precision"],
            10,
        ),
    ]
    for file_name, caption, columns, max_rows in bonus_tables:
        table_path = TABLE_DIR / file_name
        if not table_path.exists():
            continue
        df = pd.read_csv(table_path)
        story.append(
            KeepTogether(
                [
                    Spacer(1, 6),
                    paragraph(caption, styles, "Caption"),
                    dataframe_table(df, columns=columns, max_rows=max_rows),
                    Spacer(1, 10),
                ]
            )
        )


def add_image_block(story: list, styles, file_name: str, caption: str) -> None:
    image_path = FIGURE_DIR / file_name
    if not image_path.exists():
        story.append(paragraph(f"Eksik gorsel: {image_path}", styles))
        return

    img = Image(str(image_path))
    max_width = 16.0 * cm
    max_height = 9.2 * cm
    scale = min(max_width / img.imageWidth, max_height / img.imageHeight)
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    block = [paragraph(caption, styles, "Caption"), img]
    comment = FIGURE_COMMENTS.get(file_name)
    if comment:
        block.append(paragraph(comment, styles, "FigureComment"))
    block.append(Spacer(1, 10))
    story.append(KeepTogether(block))


def add_figures(story: list, styles) -> None:
    story.append(paragraph("Bu bolumde final pipeline tarafindan uretilen zorunlu grafikler sirali olarak sunulmustur.", styles))
    for file_name, caption in FIGURES:
        add_image_block(story, styles, file_name, caption)


def add_bonus_figures(story: list, styles) -> None:
    for file_name, caption in BONUS_FIGURES:
        add_image_block(story, styles, file_name, caption)


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(A4[0] - 2 * cm, 1.3 * cm, f"Sayfa {doc.page}")
    canvas.restoreState()


def build_pdf() -> Path:
    title, sections = parse_sections(REPORT_MD)
    styles = build_styles()
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.55 * cm,
    )
    story: list = [paragraph(title, styles, "ReportTitle")]

    for section_title, lines in sections:
        story.append(Paragraph(html.escape(section_title), styles["ReportH1"]))
        if section_title.startswith("6."):
            add_lines(story, lines, styles)
            add_result_tables(story, styles)
        elif section_title.startswith("7."):
            add_figures(story, styles)
        elif section_title.startswith("8."):
            add_lines(story, lines, styles)
            add_statistics_table(story, styles)
        elif section_title.startswith("9."):
            add_lines(story, lines, styles)
            add_feature_table(story, styles)
        else:
            add_lines(story, lines, styles)

    bonus_outputs = [TABLE_DIR / name for name in [
        "deep_learning_mlp_results.csv",
        "shap_summary.csv",
        "lime_explanation.csv",
        "nested_cv_results.csv",
        "feature_stability.csv",
        "ensemble_optimization.csv",
        "threshold_optimization.csv",
    ]]
    if any(path.exists() for path in bonus_outputs):
        story.append(PageBreak())
        story.append(Paragraph("12. Bonus Calismalar", styles["ReportH1"]))
        story.append(
            paragraph(
                "Bu bolumde odevde bonus olarak belirtilen SHAP, LIME, nested cross-validation, "
                "deep learning modeli, feature stability, ensemble optimizasyonu ve threshold optimization "
                "calismalarinin uretilen tablo ve gorsel sonuclari sunulmustur.",
                styles,
            )
        )
        add_bonus_tables(story, styles)
        add_bonus_figures(story, styles)

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    return OUTPUT_PDF


if __name__ == "__main__":
    output = build_pdf()
    print(f"PDF olusturuldu: {output}")
