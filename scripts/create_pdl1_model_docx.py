"""Generate an editable Korean DOCX description of the PD-L1 hierarchy model."""

from __future__ import annotations

import base64
import html
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


OUTPUT_DIR = Path("documents")
OUTPUT_DOCX = OUTPUT_DIR / "PDL1_hierarchical_YOLOv11m_model_description.docx"
FIGURE = Path("figures/pdl1_hierarchical_architecture_paper_v2.png")


def escape(text):
    return html.escape(str(text), quote=False)


def table(headers, rows, widths=None):
    widths = widths or [1] * len(headers)
    colgroup = "".join(f'<col style="width:{width}%">' for width in widths)
    head = "".join(f"<th>{escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<table><colgroup>{colgroup}</colgroup><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def build_html(figure_uri):
    class_rows = [
        (0, "Negative Epithelial", "Epithelial", "Negative", "–"),
        (1, "Negative Lymphocyte", "Immune", "Negative", "Lymphocyte"),
        (2, "Negative Macrophage", "Immune", "Negative", "Macrophage"),
        (3, "Positive Epithelial", "Epithelial", "Positive", "–"),
        (4, "Positive Lymphocyte", "Immune", "Positive", "Lymphocyte"),
        (5, "Positive Macrophage", "Immune", "Positive", "Macrophage"),
        (6, "other", "other", "–", "–"),
    ]
    metric_rows = [
        ("Detection recall", "0.9608"),
        ("Classification accuracy (matched cells)", "0.5879"),
        ("Macro precision / recall / F1", "0.2521 / 0.2709 / 0.2290"),
        ("Overall class recall", "0.5649"),
        ("Root recall: Epithelial / Immune / other", "0.6156 / 0.3340 / 0.6650"),
        ("Root balanced accuracy", "0.5382"),
        ("Epithelial polarity accuracy", "0.8019"),
        ("Immune type accuracy", "0.7660"),
        ("Immune polarity accuracy", "0.8713"),
        ("Validation score", "0.4218"),
    ]
    class_metric_rows = [
        ("Negative Epithelial", 5805, 0.3798, 0.7027, 0.4930),
        ("Negative Lymphocyte", 168, 0.2292, 0.0655, 0.1019),
        ("Negative Macrophage", 279, 0.0758, 0.0358, 0.0487),
        ("Positive Epithelial", 2814, 0.3754, 0.0867, 0.1409),
        ("Positive Lymphocyte", 214, 0.0000, 0.0000, 0.0000),
        ("Positive Macrophage", 1510, 0.1812, 0.3629, 0.2417),
        ("other", 15399, 0.5236, 0.6430, 0.5772),
    ]
    loss_rows = [
        ("Box", "모든 foreground anchor", "IoU 기반 bbox 회귀", "5.0"),
        ("DFL", "모든 foreground anchor", "분포 기반 경계 회귀", "0.5"),
        ("Objectness", "모든 anchor", "cell / background", "1.0"),
        ("Root CE", "모든 matched cell", "Epithelial / Immune / other", "2.0"),
        ("Epithelial polarity BCE", "GT {0, 3}만", "Negative / Positive", "1.0"),
        ("Immune type BCE", "GT {1, 2, 4, 5}만", "Lymphocyte / Macrophage", "2.0"),
        ("Immune polarity BCE", "GT {1, 2, 4, 5}만", "Negative / Positive", "2.0"),
    ]

    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><title>PD-L1 Hierarchical YOLOv11-m</title>
<style>
@page {{ size: A4; margin: 20mm 18mm 18mm 18mm; }}
body {{ font-family: 'Noto Sans CJK KR', 'Malgun Gothic', sans-serif; color:#172033; font-size:10.5pt; line-height:1.55; }}
h1 {{ color:#0B3A67; font-size:22pt; margin:0 0 4pt 0; }}
h2 {{ color:#0B3A67; font-size:15pt; border-bottom:1.5px solid #2A78B8; padding-bottom:3pt; margin-top:18pt; page-break-after:avoid; }}
h3 {{ color:#245A82; font-size:12pt; margin:13pt 0 4pt 0; page-break-after:avoid; }}
p {{ margin:4pt 0 7pt 0; text-align:justify; }}
.subtitle {{ color:#56667A; font-size:11pt; margin-bottom:14pt; }}
.summary {{ background:#EFF6FB; border-left:5px solid #2A78B8; padding:10pt 12pt; margin:10pt 0 14pt; }}
.warning {{ background:#FFF4E6; border-left:5px solid #E67E22; padding:9pt 12pt; margin:10pt 0; }}
.equation {{ font-family:'Cambria Math','Times New Roman',serif; text-align:center; background:#F8FAFC; border:1px solid #CBD5E1; padding:7pt; margin:5pt 15mm; }}
.caption {{ font-size:9pt; color:#475569; text-align:left; margin:4pt 0 12pt; }}
.figure {{ width:100%; margin-top:5pt; }}
table {{ border-collapse:collapse; width:100%; margin:7pt 0 12pt; font-size:9pt; page-break-inside:avoid; }}
th {{ background:#0B3A67; color:white; padding:5pt; border:1px solid #BCC8D4; text-align:center; }}
td {{ padding:4pt 5pt; border:1px solid #BCC8D4; vertical-align:top; }}
tr:nth-child(even) td {{ background:#F5F8FA; }}
ul {{ margin-top:3pt; }} li {{ margin-bottom:3pt; }}
.small {{ font-size:9pt; color:#475569; }}
.first-section {{ page-break-before:always; break-before:page; }}
</style></head><body>
<h1>PD-L1 계층형 YOLOv11-m 모델 설명</h1>
<p class="subtitle">Epithelial–Immune–other 계층적 의미 분해를 이용한 세포 검출 및 7-class 분류</p>

<p class="summary"><b>모델 요약.</b> 본 모델은 YOLOv11-m 기반 다중 스케일 검출기에 PD-L1 세포의 생물학적 관계를 반영한 계층형 semantic head를 결합한다. 모든 세포를 직접 7개 클래스로 분류하는 대신, 먼저 Epithelial·Immune·other의 root 확률을 추정하고, Epithelial에는 polarity를, Immune에는 type과 polarity를 조건부로 학습한다. 추론 시에는 hard routing을 사용하지 않고 각 확률을 곱하여 7-class 점수를 생성한다.</p>

<h2 class="first-section">1. 문제 정의와 클래스 계층</h2>
<p>입력 영상에서 세포의 위치를 검출하는 동시에 다음 7개 클래스를 예측한다. JSON의 <i>class_name</i>이 아래 명칭과 일치하는 annotation만 사용하며, 그 외 라벨은 제외한다.</p>
{table(["ID", "최종 클래스", "Root", "Polarity", "Immune type"], class_rows, [7, 35, 20, 18, 20])}
<p>계층 구조는 다음과 같다.</p>
<ul>
<li><b>Root:</b> Epithelial / Immune / other의 3-way softmax.</li>
<li><b>Epithelial branch:</b> Epithelial GT에 대해서만 Negative / Positive polarity를 학습한다.</li>
<li><b>Immune branch:</b> Immune GT에 대해서만 Lymphocyte / Macrophage type과 Negative / Positive polarity를 병렬로 학습한다.</li>
<li><b>other:</b> 추가 하위 branch가 없는 terminal class이다.</li>
</ul>

<h2>2. 전체 아키텍처</h2>
<img class="figure" src="{figure_uri}">
<p class="caption"><b>Figure 1.</b> 제안한 PD-L1 hierarchical YOLOv11-m 구조. 상단은 다중 스케일 특징 추출, 계층형 semantic factorization, 조건부 supervision mask와 soft probability fusion을 나타낸다. 하단은 실제 validation 512×512 tile, 123개 GT annotation 및 best checkpoint(epoch 48)의 실제 예측 197개(confidence ≥ 0.10)를 나타낸다.</p>

<h3>2.1 입력과 데이터 파이프라인</h3>
<p>원본 patch는 RGB 1024×1024로 변환하여 uint8 형태로 RAM에 캐시한다. 학습 시 각 접근마다 1024 영상에서 512×512 영역을 새로 random crop하며 flip, 90° rotation, 밝기·HSV 변화 및 제한적 Gaussian blur를 적용한다. validation에서는 1024 patch를 겹치지 않는 네 개의 512×512 사분면으로 고정 분할한다. 따라서 validation 원본 한 장은 정확히 네 개의 평가 sample이 된다.</p>
<p>작은 데이터셋의 유효 step 수를 늘리기 위해 동일 train dataset을 outer epoch마다 50회 사용한다. 캐시는 동일 객체를 공유하므로 영상을 50배 복제하지 않으며, 매 접근에서 crop과 augmentation이 다시 표본화된다.</p>

<h3>2.2 YOLOv11-m Backbone과 PAN-FPN</h3>
<p>Backbone은 Conv, CSP, SPP 및 PSA 모듈로 구성된다. 512×512 입력에 대해 stride 8, 16, 32의 P3, P4, P5 feature map을 생성한다. PAN-FPN은 upsampling·concatenation 기반 top-down path와 stride-2 convolution 기반 bottom-up path를 이용하여 공간 해상도와 고수준 의미 정보를 양방향으로 결합한다.</p>
{table(["Scale", "Stride", "Feature shape (C×H×W)", "주요 역할"], [
    ("P3", "8", "256×64×64", "작은 세포의 위치 및 세부 형태"),
    ("P4", "16", "512×32×32", "중간 수준 형태와 문맥"),
    ("P5", "32", "512×16×16", "고수준 의미 및 넓은 문맥"),
], [12, 12, 30, 46])}

<h3>2.3 Detection head</h3>
<p>각 scale의 fused feature는 box regression tower와 shared semantic tower로 전달된다. Box tower는 16-bin distribution을 예측하고 DFL decoder를 통해 중심 좌표와 폭·높이(xywh)를 계산한다. Semantic tower는 depthwise 및 pointwise convolution을 공유한 뒤 objectness와 계층별 logit을 출력한다. 전체 모델은 20,058,405개 parameter(학습 가능 20,058,389개)를 갖는다.</p>

<h2>3. 계층형 확률 분해</h2>
<p>Objectness logit에는 sigmoid를, root logit에는 3-way softmax를 적용한다. Epithelial polarity, Immune type 및 Immune polarity는 각각 binary sigmoid로 표현된다. 최종 class score는 아래와 같이 soft multiplication으로 구성한다.</p>
<div class="equation">P(Negative Epithelial) = P(cell) · P(Epithelial) · P(Negative | Epithelial)</div>
<div class="equation">P(Positive Epithelial) = P(cell) · P(Epithelial) · P(Positive | Epithelial)</div>
<div class="equation">P(Immune class) = P(cell) · P(Immune) · P(type | Immune) · P(polarity | Immune)</div>
<div class="equation">P(other) = P(cell) · P(other)</div>
<p>Immune branch에서는 type과 polarity의 Cartesian product로 네 클래스를 생성한다. 예를 들어 Negative Lymphocyte는 <i>P(cell)·P(Immune)·P(Lymphocyte|Immune)·P(Negative|Immune)</i>이다.</p>
<div class="warning"><b>Hard routing과의 차이.</b> Root argmax 결과로 하위 branch를 제거하지 않는다. 모든 branch 확률을 연속적으로 유지하고 곱하므로 미분 가능하며, root의 불확실성도 최종 class confidence에 반영된다. 다만 P(Immune)이 낮으면 네 immune class가 동시에 감쇠되므로 root recall이 최종 immune recall의 중요한 병목이 될 수 있다.</div>

<h2>4. 조건부 supervision과 손실함수</h2>
<p>조건부 mask는 <b>학습 loss에만</b> 사용된다. Root CE는 모든 matched cell에 적용하지만, Epithelial polarity loss는 Epithelial GT에만, Immune type·polarity loss는 Immune GT에만 적용한다. 따라서 관련 없는 branch가 다른 계열의 세포로부터 모순된 supervision을 받지 않는다.</p>
{table(["손실 항", "활성 대상", "학습 목표", "Gain"], loss_rows, [22, 25, 38, 15])}
<div class="equation">L<sub>total</sub> = L<sub>box</sub> + L<sub>obj</sub> + L<sub>root</sub> + L<sub>epi</sub> + L<sub>immune-type</sub> + L<sub>immune-polarity</sub> + L<sub>DFL</sub></div>
<p>Root class weight는 train label 분포의 square-root inverse frequency로 계산했으며 Epithelial, Immune, other에 각각 1.481, 2.860, 1.000을 적용했다. Epithelial positive weight는 2.184, Immune lymphocyte weight는 2.164, Immune negative weight는 1.997이다. Objectness에는 focal modulation(γ=1.5)을 사용한다.</p>

<h2>5. 학습 및 추론 설정</h2>
{table(["항목", "설정"], [
    ("입력", "1024 RAM cache → 512 random crop"),
    ("Validation", "1024 patch당 고정 512 tile 4개"),
    ("Model", "YOLOv11-m + PD-L1 hierarchical head"),
    ("Optimizer", "AdamW, initial LR 5×10⁻⁴, weight decay 5×10⁻⁴"),
    ("LR scheduler", "CosineAnnealingLR, minimum LR 1×10⁻⁶"),
    ("Epoch", "최대 2,000; early stopping 미사용"),
    ("Dataset passes", "outer epoch당 50회"),
    ("Effective batch", "batch 16, gradient accumulation 4 → effective batch 64"),
    ("Validation confidence", "0.10"),
    ("NMS", "class-agnostic NMS"),
    ("Matching", "중심점 거리 ≤16 px, Hungarian assignment"),
], [31, 69])}

<h2>6. Best checkpoint 결과</h2>
<p>저장된 best checkpoint는 epoch 48(epoch index 47)에서 선택되었다. 선택 점수는 0.65×Macro F1 + 0.20×Detection recall + 0.15×Root balanced accuracy이다.</p>
{table(["Metric", "Value"], metric_rows, [65, 35])}

<h3>6.1 클래스별 성능</h3>
{table(["Class", "GT", "Precision", "Recall", "F1"], class_metric_rows, [36, 12, 18, 17, 17])}

<h3>6.2 결과 해석</h3>
<p>Detection recall은 0.9608로 위치 검출 자체는 높았지만, 최종 Macro F1은 0.2290이었다. 특히 Immune root recall이 0.3340으로 낮아 immune class 확률 네 개가 동시에 감소했다. Branch accuracy는 Immune type 0.7660, polarity 0.8713으로 상대적으로 높지만, 이 값은 검출·매칭된 Immune GT의 해당 branch logit을 평가한 조건부 성능이므로 낮은 root recall을 보상하지 못한다.</p>
<p>따라서 본 실험에서 주요 병목은 하위 Immune type/polarity 분류보다 Epithelial·Immune·other root 분류이다. 실제 train root label은 Epithelial 14,406, Immune 3,860, other 31,584로 불균형하며, class-aware crop을 사용하더라도 crop 내부 세포 수 기준 Immune 비율은 낮게 유지될 수 있다.</p>

<h2>7. 논문 기재 시 권장 설명</h2>
<p>“The semantic head factorizes the seven-class posterior into objectness, a three-way epithelial–immune–other root distribution, epithelial polarity, immune type, and immune polarity. Branch-specific losses are conditionally masked according to the ground-truth lineage, whereas inference retains all factors and forms final class scores by soft probability multiplication rather than hard routing.”</p>

<h2>8. 재현 정보</h2>
<ul>
<li>학습 노트북: <code>Precise_PDL1_ST_hierarchical_train.ipynb</code></li>
<li>모델 정의: <code>nets/nn.py</code>의 <code>PDL1HierarchicalYOLO</code> 및 <code>PDL1HierarchicalHead</code></li>
<li>손실함수: <code>utils/hierarchical.py</code>의 <code>PDL1HierarchicalComputeLoss</code></li>
<li>Best checkpoint: <code>pdl1_hierarchical_eio_yolov11/best_model.pt</code></li>
</ul>
<p class="small">본 문서의 성능 수치는 현재 workspace에 저장된 best checkpoint 및 해당 validation split을 기준으로 작성되었다.</p>
</body></html>"""


def fix_docx_figure_extent(docx_path):
    """Set the embedded architecture figure to the A4 text width."""
    width_emu = 6_260_000
    height_emu = round(width_emu * 2400 / 3072)
    temporary_docx = docx_path.with_suffix(".layout-fixed.docx")
    with zipfile.ZipFile(docx_path, "r") as source_zip:
        entries = {
            item.filename: source_zip.read(item.filename)
            for item in source_zip.infolist()
        }
    document_xml = entries["word/document.xml"].decode("utf-8")
    document_xml, extent_count = re.subn(
        r'<wp:extent cx="\d+" cy="\d+"/>',
        f'<wp:extent cx="{width_emu}" cy="{height_emu}"/>',
        document_xml, count=1,
    )
    document_xml, transform_count = re.subn(
        r'<a:ext cx="\d+" cy="\d+"/>',
        f'<a:ext cx="{width_emu}" cy="{height_emu}"/>',
        document_xml, count=1,
    )
    if extent_count != 1 or transform_count != 1:
        raise RuntimeError("Could not locate the embedded figure extent")
    entries["word/document.xml"] = document_xml.encode("utf-8")
    with zipfile.ZipFile(
        temporary_docx, "w", compression=zipfile.ZIP_DEFLATED
    ) as destination_zip:
        for filename, data in entries.items():
            destination_zip.writestr(filename, data)
    temporary_docx.replace(docx_path)


def main():
    if not FIGURE.exists():
        raise FileNotFoundError(FIGURE)
    libreoffice = shutil.which("libreoffice")
    if libreoffice is None:
        raise RuntimeError("LibreOffice is required to convert the editable document")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pdl1_docx_") as temporary_directory:
        temporary_path = Path(temporary_directory)
        html_path = temporary_path / "PDL1_model_description.html"
        figure_data_uri = (
            "data:image/png;base64,"
            + base64.b64encode(FIGURE.read_bytes()).decode("ascii")
        )
        html_path.write_text(
            build_html(figure_data_uri), encoding="utf-8"
        )
        subprocess.run(
            [
                libreoffice, "--headless", "--convert-to",
                "docx:Office Open XML Text",
                "--outdir", str(temporary_path), str(html_path),
            ],
            check=True,
        )
        converted = temporary_path / "PDL1_model_description.docx"
        if not converted.exists():
            raise RuntimeError("LibreOffice did not create the expected DOCX")
        shutil.copy2(converted, OUTPUT_DOCX)
        fix_docx_figure_extent(OUTPUT_DOCX)
    print(OUTPUT_DOCX.resolve())


if __name__ == "__main__":
    main()
