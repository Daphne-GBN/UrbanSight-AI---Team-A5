"""
UrbanSight AI - PDF Inspection Report Generator (ReportLab)
Generates structured municipal inspection reports in PDF format.
Includes privacy processing details, YOLOP vehicle counts, and scene limitations.
"""

import io
from datetime import datetime
from html import escape
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_report_pdf(
    session_id: str,
    user_prompt: str,
    detection_summary: dict,
    faces_blurred: int,
    plates_blurred: int,
    description: str
) -> bytes:
    """Generates PDF report bytes using ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a")
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#64748b")
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155")
    )
    alert_style = ParagraphStyle(
        "PrivacyAlert",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#047857")
    )

    story = []

    # 1. Header
    story.append(Paragraph("UrbanSight AI — Street Scene Inspection Report", title_style))
    story.append(Spacer(1, 4))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(
        Paragraph(
            f"<b>Evaluation Session:</b> {session_id} &nbsp;|&nbsp; <b>Timestamp:</b> {timestamp} &nbsp;|&nbsp; <b>Team:</b> A_5 (PS 112)",
            subtitle_style
        )
    )
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#cbd5e1"), spaceAfter=12))

    # 2. Privacy processing statement
    story.append(Paragraph("1. Privacy Processing Status", heading_style))
    privacy_info = [
        [Paragraph("<b>Status:</b>", body_style), Paragraph("Best-effort blur; manual review recommended", alert_style)],
        [Paragraph("<b>Faces Blurred:</b>", body_style), Paragraph(f"{faces_blurred} (OpenCV Haar Cascade Anonymization)", body_style)],
        [Paragraph("<b>Plates Blurred:</b>", body_style), Paragraph(f"{plates_blurred} (Vehicle Number Plate Selective Blur)", body_style)],
        [Paragraph("<b>Scope:</b>", body_style), Paragraph("Detected regions were blurred before object detection; missed regions are possible.", body_style)]
    ]
    privacy_table = Table(privacy_info, colWidths=[140, 390])
    privacy_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#bbf7d0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dcfce7")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(privacy_table)
    story.append(Spacer(1, 12))

    # 3. YOLOP Detection Results
    story.append(Paragraph("2. YOLOP Panoptic Object Detection", heading_style))
    det_rows = [[Paragraph("<b>Object Category</b>", body_style), Paragraph("<b>Count Detected</b>", body_style), Paragraph("<b>Perception Role</b>", body_style)]]

    if not detection_summary:
        det_rows.append([Paragraph("No objects detected", body_style), Paragraph("0", body_style), Paragraph("Scene context", body_style)])
    else:
        for label, count in detection_summary.items():
            role = "Motor Vehicle" if label in ["Vehicle", "Car", "Bus", "Truck"] else "Other"
            det_rows.append([Paragraph(label, body_style), Paragraph(str(count), body_style), Paragraph(role, body_style)])

    det_table = Table(det_rows, colWidths=[160, 120, 250])
    det_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(det_table)
    story.append(Spacer(1, 12))

    # 4. Focus Prompt and Description
    story.append(Paragraph("3. Scene Observation & Analysis", heading_style))
    story.append(Paragraph(f"<b>Operator Focus Prompt:</b> {escape(user_prompt)}", body_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(escape(description), body_style))
    story.append(Spacer(1, 12))

    # 5. Technical Limitations & Governance
    story.append(Paragraph("4. Technical Limitations & Disclaimers", heading_style))
    limitations_text = (
        "• Single-image analysis cannot estimate vehicle velocity, acceleration, or speed limit compliance.<br/>"
        "• Detection and blurring are performed prior to any generative or visual reasoning layer.<br/>"
        "• Facial recognition and license plate OCR are strictly excluded by system design."
    )
    story.append(Paragraph(limitations_text, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
