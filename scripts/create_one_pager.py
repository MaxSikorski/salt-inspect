#!/usr/bin/env python3
"""Generate SALT Inspector one-pager PDF — RaaS (Results as a Service) positioning."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen import canvas

OUTPUT = "/Users/maxair/Documents/VL-JEPA research/SALT_Inspector_One_Pager.pdf"

# Colors
NAVY = HexColor("#0f1629")
DARK_BLUE = HexColor("#1a2340")
ACCENT = HexColor("#3b82f6")
ACCENT_LIGHT = HexColor("#60a5fa")
LIGHT_BG = HexColor("#f1f5f9")
LIGHT_BG2 = HexColor("#e8eef6")
SECTION_TEXT = HexColor("#1e293b")
BODY_TEXT = HexColor("#334155")
SUBTLE = HexColor("#64748b")
GREEN = HexColor("#10b981")
WHITE = white

W, H = letter  # 612 x 792

# Layout
LEFT_M = 36
RIGHT_M = W - 36
COL_GAP = 20
COL1_X = LEFT_M
COL1_W = (W - 2 * LEFT_M - COL_GAP) / 2
COL2_X = COL1_X + COL1_W + COL_GAP
COL2_W = COL1_W


def section_header(c, x, y, title):
    c.setFillColor(ACCENT)
    c.rect(x, y - 2, 3, 15, fill=1, stroke=0)
    c.setFillColor(SECTION_TEXT)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x + 10, y, title)
    return y - 20


def bullet(c, x, y, text, max_w=260):
    c.setFillColor(ACCENT)
    c.setFont("Helvetica", 6)
    c.drawString(x, y + 2.5, "\u25cf")
    c.setFillColor(BODY_TEXT)
    c.setFont("Helvetica", 9)
    words = text.split()
    line = ""
    ly = y
    for word in words:
        test = (line + " " + word).strip()
        if c.stringWidth(test, "Helvetica", 9) < max_w - 14:
            line = test
        else:
            c.drawString(x + 12, ly, line)
            ly -= 13
            line = word
    if line:
        c.drawString(x + 12, ly, line)
        ly -= 13
    return ly - 1


def bold_bullet(c, x, y, bold, rest, max_w=260):
    c.setFillColor(ACCENT)
    c.setFont("Helvetica", 6)
    c.drawString(x, y + 2.5, "\u25cf")
    c.setFillColor(SECTION_TEXT)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 12, y, bold)
    bw = c.stringWidth(bold, "Helvetica-Bold", 9)
    c.setFillColor(BODY_TEXT)
    c.setFont("Helvetica", 9)
    full_rest = " " + rest
    # Check if it fits on one line
    total_w = bw + c.stringWidth(full_rest, "Helvetica", 9)
    if total_w < max_w - 14:
        c.drawString(x + 12 + bw, y, full_rest)
        return y - 15
    else:
        c.drawString(x + 12 + bw, y, full_rest)
        return y - 15


def step_box(c, x, y, num, title, desc, box_w):
    box_h = 50
    c.setFillColor(LIGHT_BG)
    c.roundRect(x, y - box_h, box_w, box_h, 5, fill=1, stroke=0)
    # Number circle
    c.setFillColor(ACCENT)
    c.circle(x + 18, y - 15, 10, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(x + 18, y - 19, str(num))
    # Title
    c.setFillColor(SECTION_TEXT)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(x + 34, y - 18, title)
    # Desc
    c.setFillColor(BODY_TEXT)
    c.setFont("Helvetica", 8)
    words = desc.split()
    line = ""
    ly = y - 33
    for word in words:
        test = (line + " " + word).strip()
        if c.stringWidth(test, "Helvetica", 8) < box_w - 22:
            line = test
        else:
            c.drawString(x + 12, ly, line)
            ly -= 11
            line = word
    if line:
        c.drawString(x + 12, ly, line)
    return y - box_h - 7


def main():
    c = canvas.Canvas(OUTPUT, pagesize=letter)
    c.setTitle("SALT Inspector - Inspection Results in 48 Hours")
    c.setAuthor("Sikorski AI Research Lab")

    # ══════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════
    header_h = 82
    tag_h = 22
    y_top = H

    c.setFillColor(NAVY)
    c.rect(0, y_top - header_h, W, header_h, fill=1, stroke=0)
    c.setFillColor(ACCENT)
    c.rect(0, y_top - header_h, W, 2.5, fill=1, stroke=0)

    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 28)
    c.drawString(LEFT_M, y_top - 36, "SALT Inspector")

    c.setFillColor(ACCENT_LIGHT)
    c.setFont("Helvetica", 11.5)
    c.drawString(LEFT_M, y_top - 54, "Inspection Results in 48 Hours  \u2014  No Defects Needed, No Install Required")

    c.setFillColor(HexColor("#94a3b8"))
    c.setFont("Helvetica", 8.5)
    c.drawRightString(RIGHT_M, y_top - 30, "Sikorski AI Research Lab")
    c.setFillColor(ACCENT_LIGHT)
    c.setFont("Helvetica-Bold", 9)
    c.drawRightString(RIGHT_M, y_top - 44, "max@sikorski.ai")
    c.setFillColor(HexColor("#94a3b8"))
    c.setFont("Helvetica", 8.5)
    c.drawRightString(RIGHT_M, y_top - 58, "salt-inspect.com")

    # Tagline bar
    c.setFillColor(DARK_BLUE)
    c.rect(0, y_top - header_h - tag_h, W, tag_h, fill=1, stroke=0)
    c.setFillColor(HexColor("#cbd5e1"))
    c.setFont("Helvetica-Oblique", 8)
    c.drawCentredString(W / 2, y_top - header_h - 15,
        "Results in 48 Hours  \u2022  No Defects Needed  \u2022  No Install Required  \u2022  The Demo IS the Product")

    content_top = y_top - header_h - tag_h - 16

    # ══════════════════════════════════════
    # COLUMN 1
    # ══════════════════════════════════════
    y1 = content_top

    # THE PROBLEM
    y1 = section_header(c, COL1_X, y1, "THE PROBLEM")
    y1 = bold_bullet(c, COL1_X, y1, "Every manufacturer KNOWS",
        "they need AI inspection", COL1_W)
    y1 = bold_bullet(c, COL1_X, y1, "Every vendor requires",
        "defect data, proprietary hardware, and months", COL1_W)
    y1 = bold_bullet(c, COL1_X, y1, "So it stays on the backlog",
        "\u2014 until now", COL1_W)
    y1 -= 10

    # THE SOLUTION
    y1 = section_header(c, COL1_X, y1, "THE SOLUTION")
    y1 = bullet(c, COL1_X, y1,
        "Tell us what you make (or send photos of good parts)", COL1_W)
    y1 = bullet(c, COL1_X, y1,
        "We train your custom model in 48 hours \u2014 you touch nothing", COL1_W)
    y1 = bullet(c, COL1_X, y1,
        "Open your link \u2014 browser-based results, no install required", COL1_W)
    y1 -= 8

    # HOW IT WORKS
    y1 = section_header(c, COL1_X, y1, "HOW IT WORKS")
    y1 = step_box(c, COL1_X, y1, 1, "Send Good Parts",
        "Email us 100\u2013500 photos of normal parts. Any camera works.", COL1_W)
    y1 = step_box(c, COL1_X, y1, 2, "We Train (48h)",
        "Custom AI model trained on YOUR parts. You touch nothing.", COL1_W)
    y1 = step_box(c, COL1_X, y1, 3, "Open Your Link",
        "Browser-based results. Phone, tablet, or factory floor.", COL1_W)

    y1 -= 8

    # TARGET INDUSTRIES (fills bottom of left column)
    y1 = section_header(c, COL1_X, y1, "TARGET INDUSTRIES")
    industries = [
        "Electronics Manufacturing \u2014 PCB & solder inspection",
        "Automotive Parts \u2014 CNC machined parts, surface finish",
        "Pharmaceutical \u2014 pill, capsule & packaging inspection",
        "Additive Manufacturing \u2014 3D print layer defects",
        "Textiles & Leather \u2014 fabric defect detection",
    ]
    for ind in industries:
        c.setFillColor(ACCENT)
        c.setFont("Helvetica", 6)
        c.drawString(COL1_X, y1 + 2, "\u25cf")
        c.setFillColor(BODY_TEXT)
        c.setFont("Helvetica", 8.5)
        c.drawString(COL1_X + 12, y1, ind)
        y1 -= 14

    # ══════════════════════════════════════
    # COLUMN 2
    # ══════════════════════════════════════
    y2 = content_top

    # PERFORMANCE
    y2 = section_header(c, COL2_X, y2, "PERFORMANCE")

    metrics = [
        ("48h", "Time to Results"),
        ("$5K", "Discovery Sprint"),
        ("$0", "Install Cost"),
        ("3", "Steps to Results"),
    ]
    met_w = (COL2_W - 10) / 2
    met_h = 44
    for i, (val, label) in enumerate(metrics):
        mx = COL2_X + (i % 2) * (met_w + 10)
        my = y2 - (i // 2) * (met_h + 8)
        c.setFillColor(LIGHT_BG)
        c.roundRect(mx, my - met_h, met_w, met_h, 4, fill=1, stroke=0)
        c.setFillColor(ACCENT)
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(mx + met_w / 2, my - 18, val)
        c.setFillColor(SUBTLE)
        c.setFont("Helvetica", 7.5)
        c.drawCentredString(mx + met_w / 2, my - 32, label)

    y2 -= 2 * (met_h + 8) + 8

    y2 = bullet(c, COL2_X, y2, "Send us good parts, get inspection results \u2014 that simple", COL2_W)
    y2 = bullet(c, COL2_X, y2, "Custom model trained on YOUR parts, not generic datasets", COL2_W)
    y2 = bullet(c, COL2_X, y2, "Browser-based results \u2014 nothing to install or configure", COL2_W)
    y2 -= 8

    # ENGAGEMENT MODEL
    y2 = section_header(c, COL2_X, y2, "ENGAGEMENT MODEL")

    table_data = [
        ("Stage", "Investment", "Timeline"),
        ("Discovery Sprint", "$5,000", "48 hours"),
        ("Production Deploy", "$100,000", "4\u20138 weeks"),
        ("Retainer", "$5,000/mo", "Ongoing"),
    ]
    col_ws = [COL2_W * 0.42, COL2_W * 0.32, COL2_W * 0.26]
    row_h = 17

    for r, row in enumerate(table_data):
        ry = y2 - r * row_h
        if r == 0:
            c.setFillColor(DARK_BLUE)
            c.rect(COL2_X, ry - row_h + 4, COL2_W, row_h, fill=1, stroke=0)
        elif r % 2 == 0:
            c.setFillColor(LIGHT_BG)
            c.rect(COL2_X, ry - row_h + 4, COL2_W, row_h, fill=1, stroke=0)
        cx = COL2_X
        for j, cell in enumerate(row):
            if r == 0:
                c.setFillColor(WHITE)
                c.setFont("Helvetica-Bold", 8)
            elif j == 1:
                c.setFillColor(SECTION_TEXT)
                c.setFont("Helvetica-Bold", 8)
            else:
                c.setFillColor(BODY_TEXT)
                c.setFont("Helvetica", 8)
            c.drawString(cx + 6, ry - row_h + 8, cell)
            cx += col_ws[j]

    y2 -= len(table_data) * row_h + 12

    # WHY SALT INSPECTOR
    y2 = section_header(c, COL2_X, y2, "WHY SALT INSPECTOR")

    advantages = [
        ("48 hours, not 6 months", " \u2014 results before your next standup"),
        ("No defects needed", " \u2014 just photos of good parts"),
        ("No install required", " \u2014 open a link, see results"),
        ("The demo IS the product", " \u2014 what you see is what you get"),
        ("3 clicks, not 3 months", " \u2014 send parts, get inspection"),
    ]
    for bold_part, rest_part in advantages:
        c.setFillColor(GREEN)
        c.setFont("Helvetica", 7)
        c.drawString(COL2_X, y2 + 2, "\u2713")
        c.setFillColor(SECTION_TEXT)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(COL2_X + 12, y2, bold_part)
        bw = c.stringWidth(bold_part, "Helvetica-Bold", 9)
        c.setFillColor(BODY_TEXT)
        c.setFont("Helvetica", 9)
        c.drawString(COL2_X + 12 + bw, y2, rest_part)
        y2 -= 16

    y2 -= 8

    # CALL TO ACTION box
    cta_h = 52
    c.setFillColor(LIGHT_BG2)
    c.roundRect(COL2_X, y2 - cta_h, COL2_W, cta_h, 5, fill=1, stroke=0)
    # Border
    c.setStrokeColor(ACCENT)
    c.setLineWidth(1)
    c.roundRect(COL2_X, y2 - cta_h, COL2_W, cta_h, 5, fill=0, stroke=1)

    c.setFillColor(SECTION_TEXT)
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(COL2_X + COL2_W / 2, y2 - 18, "Get inspection results in 48 hours")
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(COL2_X + COL2_W / 2, y2 - 34,
        "Start with a $5K Discovery Sprint")
    c.setFillColor(SUBTLE)
    c.setFont("Helvetica", 8)
    c.drawCentredString(COL2_X + COL2_W / 2, y2 - 47,
        "Send us good parts. We send you results.")

    # ══════════════════════════════════════
    # DIVIDER LINE between columns
    # ══════════════════════════════════════
    c.setStrokeColor(HexColor("#e2e8f0"))
    c.setLineWidth(0.5)
    div_x = COL1_X + COL1_W + COL_GAP / 2
    c.line(div_x, content_top + 5, div_x, 52)

    # ══════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════
    footer_h = 40
    c.setFillColor(NAVY)
    c.rect(0, 0, W, footer_h, fill=1, stroke=0)
    c.setFillColor(ACCENT)
    c.rect(0, footer_h - 1.5, W, 1.5, fill=1, stroke=0)

    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(LEFT_M, footer_h - 17, "Try it live:")
    c.setFillColor(ACCENT_LIGHT)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(LEFT_M + 56, footer_h - 17, "salt-inspect.com")
    c.setFillColor(HexColor("#94a3b8"))
    c.setFont("Helvetica", 7.5)
    c.drawString(LEFT_M, footer_h - 31, "Results in 48 hours \u2014 No defects needed \u2014 No install required")

    c.setFillColor(ACCENT_LIGHT)
    c.setFont("Helvetica-Bold", 9)
    c.drawRightString(RIGHT_M, footer_h - 17, "Tap the NFC card to see it live")
    c.setFillColor(HexColor("#94a3b8"))
    c.setFont("Helvetica", 7.5)
    c.drawRightString(RIGHT_M, footer_h - 31, "max@sikorski.ai  |  Sikorski AI Research Lab")

    c.save()
    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    main()
