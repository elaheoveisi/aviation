"""Generate the manuscript preprocessing/evaluation workflow figure.

This script creates a deterministic PNG using Pillow. No generative-image model is used.
"""
from __future__ import annotations

import argparse
import textwrap
from math import atan2, cos, sin
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    raise FileNotFoundError("No supported TrueType font was found. Install DejaVu Sans or Arial.")


def make_figure(output: Path) -> None:
    width, height = 4600, 4500
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    title_font = _font(90, True)
    subtitle_font = _font(42)
    box_heading_font = _font(62, True)
    box_body_font = _font(58)
    number_font = _font(66, True)
    note_heading_font = _font(74, True)
    note_body_font = _font(50)

    navy = "#17324d"
    body_color = "#1f2937"
    muted = "#4b5563"
    shadow = "#d8e0ea"
    arrow_color = "#2b6cb0"
    outline = "#23384d"
    fills = ["#dbeafe", "#ece7ff", "#fff4d6", "#def7f5", "#dff3e4", "#e8f1ff", "#fbe4e6", "#f5f7fb"]

    def centered(y: int, text: str, font, fill: str) -> None:
        box = draw.textbbox((0, 0), text, font=font)
        draw.text(((width - (box[2] - box[0])) / 2, y), text, font=font, fill=fill)

    centered(70, "Leakage-controlled aviation severity benchmark workflow", title_font, navy)
    centered(
        190,
        "All data-dependent steps were fit on training data only, and the safety threshold was selected without test-set information.",
        subtitle_font,
        muted,
    )

    margin_x, box_width, box_height, gap_x = 90, 980, 700, 120
    row1_y, row2_y = 340, 2140
    x_positions = [margin_x + i * (box_width + gap_x) for i in range(4)]
    steps = [
        ("Data construction", "Merge event, selected primary aircraft, and aggregated crew records.", fills[0]),
        ("Outcome", "Label severe events as at least one fatal or serious injury.", fills[1]),
        ("Timing audit", "Exclude leaked, post-event, and metadata-only predictors using the eADMS dictionary.", fills[2]),
        ("Transformations", "Create aircraft age and elapsed-inspection fields, and encode cyclic time and direction variables.", fills[3]),
        ("Training pipeline", "Impute, encode, oversample, and standardize logistic-regression inputs using training data only.", fills[4]),
        ("Probability correction", "Adjust predicted odds back to the original training prevalence before evaluation.", fills[5]),
        ("Validation", "Evaluate random split, future-year split, rolling temporal cuts, and transferred threshold performance.", fills[6]),
        ("Interpretation", "Report calibration, subgroup results, confusion counts, and SHAP profiles.", fills[7]),
    ]

    def center_lines(lines, center_x, y, font, fill, gap=10):
        current_y = y
        for line in lines:
            box = draw.textbbox((0, 0), line, font=font)
            draw.text((center_x - (box[2] - box[0]) / 2, current_y), line, font=font, fill=fill)
            current_y += (box[3] - box[1]) + gap
        return current_y

    def rounded_box(x, y, fill, title, body, number):
        draw.rounded_rectangle((x + 12, y + 14, x + box_width + 12, y + box_height + 14), radius=42, fill=shadow)
        draw.rounded_rectangle((x, y, x + box_width, y + box_height), radius=42, fill=fill, outline=outline, width=5)
        radius, badge_x, badge_y = 54, x + 22, y + 20
        draw.ellipse((badge_x, badge_y, badge_x + 2 * radius, badge_y + 2 * radius), fill=navy)
        number_box = draw.textbbox((0, 0), str(number), font=number_font)
        draw.text(
            (badge_x + radius - (number_box[2] - number_box[0]) / 2,
             badge_y + radius - (number_box[3] - number_box[1]) / 2 - 4),
            str(number), font=number_font, fill="white"
        )
        title_end = center_lines(textwrap.wrap(title, width=18), x + box_width / 2, y + 22, box_heading_font, navy, gap=6)
        center_lines(textwrap.wrap(body, width=20), x + box_width / 2, max(title_end + 26, y + 185), box_body_font, body_color, gap=12)

    for index in range(4):
        rounded_box(x_positions[index], row1_y, steps[index][2], steps[index][0], steps[index][1], index + 1)
    for index in range(4, 8):
        rounded_box(x_positions[7 - index], row2_y, steps[index][2], steps[index][0], steps[index][1], index + 1)

    def arrow(x1, y1, x2, y2, line_width=8):
        draw.line((x1, y1, x2, y2), fill=arrow_color, width=line_width)
        angle = atan2(y2 - y1, x2 - x1)
        length = 34
        p1 = (x2 + length * cos(angle + 2.7), y2 + length * sin(angle + 2.7))
        p2 = (x2 + length * cos(angle - 2.7), y2 + length * sin(angle - 2.7))
        draw.polygon([(x2, y2), p1, p2], fill=arrow_color)

    center_y_1 = row1_y + box_height / 2
    for index in range(3):
        arrow(x_positions[index] + box_width, center_y_1, x_positions[index + 1] - 30, center_y_1)
    arrow(x_positions[3] + box_width / 2, row1_y + box_height, x_positions[3] + box_width / 2, row2_y - 40)
    center_y_2 = row2_y + box_height / 2
    for index in range(3, 0, -1):
        arrow(x_positions[index], center_y_2, x_positions[index - 1] + box_width + 30, center_y_2)

    note_x, note_y, note_width, note_height = 140, 3900, 4320, 480
    draw.rounded_rectangle((note_x + 12, note_y + 14, note_x + note_width + 12, note_y + note_height + 14), radius=40, fill=shadow)
    draw.rounded_rectangle((note_x, note_y, note_x + note_width, note_y + note_height), radius=40, fill="#f8fafc", outline="#2c5282", width=5)
    draw.text((note_x + 38, note_y + 30), "Key safeguards", font=note_heading_font, fill=navy)
    notes = [
        "• Training-only preprocessing prevents leakage from development or test data.",
        "• The transferred safety threshold tests whether a historically selected operating point holds on future events.",
        "• Final interpretation emphasizes missed severe events, calibration, subgroup stability, and SHAP-based interpretation.",
    ]
    current_y = note_y + 130
    for note in notes:
        for line in textwrap.wrap(note, width=92):
            draw.text((note_x + 48, current_y), line, font=note_body_font, fill=body_color)
            current_y += 58
        current_y += 18

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, dpi=(300, 300))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/figures_final/figure_0_preprocessing_workflow.png")
    args = parser.parse_args()
    make_figure(Path(args.output))


if __name__ == "__main__":
    main()
