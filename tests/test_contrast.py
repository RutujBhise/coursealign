"""
WCAG 2.1 AA color contrast for the dashboard, checked from the color tokens
in web/src/styles.css, in light and dark mode. No browser needed, so a
palette edit that breaks contrast fails here before anyone opens the page.
(axe-core checks the rendered page too: web/e2e/accessibility.spec.ts.)

RUN:  pytest tests/   or   python3 tests/test_contrast.py
"""

import re
from pathlib import Path

CSS = (Path(__file__).resolve().parents[1] / "web" / "src" / "styles.css").read_text(encoding="utf-8")

# (foreground, background) pairs as the page actually uses them.
TEXT = [  # 4.5:1 (1.4.3)
    *[("graphite", bg) for bg in ("paper", "sheet", "band")],
    *[("graphite-soft", bg) for bg in ("paper", "sheet", "band")],
    ("dropout-strong", "sheet"), ("ink", "sheet"), ("ink", "band"),
    ("pen", "sheet"), ("pen", "pen-wash"), ("amber", "sheet"), ("amber", "amber-wash"),
    ("on-dropout", "dropout"), ("on-dropout", "dropout-strong"),
]
SHAPES = [  # 3:1 (1.4.11): bubbles, borders that mark controls, focus, margin marks
    *[(fg, bg) for fg in ("graphite", "graphite-soft", "ink", "focus") for bg in ("sheet", "band")],
    ("focus", "paper"), ("pen", "sheet"), ("pen", "band"), ("amber", "sheet"), ("amber", "band"),
    ("dropout", "sheet"), ("dropout", "band"),
]


def tokens(block: str) -> dict[str, str]:
    return dict(re.findall(r"--([a-z-]+):\s*(#[0-9a-fA-F]{6})", block))


def schemes() -> dict[str, dict[str, str]]:
    light = tokens(CSS.split("@media (prefers-color-scheme: dark)")[0])
    dark_block = re.search(r"@media \(prefers-color-scheme: dark\)\s*\{(.*?)\n\}", CSS, re.S).group(1)
    return {"light": light, "dark": {**light, **tokens(dark_block)}}


def luminance(hex_color: str) -> float:
    channels = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def failures(pairs, minimum: float) -> list[str]:
    found = []
    for scheme, palette in schemes().items():
        for fg, bg in pairs:
            ratio = contrast(palette[fg], palette[bg])
            if ratio < minimum:
                found.append(f"{scheme}: {fg} on {bg} is {ratio:.2f}:1, needs {minimum}:1")
    return found


def test_text_meets_4_5_to_1():
    assert failures(TEXT, 4.5) == []


def test_marks_borders_and_focus_meet_3_to_1():
    assert failures(SHAPES, 3.0) == []


if __name__ == "__main__":
    for scheme, palette in schemes().items():
        worst = min((contrast(palette[f], palette[b]), f, b) for f, b in TEXT)
        print(f"  {scheme}: lowest text contrast {worst[0]:.2f}:1 ({worst[1]} on {worst[2]})")
    for test in (test_text_meets_4_5_to_1, test_marks_borders_and_focus_meet_3_to_1):
        test()
        print(f"  ok  {test.__name__}")
    print("\n2 passed")
