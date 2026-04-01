#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


WIDTH = 1600
HEIGHT = 900
BG = (246, 241, 233)
INK = (33, 29, 24)
ACCENT = (140, 62, 38)
PANEL = (232, 225, 214)
MONO_BG = (245, 242, 236)


def _font(size: int, *, mono: bool = False, bold: bool = False):
    candidates = []
    if mono:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            ]
        )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


TITLE_FONT = _font(38, bold=True)
BODY_FONT = _font(24)
SMALL_FONT = _font(20)
MONO_FONT = _font(18, mono=True)
MONO_BOLD = _font(18, mono=True, bold=True)


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)


def _bullet_lines(items: list[str], *, width: int) -> list[str]:
    lines: list[str] = []
    for item in items:
        wrapped = _wrap(item, width)
        if not wrapped:
            continue
        lines.append(f"- {wrapped[0]}")
        lines.extend([f"  {line}" for line in wrapped[1:]])
    return lines


def _read_lines(path: Path, start: int, end: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    chunk = lines[start - 1 : end]
    return "\n".join(chunk)


def _base_canvas(title: str, *, subtitle: str | None = None) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    draw.text((70, 50), title, font=TITLE_FONT, fill=INK)
    draw.line((70, 110, WIDTH - 70, 110), fill=ACCENT, width=4)
    y = 135
    if subtitle:
        for line in _wrap(subtitle, 90):
            draw.text((70, y), line, font=BODY_FONT, fill=INK)
            y += 30
        y += 10
    return img, draw, y


def _draw_text_block(draw: ImageDraw.ImageDraw, x: int, y: int, lines: list[str], *, font, fill=INK, line_h=30) -> int:
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h
    return y


def _paste_contain(canvas: Image.Image, path: Path, box: tuple[int, int, int, int]) -> None:
    with Image.open(path) as img:
        fitted = ImageOps.contain(img.convert("RGB"), (box[2] - box[0], box[3] - box[1]))
        x = box[0] + (box[2] - box[0] - fitted.width) // 2
        y = box[1] + (box[3] - box[1] - fitted.height) // 2
        canvas.paste(fitted, (x, y))


def _draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], *, outline=(205, 190, 170), fill=PANEL, width=2) -> None:
    draw.rounded_rectangle(box, radius=18, outline=outline, fill=fill, width=width)


def _render_text_slide(title: str, bullets: list[str], out_png: Path) -> None:
    img, draw, y = _base_canvas(title)
    panel = (60, 150, WIDTH - 60, HEIGHT - 60)
    _draw_panel(draw, panel)
    lines = _bullet_lines(bullets, width=80)
    _draw_text_block(draw, 100, 190, lines, font=BODY_FONT, line_h=34)
    img.save(out_png)


def _render_code_slide(title: str, blocks: list[tuple[str, str]], bullets: list[str], out_png: Path) -> None:
    img, draw, _ = _base_canvas(title, subtitle="Real source-code evidence from the bridge layer.")
    code_boxes = [
        (60, 200, 500, 760),
        (540, 200, 1040, 760),
        (1080, 200, 1540, 760),
    ]
    for (label, code), box in zip(blocks, code_boxes):
        _draw_panel(draw, box, fill=MONO_BG)
        draw.text((box[0] + 20, box[1] + 16), label, font=SMALL_FONT, fill=ACCENT)
        code_lines = code.splitlines()
        y = box[1] + 56
        for line in code_lines[:24]:
            draw.text((box[0] + 20, y), line, font=MONO_FONT, fill=INK)
            y += 22
    bullet_box = (80, 780, WIDTH - 80, HEIGHT - 40)
    lines = _bullet_lines(bullets, width=95)
    _draw_text_block(draw, bullet_box[0], bullet_box[1], lines, font=SMALL_FONT, line_h=26)
    img.save(out_png)


def _render_image_slide(title: str, image_path: Path, bullets: list[str], out_png: Path) -> None:
    img, draw, _ = _base_canvas(title)
    left = (60, 150, 1080, 820)
    right = (1110, 150, 1540, 820)
    _draw_panel(draw, left, fill=(250, 248, 243))
    _draw_panel(draw, right)
    _paste_contain(img, image_path, left)
    lines = _bullet_lines(bullets, width=30)
    _draw_text_block(draw, right[0] + 22, right[1] + 24, lines, font=SMALL_FONT, line_h=28)
    img.save(out_png)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the six-slide self-collision update pack.")
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--exactness-json", type=Path, default=None)
    parser.add_argument("--strict-parity-report", type=Path, default=None)
    parser.add_argument("--off-parity-report", type=Path, default=None)
    parser.add_argument("--compare-summary-json", type=Path, default=None)
    args = parser.parse_args()

    root = args.campaign_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exactness_path = (
        args.exactness_json.resolve()
        if args.exactness_json is not None
        else (root / "equivalence" / "verify_phystwin_self_collision_equivalence.json").resolve()
    )
    exactness = json.loads(exactness_path.read_text())

    strict_report_path = (
        args.strict_parity_report.resolve()
        if args.strict_parity_report is not None
        else (root / "parity" / "strict_self_collision_parity_summary.json").resolve()
    )
    strict_report = json.loads(strict_report_path.read_text())

    off_report_path = args.off_parity_report.resolve() if args.off_parity_report is not None else None
    compare_summary_path = (
        args.compare_summary_json.resolve() if args.compare_summary_json is not None else None
    )
    compare_summary = (
        json.loads(compare_summary_path.read_text()) if compare_summary_path is not None else None
    )

    if compare_summary is None and off_report_path is not None:
        off_report = json.loads(off_report_path.read_text())
        compare_summary = {
            "case": "blue_cloth_double_lift_around",
            "frames": int(strict_report["rollout_summary"]["simulation"]["frames_run"]),
            "off": {k: float(off_report["checks"][k]["value"]) for k in ("x0_rmse", "rmse_mean", "rmse_max", "first30_rmse", "last30_rmse")},
            "strict_phystwin": {k: float(strict_report["checks"][k]["value"]) for k in ("x0_rmse", "rmse_mean", "rmse_max", "first30_rmse", "last30_rmse")},
        }

    slide1_md = """# Problem, Claim, And Constraint

- Problem: Newton native self-collision is not the same semantic object as PhysTwin self-collision under the current bridge mapping.
- Final claim: We cannot rely only on Newton native self-collision as the final solution.
- Chinese clarification: 我们不能把 Newton 自带的 self-collision 当最终方案。
- No Newton core change.
- Final self-collision mode for the demo: `phystwin`.
- Current campaign state: exactness PASS, demo QC PASS, full-rollout A/B FAIL.
"""
    slide2_md = """# Source-Code Evidence

- `newton_import_ir.py:973-980` maps `contact_collision_dist` into Newton per-particle radius semantics.
- `demo_cloth_box_drop_with_self_contact.py:260-298` defines `off/native/custom/phystwin` and marks `particle-self-contact-scale` as compatibility-only.
- `self_contact_bridge_kernels.py:235-304` implements the bridge-side PhysTwin-style velocity operator.
- PhysTwin cloth parity source only defines pairwise `object_collision` plus implicit `integrate_ground_collision`.
"""
    slide3_md = """# Cloth+Box Decision Matrix

- Controlled cloth+box scene only.
- The matrix scorer provisionally ranks `native`, but that is not the campaign decision.
- Final decision rule combines:
  - scene evidence from this matrix
  - source-code semantic mismatch
  - operator-level exactness requirement
- Strict `phystwin` parity is not claimed on the box scene because the PhysTwin source does not define generic box-support contact for this path.
- Final conclusion: native is not sufficient as the final solution.
"""
    slide4_md = f"""# Bridge-Side PhysTwin Exactness

- `pass = {exactness['pass']}`
- `max_abs_dv = {exactness['max_abs_dv']}`
- `median_rel_dv = {exactness['median_rel_dv']}`
- Gate:
  - `max_abs_dv <= 1e-5`
  - `median_rel_dv <= 1e-4`
"""
    slide5_md = """# Final Demo A

- Video: `selected/self_collision_on_cloth_box_phystwin.mp4`
- Mode: `phystwin`
- QC verdict: `PASS`
- Camera: top-down presentation view
- Ground plane: disabled
- Pair semantics: zero excluded pairs
- This slide is scene-level demo evidence, not the strict parity scope.
"""
    if compare_summary is not None:
        slide6_md = f"""# Strict Self-Collision Parity

- Case: `{compare_summary['case']}`
- Full-rollout A/B gate: require `rmse_mean(phystwin) < rmse_mean(off)`
- OFF `rmse_mean = {compare_summary['off']['rmse_mean']}`
- strict `phystwin` `rmse_mean = {compare_summary['strict_phystwin']['rmse_mean']}`
- OFF `last30_rmse = {compare_summary['off']['last30_rmse']}`
- strict `phystwin` `last30_rmse = {compare_summary['strict_phystwin']['last30_rmse']}`
- strict `phystwin` only improves the early window: `first30_rmse = {compare_summary['strict_phystwin']['first30_rmse']}` vs OFF `{compare_summary['off']['first30_rmse']}`
- Conclusion: full-rollout A/B currently fails; rollout mismatch remains.
"""
    else:
        slide6_md = f"""# Strict Self-Collision Parity

- Case: `blue_cloth_double_lift_around`
- `rmse_mean = {strict_report['rmse_mean']}`
- `rmse_max = {strict_report['rmse_max']}`
- `first30_rmse = {strict_report['first30_rmse']}`
- `last30_rmse = {strict_report['last30_rmse']}`
- Strict parity scope: PhysTwin-native cloth contact only (`object_collision` + implicit `z=0` ground plane).
- Conclusion: the current bridge rollout semantics do not satisfy the requested `1e-5` gate on the in-scope cloth self-collision case.
"""

    (out_dir / "01_question_and_claim.md").write_text(slide1_md, encoding="utf-8")
    (out_dir / "02_native_source_code_evidence.md").write_text(slide2_md, encoding="utf-8")
    (out_dir / "03_native_failure_matrix.md").write_text(slide3_md, encoding="utf-8")
    (out_dir / "04_phystwin_exactness.md").write_text(slide4_md, encoding="utf-8")
    (out_dir / "05_final_demo_frames.md").write_text(slide5_md, encoding="utf-8")
    (out_dir / "06_strict_parity.md").write_text(slide6_md, encoding="utf-8")

    slide1_bullets = [
        "Problem: Newton native self-collision is not the same semantic object as PhysTwin self-collision under the current bridge mapping.",
        "Final claim: We cannot rely only on Newton native self-collision as the final solution.",
        "Chinese clarification: 我们不能把 Newton 自带的 self-collision 当最终方案。",
        "No Newton core change.",
        "Final self-collision mode for Demo A: phystwin.",
        "Campaign state: exactness PASS, demo QC PASS, strict parity BLOCKED by rollout mismatch.",
    ]
    _render_text_slide("Problem, Claim, And Constraint", slide1_bullets, out_dir / "01_question_and_claim.png")

    blocks = [
        (
            "newton_import_ir.py:1020-1028",
            _read_lines(root.parents[1] / "tools" / "core" / "newton_import_ir.py", 1020, 1028),
        ),
        (
            "demo_cloth_box_drop_with_self_contact.py:260-307",
            _read_lines(root.parents[1] / "demos" / "demo_cloth_box_drop_with_self_contact.py", 260, 307),
        ),
        (
            "self_contact_bridge_kernels.py:491-558",
            _read_lines(root.parents[1] / "demos" / "self_contact_bridge_kernels.py", 491, 558),
        ),
    ]
    slide2_bullets = [
        "The importer maps PhysTwin pairwise collision distance into Newton particle-radius semantics.",
        "The cloth-box driver explicitly separates off/native/custom/phystwin.",
        "The strict bridge path now uses a frame-frozen explicit collision table and a bridge-side PhysTwin-style operator.",
        "PhysTwin cloth parity source only defines object_collision plus implicit z=0 ground.",
    ]
    _render_code_slide("Source-Code Evidence", blocks, slide2_bullets, out_dir / "02_native_source_code_evidence.png")

    slide3_bullets = [
        "Controlled cloth+box scene only.",
        "The matrix scorer provisionally ranks native, but that is not the campaign decision.",
        "The final decision combines this scene evidence with the source-code semantic mismatch and the exactness gate.",
        "Strict phystwin parity is intentionally not claimed on the box scene.",
        "Final conclusion: native is not sufficient as the final solution.",
    ]
    _render_image_slide(
        "Cloth+Box Decision Matrix",
        root / "matrix" / "self_collision_decision_4panel.png",
        slide3_bullets,
        out_dir / "03_native_failure_matrix.png",
    )

    slide4_bullets = [
        f"pass = {exactness['pass']}",
        f"max_abs_dv = {exactness['max_abs_dv']}",
        f"median_rel_dv = {exactness['median_rel_dv']}",
        "Gate: max_abs_dv <= 1e-5, median_rel_dv <= 1e-4.",
        "This is operator-level evidence only.",
    ]
    _render_text_slide("Bridge-Side PhysTwin Exactness", slide4_bullets, out_dir / "04_phystwin_exactness.png")

    slide5_bullets = [
        "Video: selected/self_collision_on_cloth_box_phystwin.mp4",
        "Mode: phystwin",
        "QC verdict: PASS",
        "Top-down presentation camera",
        "Ground plane disabled",
        "Pair semantics: zero excluded pairs",
        "Scene-level demo evidence only, not the strict parity scope.",
    ]
    _render_image_slide(
        "Final Demo A",
        root / "video_qc" / "phystwin_topdown_v3" / "contact_sheet.png",
        slide5_bullets,
        out_dir / "05_final_demo_frames.png",
    )

    if compare_summary is not None:
        slide6_bullets = [
            f"Case: {compare_summary['case']}",
            "Full-rollout A/B gate: require rmse_mean(phystwin) < rmse_mean(off).",
            f"OFF rmse_mean = {compare_summary['off']['rmse_mean']}",
            f"strict phystwin rmse_mean = {compare_summary['strict_phystwin']['rmse_mean']}",
            f"OFF last30_rmse = {compare_summary['off']['last30_rmse']}",
            f"strict phystwin last30_rmse = {compare_summary['strict_phystwin']['last30_rmse']}",
            (
                "strict phystwin only improves first30_rmse: "
                f"{compare_summary['strict_phystwin']['first30_rmse']} vs OFF {compare_summary['off']['first30_rmse']}"
            ),
            "Conclusion: full-rollout A/B currently fails; rollout mismatch remains.",
        ]
    else:
        slide6_bullets = [
            "Case: blue_cloth_double_lift_around",
            f"rmse_mean = {strict_report['rmse_mean']}",
            f"rmse_max = {strict_report['rmse_max']}",
            f"first30_rmse = {strict_report['first30_rmse']}",
            f"last30_rmse = {strict_report['last30_rmse']}",
            "Strict scope: object_collision + implicit z=0 ground plane.",
            "Conclusion: the current bridge rollout semantics do not satisfy the requested 1e-5 gate on the in-scope cloth self-collision case.",
        ]
    _render_image_slide(
        "Strict Self-Collision Parity",
        root / "video_qc" / "parity_support_v1" / "contact_sheet.png",
        slide6_bullets,
        out_dir / "06_strict_parity.png",
    )

    update_md = f"""# SLIDES_UPDATE

Use this pack for the April 1 deck self-collision block.

1. 01_question_and_claim
2. 02_native_source_code_evidence
3. 03_native_failure_matrix
4. 04_phystwin_exactness
5. 05_final_demo_frames
6. 06_strict_parity

Campaign root: `{root}`
Demo A: `{root / 'selected' / 'self_collision_on_cloth_box_phystwin.mp4'}`
Demo A QC: `{root / 'video_qc' / 'phystwin_topdown_v3' / 'video_qc.json'}`
Exactness JSON: `{exactness_path}`
Strict parity report: `{strict_report_path}`
OFF parity report: `{off_report_path if off_report_path is not None else 'N/A'}`
Compare summary: `{compare_summary_path if compare_summary_path is not None else 'N/A'}`
Parity blocker: `{root / 'BLOCKER_strict_self_collision_parity_bridge_rollout_mismatch.md'}`
"""
    (out_dir / "SLIDES_UPDATE.md").write_text(update_md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
