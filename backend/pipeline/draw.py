"""Frame annotation: boxes, stable track IDs, HUD."""

import numpy as np
import cv2

from .detector import Track

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# HUD colours matched to the frontend's modern-minimal Pastel theme (BGR). Cool near-white
# text on a cool-graphite slab, so the baked-in overlay reads as part of the surrounding UI.
_HUD_TEXT = (238, 232, 226)   # cool near-white
_HUD_SLAB = (54, 44, 38)      # cool graphite, not pure black


def _color_for(track_id: int | None) -> tuple[int, int, int]:
    """Deterministic per-ID colour — same object keeps its colour across frames."""
    if track_id is None:
        return (128, 128, 128)
    rng = (track_id * 47) % 180
    hsv = np.uint8([[[rng, 220, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def draw_overlay(
    frame: np.ndarray,
    tracks: list[Track],
    clip_result: list[tuple[str, float]],
    fps: float,
) -> np.ndarray:
    out = frame.copy()

    for tr in tracks:
        x1, y1, x2, y2 = tr.xyxy
        color = _color_for(tr.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        tag = f"#{tr.track_id} {tr.label} {tr.conf:.2f}" if tr.track_id is not None \
            else f"{tr.label} {tr.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, _FONT, 0.5, 1)

        # Boxes touching the top edge would render their label off-frame; flip it inside.
        # Same for the right edge, where a long tag would run off.
        top = y1 - th - 6
        ty = top if top >= 0 else y1 + 2
        tx = min(x1, out.shape[1] - tw - 6)
        tx = max(tx, 0)
        cv2.rectangle(out, (tx, ty), (tx + tw + 4, ty + th + 6), color, -1)
        cv2.putText(out, tag, (tx + 2, ty + th + 1), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    hud = [f"FPS {fps:5.1f}", f"tracked {len(tracks)}"]
    if clip_result:
        prompt, score = clip_result[0]
        clipped = prompt if len(prompt) <= 42 else prompt[:39] + "..."
        hud.append(f"CLIP: {clipped} ({score:.0%})")

    y = 22
    for line in hud:
        (tw, th), _ = cv2.getTextSize(line, _FONT, 0.6, 2)
        cv2.rectangle(out, (8, y - th - 4), (12 + tw, y + 4), _HUD_SLAB, -1)
        cv2.putText(out, line, (10, y), _FONT, 0.6, _HUD_TEXT, 2, cv2.LINE_AA)
        y += th + 12

    return out
