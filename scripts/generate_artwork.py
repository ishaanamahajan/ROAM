"""Generate deterministic, lightweight destination postcard SVGs.

The illustrations are intentionally bundled instead of fetched from a remote
image service, so pairwise choices continue to work offline.
"""

from __future__ import annotations

import html
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roam.data import DESTINATIONS  # noqa: E402 (path bootstrap for direct execution)


OUTPUT = ROOT / "assets" / "destinations"


def postcard(destination, index: int) -> str:
    dark, mid, light = destination.palette
    sun_x = 135 + (index * 67) % 520
    ridge = 235 + (index % 4) * 18
    title = html.escape(destination.name.upper())
    country = html.escape(destination.country)
    # A compact procedural landscape gives every item a consistent visual style
    # while palette and geometry variations keep comparisons visually distinct.
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="760" viewBox="0 0 1200 760" role="img" aria-label="Illustrated travel postcard for {title}, {country}">
<defs><linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop stop-color="{dark}"/><stop offset="1" stop-color="{mid}"/></linearGradient><filter id="grain"><feTurbulence baseFrequency=".8" numOctaves="2" seed="{index+2}" type="fractalNoise"/><feColorMatrix values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 .055 0"/></filter></defs>
<rect width="1200" height="760" fill="url(#sky)"/><circle cx="{sun_x}" cy="170" r="86" fill="{light}" opacity=".9"/>
<path d="M0 460 L180 {ridge} L350 455 L535 {ridge-90} L720 445 L930 {ridge-40} L1200 450 V760 H0Z" fill="{dark}" opacity=".88"/>
<path d="M0 525 Q220 455 420 530 T820 515 T1200 520 V760 H0Z" fill="{mid}"/>
<path d="M0 610 Q180 550 390 625 T790 595 T1200 610 V760 H0Z" fill="{light}" opacity=".92"/>
<rect x="45" y="42" width="1110" height="676" rx="18" fill="none" stroke="{light}" stroke-width="3" opacity=".65"/>
<text x="72" y="635" fill="{dark}" font-family="Arial, Helvetica, sans-serif" font-weight="800" font-size="68" letter-spacing="5">{title}</text>
<text x="76" y="685" fill="{dark}" font-family="Arial, Helvetica, sans-serif" font-weight="600" font-size="27" letter-spacing="7">{country.upper()} · ROAM</text>
<rect width="1200" height="760" filter="url(#grain)" opacity=".75"/></svg>'''


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for index, destination in enumerate(DESTINATIONS):
        (OUTPUT / f"{destination.slug}.svg").write_text(postcard(destination, index), encoding="utf-8")
    print(f"Generated {len(DESTINATIONS)} postcards in {OUTPUT}")


if __name__ == "__main__":
    main()
