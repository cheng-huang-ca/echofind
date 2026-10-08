"""A submerged body as a few aspect-dependent highlights over about 1.7 m.

This is a phenomenological model, not body acoustics (see
reports/decisions/102-body-highlight-model.md). The body is a line of point scatterers along its
long axis: head, chest (air in the lungs dominates), pelvis, knees and feet. Each highlight's
amplitude follows a broadside lobe so the echo is strongest when the beam meets the body side-on
and falls to `end_on_db` below that end-on. The total target strength at broadside equals
`ts_broadside_db` (incoherent sum of highlight powers).

Aspect convention: `aspect_deg` is the angle between the line of sight and the body's long axis;
90 is broadside, 0 is head-on. Highlight k sits at along-axis offset x_k, so its extra one-way
range is x_k cos(aspect).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# along-axis position (m, head at -0.85) and relative power of each highlight
DEFAULT_HIGHLIGHTS = (
    (-0.80, 0.10),   # head
    (-0.45, 0.50),   # chest / lungs
    (-0.05, 0.20),   # pelvis
    (0.35, 0.10),    # knees
    (0.80, 0.10),    # feet
)


@dataclass(frozen=True)
class Body:
    length_m: float = 1.7
    ts_broadside_db: float = -20.0
    end_on_db: float = -10.0       # end-on TS relative to broadside
    lobe_width_deg: float = 30.0   # half-power width of the broadside lobe
    highlights: tuple = field(default=DEFAULT_HIGHLIGHTS)
    fluctuating: bool = True       # random highlight phases each ping

    def aspect_gain(self, aspect_deg: float) -> float:
        """Power gain of the broadside lobe at this aspect, floored at end_on_db."""
        off = np.deg2rad(aspect_deg - 90.0)
        bw = np.deg2rad(self.lobe_width_deg)
        g = np.exp(-4 * np.log(2) * off**2 / bw**2)
        return float(max(g, 10 ** (self.end_on_db / 10)))

    def scatterers(self, aspect_deg: float, rng: np.random.Generator | None = None):
        """(range offsets in m, complex amplitudes in sqrt(sigma_bs)) for one ping.

        The sum of |amplitude|^2 is the backscattering cross-section 10^(TS/10) at this aspect.
        """
        x = np.array([h[0] for h in self.highlights]) * self.length_m / 1.7
        w = np.array([h[1] for h in self.highlights])
        w = w / w.sum()
        sigma = 10 ** (self.ts_broadside_db / 10) * self.aspect_gain(aspect_deg)
        amp = np.sqrt(sigma * w)
        if self.fluctuating:
            rng = rng or np.random.default_rng()
            amp = amp * np.exp(2j * np.pi * rng.random(len(amp)))
        return x * np.cos(np.deg2rad(aspect_deg)), amp.astype(complex)
