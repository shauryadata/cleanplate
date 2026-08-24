# Working with hair in CleanPlate

Written after the first real user test, where the `hair` demo shot comped over a bright
lake plate showed washed-out backlit hair and stretches of missing beard. Everything
below is measured, and the limits are stated as plainly as the wins.

## The short version

1. **Turn on "High quality (hair)".** It is now on by default. On the one clip where we
   hold reference alpha for this actor it recovers **−10.6 %** of hair-region error for
   about **+0.07 s/frame** at 960 px. It is the single biggest lever.
2. **Click the subject's distinct parts, not just the torso** — hair, face, body. Worth
   a further **−1.1 points** on hair (−11.7 % total). Small, but free.
3. **Comp against a flat colour before you trust an edge.** A soft, displaced edge is
   invisible over a dark plate and glaring over a bright one. The user's lake plate has
   median luma 179; the same matte looked fine over the ToS interiors.
4. **Do not expect strands.** See the limits below.

## What actually helps, measured

On `B1_tos_greenscreen_hair` — the Tears of Steel green-screen plate of the *same actor*
as the `hair` shot, and the only clip where we hold reference alpha for this hair.
Tier B: keyed reference, not gospel.

| Config | hair MAD ↓ | Grad ↓ | Boundary-F ↑ |
|---|---|---|---|
| One click, MatAnyone v1 (what the user ran) | 12.41 | 0.291 | 0.8825 |
| One click + High quality | 11.10 | 0.275 | 0.9008 |
| Well-formed clicks + High quality | **10.96** | **0.269** | **0.9048** |

Total recoverable: **−11.7 %**. That is the honest ceiling on this shot today — most of
it from the toggle, very little from clicking better.

## How to click a person with hair

Place one positive click in each region that looks *different* to a segmentation model:

- the **hair mass** (not the wispy edge — the solid part)
- the **face**, below the brow
- the **torso**

On the `hair` demo shot those are `(628, 96)`, `(700, 250)` and `(660, 320)`.

Two things not to do, both learned the hard way:

- **Do not stack clicks.** Two positive points a few pixels apart do not reinforce each
  other; they change which mask hypothesis SAM 2 selects. A duplicate click 1 px away
  once cut a matte by 34 %. The app warns within 12 px.
- **Do not add a lone corrective click on a later frame.** Conditioning frames are
  global in SAM 2 — a click at frame 67 changes frame 24. If you correct a later frame,
  put clicks on *every* distinct part of the subject in that frame, not just the bit you
  are fixing. The Changes tab shows what moved.

## Limits, stated honestly

- **No strand transparency.** The matte's soft band is a few pixels of anti-aliased
  edge. Measured against reference alpha this is roughly what the truth looks like too
  (median 2.0–2.8 px), so the matte is not obviously "too hard" — but neither tier of
  our ground truth contains much strand detail, so this cannot currently be scored
  properly. See docs/TASK4_REPORT.md.
- **Bright plates expose everything.** The identical matte scored the same over a dark
  plate and a bright one; only the *comp* changed. If your background is bright and
  low-contrast, budget for edge work.
- **The beard/jaw dropout is not fixable by clicking.** See
  [KNOWN_ISSUES.md](KNOWN_ISSUES.md#face-edge-dropout-on-low-contrast-shadowed-jaw).
