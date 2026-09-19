# What "the subject" is, in RotoBench

A reference alpha says which pixels are foreground. A click says "this object". Those
are not the same question, and in Task 4 the gap cost two clips:

- **A2** is a woman and a child touching. The reference keyed both. One click at the
  most interior point of the blob can only ever select one of them.
- **A4** holds a notepad. The reference keys the notepad with the person, and the
  interior point landed on the notepad, so SAM 2 segmented a notepad and every method
  scored as a catastrophe.

Neither was a matting failure. Both were a disagreement about what the subject is.
This page fixes that definition. The oracle prompts follow it, and every method gets
the same prompts.

## The convention

1. **The subject is the key's foreground, as the compositor made it.** For Tier P that
   is the Mango team's own key. We do not redraw it.
2. **Held objects are part of the subject.** The notepad, the rifle, the device in
   someone's hand: they are in front of the screen and the key holds them.
3. **A second person is part of the subject** when they are significant (at least
   0.5% of the frame) on the prompt frame. Each person gets their own prompt object.
4. **Keyed set dressing is part of the subject too.** A desk lamp or monitor the
   compositor kept is foreground. Shots where set dressing *dominates* the key are
   either excluded (the set keys, where the "foreground" is a whole room with a
   window keyed out) or placed in a separate **stress** group that is reported apart
   from the core standings.
5. **Late entrants are designed out, not scored.** A second actor who walks in at
   frame 40 was never clicked. Each clip's 96-frame window is chosen by a stated rule:
   every significant component must descend from one on the first frame
   (`scripts/select_pro.py`). That rule rejected 04_2c outright, since its second actor
   enters at frame 104 of 111.
6. **Specks are ignored, not scored either way.** A floor marker or the edge of a flag
   that the key also holds, and that is not attached to the subject, goes in an
   `ignore/` mask. The prediction is set equal to the reference there for every
   method, so it adds no error and moves no boundary. Soft pixels *attached* to the
   subject (hair wisps, the halo) are never ignored. A first version of the rule did
   ignore them and was caught before anything was scored: it would have removed
   exactly the outer hair this benchmark exists to measure.
7. **Deliberate hold-outs disqualify a shot.** In 07_1f the compositor cut the prop
   out of the key, presumably for a CG replacement. The reference then contradicts
   rule 2, so the shot is excluded rather than scored against a truth we disagree
   with.

## The oracle prompt

Derived from frame 0 of the reference and nothing else, and frozen into each
`truth/P*/recipe.json` before any method ran on Tier P:

- **One person: one object, two clicks.** The head click is the most interior point of
  the top quarter of the component. The body click is the most interior point of the
  whole component. Task 4 showed a single click can leave SAM 2 selecting "skin only".
- **The frame edge counts as boundary** when finding "most interior". Otherwise a torso
  cut off by the bottom of frame gets clicked on the frame edge. The Task 4 oracle had
  that flaw; it happened not to bite on those clips.
- **Several things in one blob: parts pinned by hand**, one object each, named, with the
  reason recorded (`oracle_parts` in `scripts/oracle_pro.py`). This is the A2/A4
  resolution, generalised.

What a method is asked, then: produce the key's foreground, given clicks that
describe it. A score difference is a matte-quality difference, not a
guess-the-subject difference.

## What a key cannot tell you

- **Occlusion by anything that is not foreground.** If an object passes in front of
  the actor and is itself in front of the green screen, the key holds both. A key
  cannot express "the actor, minus what covers them", so that axis is represented
  only weakly, by keyed props in the stress group.
- **Interior strand transparency beyond what the keyer produced.** A professional key
  is far better than ours, but it is still a key.
