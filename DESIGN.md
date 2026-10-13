# Pigeon · design direction

> **Source.** Transcribed on 2026-10-07 from decisions already made: `docs/PLAN.md` (the 2026-09-23 design grill: "Field-guide aesthetic", the mascot rules), the comment at the top of `site/src/styles/tokens.css`, and Jarod's choices on 2026-10-07 (the v1 field-guide look for the whole site; the birds gamify every model; Pigeon is the guide). Lines marked *inferred* were filled in by Claude to unblock work and are open: Jarod's answers to the seven direction questions override anything here.

## Identity

A field guide to model compression. Pigeon is a small, real image model with one job (*is there a bird, and where?*), and the site teaches compression by letting you shrink it, and dozens of other models, with your own hands. The mascot is the scoreboard and the guide: how much smaller a model gets decides which bird it becomes; how much accuracy it keeps decides the bird's health.

**Audience** *(inferred from PLAN.md's goal)*: curious engineers and students who want to understand compression well enough to apply it to their own models, LLMs included. Comfortable with code, not necessarily with the maths.

## Personality

Field-naturalist, exact, a little wry. It shows its working (real bytes, real photos, real source code) and is honest when something is missing. Humor lives in the birds (a dodo for breaking the uncompressed model), never in the copy's claims.

**Never:** a hype product page. No "AI-powered", no stats it can't back, no decoration that doesn't carry information.

## Palette

"A field guide on sage paper, inked in slate, with the rock pigeon's iridescent neck (teal and magenta) as the only accent pair."

| Role | Light | Dark | Use |
|---|---|---|---|
| Paper | `#eef1eb` | `#121a1e` | page |
| Card | `#f7f9f5` | `#182227` | panels, plates |
| Ink | `#1c2a33` | `#e2e8e2` | text |
| Muted | `#586970` | `#9aaaa8` | secondary text (AA on paper and card) |
| Rule | `#c9d2ca` | `#2e3c40` | hairlines, dividers |
| Teal | `#277265` | `#6cc1af` | the action color: links, primary buttons, "still works" |
| Magenta | `#924070` | `#d88ab8` | the second accent: focus ring, Latin names, the frontier line, evidence heatmaps |

Status colors (good = teal, warn `#98591a` / `#e0a15c`, bad `#a83a3a` / `#e07a7a`) mark real states only. Anything else (chart series, syntax highlighting, detection boxes) gets a named token in `tokens.css` so the palette stays countable.

## Typography

| Face | Role | Reason *(inferred)* |
|---|---|---|
| Newsreader (serif, optical sizes) | headings, species and Latin names, big numbers | The voice of a printed field guide: editorial, warm, readable at display sizes. |
| Public Sans | body and UI | Neutral, highly legible, made for public-service interfaces; it stays out of the way of the content. |
| IBM Plex Mono | code, measured numbers, status lines | Code is real code here; tabular figures for measurements. Never used as decoration or for headings. |

Labels are sentence case. No uppercase-with-tracking labels.

## Composition

- Ruled, not boxed: sections are separated by hairline rules and whitespace, like a printed guide. Cards and panels are flat (no shadows), square-cornered; buttons, inputs and selects use the one small radius (`--radius`, 4px).
- Field-guide plates: figures and birds sit on light "plates". Matplotlib figures stay on a light plate in dark mode on purpose, the way a printed plate does.
- One focal point per screen: on a method page it is the write-up; on the Coop it is the bird and its numbers.
- Content decides structure. Sections exist because content exists; "Coming next" is said plainly when it isn't there.

## Motif

The birds. Species tracks size (rock pigeon, mourning dove, common swift, barn swallow, ruby-throated hummingbird; a dodo for breaking a model without making it smaller), condition tracks accuracy (healthy, starving, skeleton). Every place a model gets smaller should end in a bird, wherever the page knows the model's storage and has a reference to measure it against; where it doesn't, there is no bird rather than a guessed one.

- **One rule for every model** (`site/src/lib/ui/birds.ts`): the species comes from the size ratio (the reference's storage over this model's), the condition from the score kept (this model's over the reference's): healthy at 95% or more, bones at 60% or less, starving in between. Pigeon in the Coop keeps its documented "still works" line (90% accuracy, 75% pointing), which lands in the same place. The rule is always shown where a bird is.
- **The life list**: every bird someone hatches (in the Coop's two cages or from a Field Guide run) is saved in their browser, the way a birder keeps a life list; each species counts once, at its best sighting. Only real, measured runs count.
- **Recorded results**: an experiment page ends in a key of the birds its table hatches, one drawing per kind of bird, measured against that experiment's teacher. Storage is the table's kbits column, or 32 bits a parameter for a dense fp32 net (`lab/compress.py` `storage_bits`); a pruned or quantized row without kbits gets none. These birds are the lab's, so they never go on a reader's life list.
- **Pigeon the guide**: Pigeon delivers hints and warnings as small margin notes, drawn in the condition that fits the message. It carries information or it isn't there; never more than one per view, never animated.

## Mood and references *(inferred)*

Sibley and Peterson field guides; museum specimen drawers; a well-kept lab notebook. References for feel, not for copying.

## Theme

Light and dark are both first-class. The site follows the system by default; a header control lets you pick System, Light or Dark, and the choice is remembered.

## Dials *(inferred)*

`Dial: ENERGY 2 / RHYTHM 2 / MOTION 1`

- ENERGY 2: a warm, characterful page (serif voice, the birds) on a calm ground.
- RHYTHM 2: consistent ruled sections, with a few deliberate breaks (the species ladder, the bird card, figure plates).
- MOTION 1: hover and focus states, plus motion that reports a real change (a progress bar filling, the bird changing species). No loops, no scroll choreography.
