# Pigeon: the website

An Astro static site that teaches model compression. You shrink Pigeon, a small bird-spotting image model, in your browser and watch what it costs; the lab's write-ups, lessons and experiments are built into pages at build time.

The design direction is in [`../DESIGN.md`](../DESIGN.md). It is published to GitHub Pages under `/pigeon/` by `.github/workflows/pages.yml`.

## Run it

Node 22.12 or newer.

```sh
npm install
npm run dev        # http://localhost:4321/pigeon/
npm run build      # static site in dist/
npm run preview    # serve dist/
npm test           # unit tests (vitest)
```

## The lab data

The Field Guide, Flight School and Results pages import the lab's content at build time from `src/data/lab/` (git-ignored). Build it from the repo root, not from `site/`:

```sh
# Pages only: method, lesson and experiment data plus the figures. Takes about a second.
python -m lab.web.build_static --out site/public/lab --data-out site/src/data/lab --no-frontend --data-only

# Everything, including what Run and Check need in the browser (Pyodide, the lab bundle,
# the trained models). Trains the shipped models first, so it takes a few minutes.
python -m lab.web.build_static --out site/public/lab --data-out site/src/data/lab --no-frontend
```

Without that data the build stops and prints the command above.

## Where things live

| Path | What it is |
|---|---|
| `src/pages/index.astro` | Home |
| `src/pages/coop/` | The Coop: the compression sandbox (Pigeon at `/coop/`, YOLOv8n at `/coop/yolo/`) |
| `src/pages/school/` | Flight School: the course and the coding lessons |
| `src/pages/guide/` | Field Guide: one page per method |
| `src/pages/results/` | Results: the lab's experiments and report |
| `src/pages/lab/` | Old `/lab/` links: redirects them to the pages above |
| `src/pages/dev/` | Unlinked development checks (mascot art, engine parity) |
| `src/layouts/Base.astro` | The header, nav and theme control every page shares |
| `src/styles/` | `tokens.css` (colors, type and spacing scales), `page.css` (the page kit), `prose.css` (long-form text) |
| `src/lib/engine/` | The in-browser model: ONNX Runtime in a worker, scoring on the test photos |
| `src/lib/compress/` | Quantization, pruning and palettes applied to the weights |
| `src/lib/detector/` | The Coop's YOLOv8n: TF.js on WebGL in a worker, ported from the cockpit (forward pass, NMS, bird AP@50, masking, precision lab) |
| `src/lib/ui/` | The Coop's mascot, recipe editor, chart and specimen drawer |
| `src/lib/lab/` | Build-time loaders for the lab data, its markdown, and the in-browser Python client |
| `public/model/`, `public/testset/` | Pigeon's weights and the test photos the Coop scores on |
| `public/coop/yolov8n/` | YOLOv8n's fp16 weights and manifest, written by `cockpit/build_coop.py` (needs the project venv with torch and ultralytics, so the files are committed) |
