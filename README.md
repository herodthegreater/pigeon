# Pigeon

**Learn model compression by doing it.** Pigeon takes a real image classifier, teaches it one job (*is there a bird, and where?*) and lets you shrink it with every major compression technique, right in your browser, while it shows you exactly what each one costs.

The mascot is the scoreboard: shrink the model and the chonky pigeon slims into a dove, a swift, a swallow, a hummingbird. Push too far and it starves, or turns to bones.

> **Status:** one site, assembled from three pieces that grew up separately (below). The plan and every design decision live in [`docs/PLAN.md`](docs/PLAN.md). The live site is at <https://herodthegreater.github.io/pigeon/>.

## What's in the repo

| Folder | What it is |
|---|---|
| [`site/`](site/) | The web app (Astro). **The Coop** is open: a sandbox where you quantize, prune and palettize the real Pigeon model in your browser and watch accuracy, pointing score and true size move live. **Flight School** and the **Field Guide** are posted as syllabus and list; their content is moving in from the lab. |
| [`lab/`](lab/) | The simulation lab: 43 compression methods with the real code, each runnable on a toy network with a known best answer, ten build-it-yourself coding lessons, and 28 experiments. Pure numpy; on the site it runs in the browser through Pyodide. See [`lab/README.md`](lab/README.md). |
| [`cockpit/`](cockpit/) | The YOLOv8n reverse-training cockpit: carve a general detector down to a bird detector, with a precision lab, stackable recipes, prune-then-regrow, and a browser edition. See [`cockpit/README.md`](cockpit/README.md). |
| [`train/`](train/) | PyTorch pipeline that makes Pigeon the model: data, specialization, evaluation, ONNX export. |
| [`docs/`](docs/) | The plan and the measured findings. |

## How the site is built

`.github/workflows/pages.yml` publishes to GitHub Pages on every push to `main` that touches `site/` or `lab/`: it trains and pre-renders the lab into `site/public/lab/`, builds the Astro site around it, and deploys `site/dist/`. The lab is served under `/lab/` until its pages have moved into the site proper.

## Quickstart

The site:

```bash
cd site && npm install && npm run dev      # http://localhost:4321/pigeon/
```

The lab, served locally with a Python backend (or pre-rendered for the site with `python -m lab.web.build_static --out site/public/lab`):

```bash
pip install numpy matplotlib fastapi uvicorn
python -m lab.web                          # http://127.0.0.1:8100
```

Pigeon the model:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r train/requirements.txt
python -m train.data        # download + prepare the Open Images bird / no-bird set
python -m train.train       # specialize MobileNetV3-Small into Pigeon
python -m train.evaluate    # accuracy, pointing score, size
```

The cockpit has its own requirements and datasets; its README has the steps.

## Data

Images and box annotations come from [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) (annotations CC-BY 4.0; images listed as CC-BY 2.0 by their authors). Per-image attribution ships with the site's test set. The cockpit's eval set is a COCO subset; see its README.
