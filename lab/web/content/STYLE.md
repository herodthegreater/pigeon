# Writing guide for method pages

Each file is `lab/web/content/methods/<method id>.md`. The website renders it
under the method's title, summary, parameter panel and source code (those come
from `lab/methods.py`, so don't repeat the title, summary or paper list).

## Audience
Someone who knows Python and roughly what a neural network is, and wants to
understand *and be able to build* each algorithm. Explain every term the first
time it appears on the page. Prefer one concrete example over an abstraction.

## Required sections (exact headings, in this order; skip ones marked optional if empty)

    ## The idea
    2-5 sentences. What problem it solves, the one insight that makes it work.

    ## How it works
    Numbered steps of the algorithm, precise enough to implement from.
    Use inline code for formulas: `p = softmax(z / T)`. Plain text/unicode math
    only (no LaTeX). A short ```python block of pseudo-code is fine.

    ## The code, line by line
    Walk through the functions listed in the method's `code` field in
    lab/methods.py (read them!). Refer to real names and lines: "`err = (w - q) / U[j, j]`
    is the rounding error scaled by...". This is the most important section.

    ## What the simulations show            (optional)
    Only real numbers: from lab/results/<experiment>.json for the experiments
    listed in the method's `experiments` field, or from running the method
    yourself: python3 -c "from lab.methods import REGISTRY as R; print(R['<id>'].execute({}))"
    (pass params as a dict to try settings; most runs take seconds).
    Numbers quoted from a page's Run must be measured in Pyodide, the browser's
    runtime, with the shipped models in lab/models
    (`node lab/web/run_in_pyodide.cjs site/public/lab runs.json`): a training
    run can end a few test examples apart on another BLAS. Quote the four
    decimals the run returns (the page's result table rounds them to three).
    lab/tests/test_quotes.py re-runs a sample of the quotes that train
    nothing; add a case when you quote a new one.
    Never invent or round a number into something it isn't. Say what the toy
    world can't tell us.

    ## When it breaks
    Bullets: failure modes, edge cases, wrong assumptions.

    ## Try this                              (runnable methods only)
    2-4 concrete experiments using the parameter labels on the page,
    each with what to look for.

    ## Going further
    How it scales to real models; for `reference`/`cockpit` methods, what you
    would need to run it and which file in this repo is the closest analogue.

## Markdown subset the site renders
`##`/`###` headings, paragraphs, `-` and `1.` lists (one level), ```python
fenced code, `inline code`, **bold**, *italic*, [links](https://...), and
simple pipe tables. Nothing else (no HTML, no nested lists, no images, no LaTeX).

## Voice
Plain, specific, calm. Short sentences. No hype or filler ("powerful",
"cutting-edge", "seamless", "delve", "it's important to note", "in summary",
"let's dive in"). No emojis. No rhetorical questions as headings. Don't
end with a recap. Every sentence should teach something or point at code.
Length: 400-900 words for runnable methods, 250-500 for reference ones.
