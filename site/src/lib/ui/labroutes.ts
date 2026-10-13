/**
 * Old lab URLs. The lab was a hash-routed app (at the site root, then at /lab/)
 * with routes like #/method/<id>. Its pages are now the Field Guide, Flight
 * School and Results. `labRedirectScript` builds a small inline script that
 * sends an old URL to its new page before the rest of the page renders.
 *
 *   #/method/<id>     -> guide/<id>/
 *   #/family/<f>      -> guide/#<f>
 *   #/learn           -> school/#build
 *   #/learn/<id>      -> school/<id>/
 *   #/results         -> results/
 *   #/results/<name>  -> results/<name>/
 *
 * Ids are checked against the lab data, so a stale id lands on the section's
 * index instead of a missing page.
 */

export interface LabIds {
  methods: string[];
  families: string[];
  lessons: string[];
  experiments: string[];
}

export interface RedirectOptions {
  /** The site's base URL, ending in "/". */
  base: string;
  /** Where a hash that is not a lab route goes, relative to `base`; null stays on the page. */
  fallback: string | null;
  /** Only act on "#/..." hashes, so in-page anchors such as #main are left alone. */
  strict: boolean;
}

const json = (v: unknown) => JSON.stringify(v).replace(/</g, "\\u003c");

/**
 * The script runs once as the page loads, and again on `hashchange`: pasting an
 * old link while already on the page changes only the hash, without a reload.
 */
export function labRedirectScript(ids: LabIds, o: RedirectOptions): string {
  return `(function (win, ids, base, fallback, strict) {
  function go() {
    var h = win.location.hash;
    if (strict && h.indexOf("#/") !== 0) return;
    var p = h.replace(/^#\\/?/, "").split("/").filter(Boolean);
    var id = "";
    try { id = p[1] ? decodeURIComponent(p[1]) : ""; } catch (e) { id = ""; }
    var has = function (list) { return id !== "" && list.indexOf(id) !== -1; };
    var to = fallback;
    if (p[0] === "method") to = has(ids.methods) ? "guide/" + id + "/" : "guide/";
    else if (p[0] === "family") to = has(ids.families) ? "guide/#" + id : "guide/";
    else if (p[0] === "learn") to = has(ids.lessons) ? "school/" + id + "/" : "school/#build";
    else if (p[0] === "results") to = has(ids.experiments) ? "results/" + id + "/" : "results/";
    if (to !== null) win.location.replace(base + to);
  }
  go();
  win.addEventListener("hashchange", go);
})(window, ${json(ids)}, ${json(o.base)}, ${json(o.fallback)}, ${json(o.strict)});`;
}
