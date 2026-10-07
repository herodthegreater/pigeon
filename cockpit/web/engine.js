/* pigeon web engine -- the real YOLOv8n, running in this browser tab (TF.js).
 *
 * A line-for-line port of what server.py + harness/ do in Python:
 *   forward pass (fused convs, C2f, SPPF, Detect + DFL decode), Ultralytics-style
 *   NMS, the single-class AP@50 scorer, channel masking, the precision lab's
 *   fake-quantization (tensor/channel/group scales, INT/FP grids, max/MSE range),
 *   storage accounting, live-channel compute, scans and the weight microscope.
 * Every accuracy number it returns is measured, here, on real photos.
 */
const ENGINE = (() => {
  const BATCH = 4, CONF_EVAL = 0.001, CONF_SHOW = 0.25, IOU_NMS = 0.7, MAX_DET = 300, MAX_NMS = 30000, BIRD = 14;
  const SCAN_N = 12, PROBE_RES = 64, SCALE_BITS = 16, NATIVE_BITS = 16;
  const FP_FORMATS = {8: [4, 3], 7: [3, 3], 6: [3, 2], 5: [2, 2], 4: [2, 1], 3: [1, 1], 2: [1, 0]};
  const ALPHAS = Array.from({length: 29}, (_, i) => Math.fround(0.3 + i * (0.7 / 28)));

  let M = null, S = 416;
  const NAT = {};        // conv name -> native {w, b, O, cols, k, stride, pad, stage, order}
  const GPU = {};        // conv name -> {w: tf.Tensor4D [kh,kw,I,O], b: tf.Tensor1D, key}
  let PHOTOS = [], EVAL = [], SCAN = [], PREVIEW = [];
  let ANCH = null, STRD = null, PROJ = null;
  let NATIVE = null;     // {map, mapScan, cost, macs}
  let probe = null;      // live-channel accounting during a probe forward
  const cache = new Map();   // recipe key -> {preds: Map(photoIdx -> boxes)}
  let queue = Promise.resolve();
  const lock = fn => { const p = queue.then(fn, fn); queue = p.catch(() => {}); return p; };

  /* ------------------------------ loading ------------------------------ */
  const HALF = (() => { const t = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) { const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
      t[h] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024); }
    return t; })();

  async function fetchBytes(url, onProgress) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`could not load ${url} (${r.status})`);
    const total = +r.headers.get('content-length') || 0;
    if (!r.body || !total) return new Uint8Array(await r.arrayBuffer());
    const out = new Uint8Array(total), rd = r.body.getReader(); let got = 0;
    for (;;) { const {done, value} = await rd.read(); if (done) break; out.set(value, got); got += value.length; onProgress && onProgress(got / total); }
    return out;
  }
  const loadImage = src => new Promise((ok, bad) => { const im = new Image(); im.onload = () => ok(im); im.onerror = () => bad(new Error('could not load ' + src)); im.src = src; });
  const stageOf = name => { const p = name.split('.'); return p[0] === 'model' && /^\d+$/.test(p[1]) ? 'model.' + p[1] : null; };

  async function load(onStep = () => {}) {
    onStep('model', 0);
    M = await (await fetch('model.json')).json(); S = M.imgsz;
    const b64 = new TextDecoder().decode(await fetchBytes('weights.txt', f => onStep('weights', f)));
    const bin = atob(b64.trim()), bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const u16 = new Uint16Array(bytes.buffer, 0, bytes.byteLength / 2);
    for (const c of M.convs) {
      const [O, I, kh, kw] = c.shape, cols = I * kh * kw, w = new Float32Array(O * cols), b = new Float32Array(O);
      for (let i = 0; i < w.length; i++) w[i] = HALF[u16[c.w + i]];
      for (let i = 0; i < O; i++) b[i] = HALF[u16[c.b + i]];
      const l1 = new Float64Array(O);
      for (let o = 0; o < O; o++) { let s = 0; for (let j = 0; j < cols; j++) s += Math.abs(w[o * cols + j]); l1[o] = s; }
      const order = Uint32Array.from([...l1.keys()].sort((a, z) => l1[a] - l1[z]));
      NAT[c.name] = {w, b, O, I, cols, kh, kw, stride: c.stride, pad: c.pad, stage: stageOf(c.name), order};
    }
    onStep('photos', 0);
    const cv = document.createElement('canvas'); cv.width = cv.height = S;
    const ctx = cv.getContext('2d', {willReadFrequently: true});
    for (let i = 0; i < M.photos.length; i++) {
      const p = M.photos[i], im = await loadImage(p.file);
      const r = Math.min(S / p.w, S / p.h), nw = Math.round(p.w * r), nh = Math.round(p.h * r);
      const left = Math.round((S - nw) / 2 - 0.1), top = Math.round((S - nh) / 2 - 0.1);
      ctx.fillStyle = 'rgb(114,114,114)'; ctx.fillRect(0, 0, S, S); ctx.drawImage(im, left, top, nw, nh);
      const rgba = ctx.getImageData(0, 0, S, S).data, px = new Uint8Array(S * S * 3);
      for (let j = 0, k = 0; j < rgba.length; j += 4) { px[k++] = rgba[j]; px[k++] = rgba[j + 1]; px[k++] = rgba[j + 2]; }
      PHOTOS.push({...p, px, r, left, top});
      onStep('photos', (i + 1) / M.photos.length);
    }
    EVAL = PHOTOS.map((p, i) => p.eval ? i : -1).filter(i => i >= 0);
    SCAN = EVAL.slice(0, SCAN_N);
    PREVIEW = PHOTOS.map((p, i) => p.preview ? i : -1).filter(i => i >= 0);

    onStep('gpu', 0);
    try { await tf.setBackend('webgl'); } catch (e) { /* falls back below */ }
    await tf.ready();
    if (tf.getBackend() !== 'webgl') { await tf.setBackend('cpu'); await tf.ready(); }
    const an = [], st = [];
    for (const s of M.graph.find(n => n.type === 'Detect').strides) {
      const h = S / s; for (let y = 0; y < h; y++) for (let x = 0; x < h; x++) { an.push(x + 0.5, y + 0.5); st.push(s); }
    }
    ANCH = tf.keep(tf.tensor2d(an, [st.length, 2])); STRD = tf.keep(tf.tensor2d(st, [st.length, 1]));
    PROJ = tf.keep(tf.range(0, 16, 1, 'float32'));
    applyRecipe(nativeRecipe());

    onStep('native', 0);
    const preds = await detect(EVAL, CONF_EVAL, f => onStep('native', f));
    const nk = keyOf(nativeRecipe()); cache.set(nk, {preds: new Map(EVAL.map((i, j) => [i, preds[j]]))});
    NATIVE = {map: ap(EVAL, cache.get(nk).preds), mapScan: ap(SCAN, cache.get(nk).preds)};
    NATIVE.cost = probeCost();
    NATIVE.macs = NATIVE.cost.macs * (S / PROBE_RES) ** 2;
    current = nativeRecipe();
    return info();
  }

  /* ------------------------------ forward ------------------------------ */
  function convRaw(x, name) {
    const g = GPU[name], c = NAT[name];
    const y = tf.add(tf.conv2d(x, g.w, c.stride, c.pad), g.b);
    if (probe) {
      const live = tf.tidy(() => tf.sum(tf.abs(x), [0, 1, 2]).dataSync());
      let liveIn = 0; for (const v of live) if (v > 0) liveIn++;
      probe.push({name, liveIn, oh: y.shape[1], ow: y.shape[2]});
    }
    return y;
  }
  const silu = t => tf.mul(t, tf.sigmoid(t));
  const Conv = (x, name) => silu(convRaw(x, name + '.conv'));
  function C2f(x, p, n) {
    const [a, b] = tf.split(Conv(x, p + '.cv1'), 2, 3), ys = [a, b];
    let last = b;
    for (let k = 0; k < n.n; k++) {
      let h = Conv(Conv(last, `${p}.m.${k}.cv1`), `${p}.m.${k}.cv2`);
      if (n.add[k]) h = tf.add(last, h);
      ys.push(h); last = h;
    }
    return Conv(tf.concat(ys, 3), p + '.cv2');
  }
  function SPPF(x, p, k) {
    const y0 = Conv(x, p + '.cv1'), y1 = tf.maxPool(y0, k, 1, 'same'), y2 = tf.maxPool(y1, k, 1, 'same'), y3 = tf.maxPool(y2, k, 1, 'same');
    return Conv(tf.concat([y0, y1, y2, y3], 3), p + '.cv2');
  }
  function forward(x) {
    const outs = []; let y = x;
    for (const n of M.graph) {
      const p = 'model.' + n.i;
      const inp = n.f === -1 ? y : Array.isArray(n.f) ? n.f.map(j => j === -1 ? y : outs[j]) : outs[n.f];
      if (n.type === 'Conv') y = Conv(inp, p);
      else if (n.type === 'C2f') y = C2f(inp, p, n);
      else if (n.type === 'SPPF') y = SPPF(inp, p, n.k);
      else if (n.type === 'Upsample') y = tf.image.resizeNearestNeighbor(inp, [inp.shape[1] * 2, inp.shape[2] * 2]);
      else if (n.type === 'Concat') y = tf.concat(inp, 3);
      else if (n.type === 'Detect') {
        const bx = [], cl = [];
        inp.forEach((t, j) => {
          const b = convRaw(Conv(Conv(t, `${p}.cv2.${j}.0`), `${p}.cv2.${j}.1`), `${p}.cv2.${j}.2`);
          const c = convRaw(Conv(Conv(t, `${p}.cv3.${j}.0`), `${p}.cv3.${j}.1`), `${p}.cv3.${j}.2`);
          const [B, h, w] = b.shape;
          bx.push(b.reshape([B, h * w, 4 * n.reg_max])); cl.push(c.reshape([B, h * w, n.nc]));
        });
        return [tf.concat(bx, 1), tf.concat(cl, 1)];
      }
      outs.push(y);
    }
  }

  /* --------------------- detect: decode + NMS + rescale --------------------- */
  function nms(c) {
    c.sort((a, z) => z[4] - a[4]); if (c.length > MAX_NMS) c.length = MAX_NMS;
    const keep = [], dead = new Uint8Array(c.length);
    for (let i = 0; i < c.length && keep.length < MAX_DET; i++) {
      if (dead[i]) continue; const a = c[i]; keep.push(a);
      const aa = (a[2] - a[0]) * (a[3] - a[1]);
      for (let j = i + 1; j < c.length; j++) {
        if (dead[j]) continue; const b = c[j];
        const iw = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), ih = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
        if (iw <= 0 || ih <= 0) continue;
        const inter = iw * ih, u = aa + (b[2] - b[0]) * (b[3] - b[1]) - inter;
        if (inter / u > IOU_NMS) dead[j] = 1;
      }
    }
    return keep;
  }
  async function detect(idxs, conf, onProgress) {
    const out = [];
    for (let s = 0; s < idxs.length; s += BATCH) {
      const batch = idxs.slice(s, s + BATCH), B = batch.length;
      const flat = new Uint8Array(B * S * S * 3); batch.forEach((i, j) => flat.set(PHOTOS[i].px, j * S * S * 3));
      const [xyxy, mx, am] = tf.tidy(() => {
        const x = tf.div(tf.tensor4d(flat, [B, S, S, 3], 'int32').toFloat(), 255);
        const [box, cls] = forward(x), A = box.shape[1];
        const d = tf.sum(tf.mul(tf.softmax(box.reshape([B, A, 4, 16])), PROJ), 3);
        const lt = d.slice([0, 0, 0], [B, A, 2]), rb = d.slice([0, 0, 2], [B, A, 2]);
        return [tf.concat([tf.mul(tf.sub(ANCH, lt), STRD), tf.mul(tf.add(ANCH, rb), STRD)], 2), tf.max(cls, 2), tf.argMax(cls, 2)];
      });
      const [bx, mv, av] = await Promise.all([xyxy.data(), mx.data(), am.data()]);
      tf.dispose([xyxy, mx, am]);
      const A = mv.length / B;
      batch.forEach((pi, j) => {
        const p = PHOTOS[pi], cand = [];
        for (let a = 0; a < A; a++) {
          const k = j * A + a; if (av[k] !== BIRD) continue;
          const cf = 1 / (1 + Math.exp(-mv[k])); if (cf <= conf) continue;
          cand.push([bx[k * 4], bx[k * 4 + 1], bx[k * 4 + 2], bx[k * 4 + 3], cf]);
        }
        out.push(nms(cand).map(b => [
          Math.min(Math.max((b[0] - p.left) / p.r, 0), p.w), Math.min(Math.max((b[1] - p.top) / p.r, 0), p.h),
          Math.min(Math.max((b[2] - p.left) / p.r, 0), p.w), Math.min(Math.max((b[3] - p.top) / p.r, 0), p.h), b[4]]));
      });
      onProgress && onProgress(Math.min(s + B, idxs.length) / idxs.length);
      await tf.nextFrame();
    }
    return out;
  }

  /* -------------------- AP@50, the BirdmAPEvaluator port -------------------- */
  function iou(a, b) {
    const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const i = iw * ih; return i / Math.max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9);
  }
  function ap(idxs, preds) {
    const scores = [], tp = []; let nGt = 0;
    for (const i of idxs) {
      const gt = PHOTOS[i].gt, got = preds.get(i) || [], matched = new Uint8Array(gt.length); nGt += gt.length;
      for (const b of got.slice().sort((a, z) => z[4] - a[4])) {
        scores.push(b[4]);
        if (gt.length) {
          let j = 0, best = -1; gt.forEach((g, k) => { const v = iou(b, g); if (v > best) { best = v; j = k; } });
          if (best >= 0.5 && !matched[j]) { matched[j] = 1; tp.push(1); continue; }
        }
        tp.push(0);
      }
    }
    if (!nGt || !scores.length) return 0;
    const order = scores.map((s, i) => i).sort((a, z) => scores[z] - scores[a]);
    const mrec = [0], mpre = [0]; let ctp = 0, cfp = 0;
    for (const i of order) { tp[i] ? ctp++ : cfp++; mrec.push(ctp / nGt); mpre.push(ctp / Math.max(ctp + cfp, 1e-9)); }
    mrec.push(1); mpre.push(0);
    for (let i = mpre.length - 1; i > 0; i--) mpre[i - 1] = Math.max(mpre[i - 1], mpre[i]);
    let s = 0; for (let i = 0; i < mrec.length - 1; i++) if (mrec[i + 1] !== mrec[i]) s += (mrec[i + 1] - mrec[i]) * mpre[i + 1];
    return s;
  }

  /* ------------------ precision lab (harness/precision.py) ------------------ */
  function fpGrid(bits) {
    const [e, m] = FP_FORMATS[bits], bias = e > 1 ? (1 << (e - 1)) - 1 : 0, vals = new Set();
    for (let ex = 0; ex < 1 << e; ex++) for (let man = 0; man < 1 << m; man++)
      vals.add(ex === 0 ? (man / (1 << m)) * 2 ** (1 - bias) : (1 + man / (1 << m)) * 2 ** (ex - bias));
    const g = Float32Array.from([...vals].sort((a, z) => a - z)), top = g[g.length - 1];
    return Array.from(g, v => Math.fround(v / top));  // float32, as torch does
  }
  const gridCache = {};
  function signedGrid(bits) {
    if (!gridCache[bits]) { const g = fpGrid(bits), sg = [...g.slice(1).reverse().map(v => -v), ...g];
      gridCache[bits] = {sg, mids: sg.slice(1).map((v, i) => Math.fround(Math.fround(v + sg[i]) / 2))}; }
    return gridCache[bits];
  }
  const rhe = x => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : r; };
  // float32 at every step (Math.fround), so rounding ties fall exactly where torch puts them
  function snap(x, scale, bits, levels) {
    const f = Math.fround;
    if (levels === 'fp') {
      const {sg, mids} = signedGrid(bits), v = Math.min(Math.max(f(x / scale), -1), 1);
      let lo = 0, hi = mids.length; while (lo < hi) { const mid = (lo + hi) >> 1; if (mids[mid] < v) lo = mid + 1; else hi = mid; }
      return f(sg[lo] * scale);
    }
    const q = (1 << (bits - 1)) - 1, step = f(scale / q);
    return f(Math.min(Math.max(rhe(f(x / step)), -q), q) * step);
  }
  // one scale-sharing row: src[start..end), padded (with zeros) to length L
  function quantRow(src, dst, start, end, L, bits, levels, clip) {
    if (bits === 1) { let s = 0; for (let i = start; i < end; i++) s += Math.abs(src[i]); s = Math.fround(s / L);
      for (let i = start; i < end; i++) dst[i] = src[i] >= 0 ? s : -s; return; }
    let amax = 1e-12; for (let i = start; i < end; i++) amax = Math.max(amax, Math.abs(src[i]));
    let scale = amax;
    if (clip === 'mse') {
      let bestErr = Infinity;
      for (const a of ALPHAS) { const sc = Math.fround(amax * a); let err = 0;
        for (let i = start; i < end; i++) { const d = snap(src[i], sc, bits, levels) - src[i]; err += d * d; }
        if (err < bestErr) { bestErr = err; scale = sc; } }
    }
    for (let i = start; i < end; i++) dst[i] = snap(src[i], scale, bits, levels);
  }
  function rowsOf(O, cols, gran) {  // [start, end, paddedLength] for each scale row
    const rows = [];
    if (gran === 'tensor') rows.push([0, O * cols, O * cols]);
    else { const g = gran === 'channel' ? cols : Math.min(+gran, cols);
      for (let o = 0; o < O; o++) for (let s = 0; s < cols; s += g) rows.push([o * cols + s, o * cols + Math.min(s + g, cols), g]); }
    return rows;
  }
  function quantize(w, O, cols, bits, gran, levels, clip) {
    if (bits >= NATIVE_BITS) return w;
    if (levels === 'fp' && !FP_FORMATS[bits]) levels = 'int';
    const out = new Float32Array(w.length);
    for (const [s, e, L] of rowsOf(O, cols, gran)) quantRow(w, out, s, e, L, bits, levels, clip);
    return out;
  }
  const nScales = (O, cols, gran) => gran === 'tensor' ? 1 : gran === 'channel' ? O : O * Math.ceil(cols / +gran);

  /* ----------------------------- recipes ----------------------------- */
  const nativeRecipe = () => ({masks: {}, quant: {bits: 16, granularity: 'channel', levels: 'int', clip: 'max', stage_bits: {}}});
  let current = null;
  function convSetting(name, r) {
    const c = NAT[name], q = r.quant, frac = +(r.masks[c.stage] || 0);
    const sb = q.stage_bits || {}, bits = sb[c.stage] != null ? +sb[c.stage] : +q.bits;
    return {frac, bits, key: bits >= NATIVE_BITS ? `${frac}|16` : `${frac}|${bits}|${q.granularity}|${q.levels}|${q.clip}`};
  }
  function keyOf(r) {
    const q = r.quant, m = Object.entries(r.masks || {}).filter(([, v]) => v > 0).sort();
    const sb = Object.entries(q.stage_bits || {}).sort();
    return JSON.stringify([m, q.bits, q.granularity, q.levels, q.clip, sb]);
  }
  function applyRecipe(r) {
    let n = 0, wb = 0, sb = 0;
    for (const name in NAT) {
      const c = NAT[name], {frac, bits, key} = convSetting(name, r);
      if (!GPU[name] || GPU[name].key !== key) {
        let w = c.w, b = c.b;
        if (frac > 0) {
          const k = Math.min(Math.round(frac * c.O), c.O - 1);
          if (k > 0) { w = w.slice(); b = b.slice();
            for (let i = 0; i < k; i++) { const o = c.order[i]; w.fill(0, o * c.cols, (o + 1) * c.cols); b[o] = 0; } }
        }
        w = quantize(w, c.O, c.cols, bits, r.quant.granularity, r.quant.levels, r.quant.clip);
        let liveOut = 0;  // output channels that still carry anything (weights or bias)
        for (let o = 0; o < c.O; o++) { let alive = b[o] !== 0;
          for (let j = o * c.cols, e = j + c.cols; !alive && j < e; j++) alive = w[j] !== 0;
          if (alive) liveOut++; }
        const nw = tf.tidy(() => tf.tensor4d(w, [c.O, c.I, c.kh, c.kw]).transpose([2, 3, 1, 0]));
        if (GPU[name]) tf.dispose([GPU[name].w, GPU[name].b]);
        GPU[name] = {w: tf.keep(nw), b: tf.keep(tf.tensor1d(b)), key, liveOut};
      }
      const cnt = c.O * c.cols; n += cnt; wb += cnt * Math.min(bits, NATIVE_BITS);
      if (bits < NATIVE_BITS) sb += nScales(c.O, c.cols, r.quant.granularity) * SCALE_BITS;
    }
    return {weights: n, weight_bits: wb, scale_bits: sb, eff_bits: (wb + sb) / Math.max(n, 1)};
  }
  function probeCost() {  // MACs + weights of only the live channels (harness/mask.py active_cost)
    probe = [];
    tf.tidy(() => forward(tf.randomUniform([1, PROBE_RES, PROBE_RES, 3])));
    let macs = 0, params = 0;
    for (const p of probe) {
      const c = NAT[p.name], liveOut = GPU[p.name].liveOut, k = c.kh * c.kw;
      params += liveOut * p.liveIn * k; macs += p.oh * p.ow * liveOut * p.liveIn * k;
    }
    probe = null;
    return {macs, params};
  }

  /* ------------------------------ public API ------------------------------ */
  function info() {
    return {stages: M.prune_stages, precision_stages: M.precision_stages, head: M.head, imgsz: S,
      native: {map: +NATIVE.map.toFixed(4), params_m: +(M.params / 1e6).toFixed(3), flops_g: +(NATIVE.macs / 1e9).toFixed(3),
               size_mb: +(M.params * 2 / 1e6).toFixed(2)},
      eval_images: EVAL.length, scan_images: SCAN.length, previews: PREVIEW.length, device: tf.getBackend(), recorded: M.recorded};
  }
  function report(r, stats, m, native, cost) {
    const flopKeep = cost.macs / NATIVE.cost.macs, paramKeep = cost.params / NATIVE.cost.params;
    const other = M.params - stats.weights;
    const sizeB = ((stats.weight_bits + stats.scale_bits) * paramKeep + other * 16) / 8;
    return {map: +m.toFixed(4), map_pct: +(100 * m / Math.max(native, 1e-9)).toFixed(1),
      size_mb: +(sizeB / 1e6).toFixed(2), size_pct: +(100 * sizeB / (M.params * 2)).toFixed(1),
      eff_bits: +stats.eff_bits.toFixed(2), flops_g: +(NATIVE.macs / 1e9 * flopKeep).toFixed(3), flops_pct: +(100 * flopKeep).toFixed(1)};
  }
  async function scoreOn(r, idxs, onProgress) {
    const key = keyOf(r); let e = cache.get(key); if (!e) { e = {preds: new Map()}; cache.set(key, e); }
    const stats = applyRecipe(r), todo = idxs.filter(i => !e.preds.has(i));
    if (todo.length) { const got = await detect(todo, CONF_EVAL, onProgress); todo.forEach((i, j) => e.preds.set(i, got[j])); }
    if (!e.cost) e.cost = Object.keys(r.masks || {}).some(k => r.masks[k] > 0) ? probeCost() : NATIVE.cost;
    return {stats, m: ap(idxs, e.preds), cost: e.cost};
  }
  const measure = (r, {fast = false, commit = true, onProgress} = {}) => lock(async () => {
    const {stats, m, cost} = await scoreOn(r, fast ? SCAN : EVAL, onProgress);
    if (commit) current = JSON.parse(JSON.stringify(r));
    return report(r, stats, m, fast ? NATIVE.mapScan : NATIVE.map, cost);
  });
  const sensitivity = (probeFrac = 0.10, onProgress) => lock(async () => {
    const out = [];
    for (const [i, t] of M.prune_stages.entries()) {
      const r = nativeRecipe(); r.masks[t.stage] = probeFrac;
      const {m} = await scoreOn(r, SCAN);
      const drop = Math.max(0, (NATIVE.mapScan - m) / Math.max(NATIVE.mapScan, 1e-9));
      out.push({stage: t.stage, flop_frac: t.flop_frac, channels: t.channels, sensitivity: +drop.toFixed(3), value: +(t.flop_frac / (drop + 0.05)).toFixed(3)});
      onProgress && onProgress((i + 1) / M.prune_stages.length, t.stage);
    }
    return {probe: probeFrac, native_map: +NATIVE.mapScan.toFixed(4), stages: out};
  });
  const precisionScan = (q, onProgress) => lock(async () => {
    const out = [];
    for (const [i, s] of M.precision_stages.entries()) {
      const r = nativeRecipe(); Object.assign(r.quant, {granularity: q.granularity, levels: q.levels, clip: q.clip, stage_bits: {[s.stage]: q.bits}});
      const {m} = await scoreOn(r, SCAN);
      const drop = Math.max(0, (NATIVE.mapScan - m) / Math.max(NATIVE.mapScan, 1e-9));
      out.push({...s, sensitivity: +drop.toFixed(3), value: +(s.weight_frac / (drop + 0.05)).toFixed(3)});
      onProgress && onProgress((i + 1) / M.precision_stages.length, s.stage);
    }
    return {bits: q.bits, native_map: +NATIVE.mapScan.toFixed(4), stages: out};
  });
  function matchCounts(boxes, gt) {
    const used = new Set(); let found = 0;
    for (const b of boxes.slice().sort((a, z) => z[4] - a[4])) {
      let j = -1, best = -1; gt.forEach((g, k) => { const v = iou(b, g); if (v > best) { best = v; j = k; } });
      if (best >= 0.5 && !used.has(j)) { used.add(j); found++; }
    }
    return {found, missed: gt.length - found, false: boxes.length - found};
  }
  const preview = idx => lock(async () => {
    const n = PREVIEW.length, k = ((idx % n) + n) % n, pi = PREVIEW[k], p = PHOTOS[pi];
    let e = cache.get(keyOf(current));
    if (!e || !e.preds.has(pi)) { await scoreOn(current, [pi]); e = cache.get(keyOf(current)); }
    const nat = cache.get(keyOf(nativeRecipe())).preds.get(pi).filter(b => b[4] > CONF_SHOW);
    const boxes = e.preds.get(pi).filter(b => b[4] > CONF_SHOW);
    return {n, idx: k, w: p.w, h: p.h, src: p.file, gt: p.gt, native: nat, boxes,
            score: matchCounts(boxes, p.gt), native_score: matchCounts(nat, p.gt)};
  });
  function inspect(stage, bits, gran, levels, clip, row = 0) {
    let best = null;
    for (const name in NAT) if (NAT[name].stage === stage && (!best || NAT[name].w.length > NAT[best].w.length)) best = name;
    const c = NAT[best], w = c.w, g = gran === 'channel' || gran === 'tensor' ? gran : +gran;
    const wq = quantize(w, c.O, c.cols, bits, g, levels, clip), rows = rowsOf(c.O, c.cols, g);
    row = ((row % rows.length) + rows.length) % rows.length;
    const [s, e] = rows[row], len = Math.min(e - s, 144);
    const orig = Array.from(w.subarray(s, s + len)), quant = Array.from(wq.subarray(s, s + len));
    let grid = [];
    if (bits < NATIVE_BITS) {
      if (bits === 1) { let m = 0; for (let i = s; i < e; i++) m += Math.abs(w[i]); m /= rows[row][2]; grid = [-m, m]; }
      else {
        let amax = 0; for (let i = s; i < e; i++) amax = Math.max(amax, Math.abs(w[i]));
        if (clip === 'max' && amax > 0) {
          if (levels === 'fp' && FP_FORMATS[bits]) grid = signedGrid(bits).sg.map(v => v * amax);
          else { const q = (1 << (bits - 1)) - 1; for (let k = -q; k <= q; k++) grid.push(amax * k / q); }
        } else grid = [...new Set(Array.from(wq.subarray(s, e)))].sort((a, z) => a - z);
      }
    }
    let sig = 0, err = 0, nz = 0, zeroed = 0; const distinct = new Set();
    for (let i = 0; i < w.length; i++) { sig += w[i] * w[i]; const d = w[i] - wq[i]; err += d * d;
      if (w[i] !== 0) { nz++; if (wq[i] === 0) zeroed++; } distinct.add(wq[i]); }
    const cost = bits >= NATIVE_BITS ? NATIVE_BITS : (w.length * bits + nScales(c.O, c.cols, g) * SCALE_BITS) / w.length;
    return {layer: best, shape: [c.O, c.I, c.kh, c.kw], orig, quant, grid, row, rows: rows.length, row_len: e - s,
      sqnr_db: bits < NATIVE_BITS ? +(10 * Math.log10(sig / Math.max(err, 1e-20))).toFixed(1) : null,
      zeroed_pct: +(100 * zeroed / Math.max(nz, 1)).toFixed(1), distinct: distinct.size, numel: w.length, eff_bits: +cost.toFixed(2)};
  }
  const reset = () => { current = nativeRecipe(); return {ok: true, map: +NATIVE.map.toFixed(4)}; };
  // test hooks (parity checks against PyTorch)
  const _rawHead = (photoIdx) => lock(async () => { applyRecipe(nativeRecipe()); const p = PHOTOS[photoIdx];
    return tf.tidy(() => { const x = tf.div(tf.tensor4d(p.px, [1, S, S, 3], 'int32').toFloat(), 255); const [b, c] = forward(x);
      return {box: Array.from(b.dataSync()), cls: Array.from(c.dataSync()), px: Array.from(p.px)}; }); });
  const _photo = i => { const p = PHOTOS[i]; let bin = ''; for (let k = 0; k < p.px.length; k += 8192) bin += String.fromCharCode(...p.px.subarray(k, k + 8192));
    return {px: btoa(bin), left: p.left, top: p.top, r: p.r, w: p.w, h: p.h, gt: p.gt, eval: p.eval}; };
  const _preds = r => cache.get(keyOf(r || nativeRecipe())).preds;
  return {load, info, measure, _photo, _preds, sensitivity, precisionScan, preview, inspect, reset, keyOf, nativeRecipe, _rawHead,
          get native() { return NATIVE; }, photoUrl: i => PHOTOS[PREVIEW[i]].file};
})();
