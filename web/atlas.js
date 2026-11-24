// Millions of points in WebGL2. Each frame draws every point once into a
// half-float framebuffer with additive blending (colour times weight, and the
// weight), then a full-screen pass turns that into an image: the mean colour of
// what landed on a pixel, brightened by how much landed there. Overdraw then
// shows as density instead of whichever point was drawn last.

const params = new URLSearchParams(location.search);
const DATA = `data/${params.get("data") || "synthetic"}/`;
const $ = (id) => document.getElementById(id);

const POINT_VS = `#version 300 es
layout(location = 0) in uvec2 aPos;
layout(location = 1) in uvec2 aCatYear;
uniform vec4 uView;        // x scale, y scale, x offset, y offset: [0,1] -> clip
uniform float uSize;
uniform uint uYearMax;
uniform sampler2D uColors; // 256 x 1, alpha 0 = category hidden
out vec3 vColor;
void main() {
  vec4 c = texelFetch(uColors, ivec2(int(aCatYear.x), 0), 0);
  if (aCatYear.y > uYearMax || c.a == 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
  vec2 p = vec2(aPos) / 65535.0;
  gl_Position = vec4(p * uView.xy + uView.zw, 0.0, 1.0);
  gl_PointSize = uSize;
  vColor = c.rgb;
}`;
const POINT_FS = `#version 300 es
precision highp float;
in vec3 vColor;
out vec4 o;
void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(d, d);
  if (r2 > 1.0) discard;
  float a = 1.0 - r2;
  o = vec4(vColor * a, a);
}`;
const QUAD_VS = `#version 300 es
const vec2 P[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
void main() { gl_Position = vec4(P[gl_VertexID], 0.0, 1.0); }`;
const TONE_FS = `#version 300 es
precision highp float;
uniform sampler2D uAcc;
uniform float uExposure;
uniform vec3 uBg;
out vec4 o;
void main() {
  vec4 s = texelFetch(uAcc, ivec2(gl_FragCoord.xy), 0);
  if (s.a <= 0.0) { o = vec4(uBg, 1.0); return; }
  vec3 col = s.rgb / s.a;
  float k = 1.0 - exp(-s.a * uExposure);
  vec3 lit = mix(col, vec3(1.0), 0.35 * k * k);  // dense cores go towards white
  o = vec4(mix(uBg, lit, pow(k, 0.7)), 1.0);
}`;

function compile(gl, vs, fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) {
    const name = gl.getActiveUniform(p, i).name;
    u[name] = gl.getUniformLocation(p, name);
  }
  return { p, u };
}

function hexRgb(h) {
  return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
}

class Atlas {
  constructor(meta, buf) {
    this.meta = meta;
    this.n = meta.n;
    this.raw = new Uint8Array(buf);
    this.pts = new Uint16Array(buf, 0, (buf.byteLength / 2) | 0);
    this.view = { cx: 0.5, cy: 0.5, zoom: 1 };  // centre in data units, zoom 1 = fits
    this.yearMax = meta.year1 - meta.year0;
    this.enabled = meta.categories.map(() => true);
    this.tiles = new Map();
    this.frames = [];
    this.canvas = $("gl");
    const gl = (this.gl = this.canvas.getContext("webgl2", { antialias: false, alpha: false }));
    if (!gl) throw new Error("WebGL2 is not available");
    if (!gl.getExtension("EXT_color_buffer_float")) throw new Error("EXT_color_buffer_float is not available");
    this.points = compile(gl, POINT_VS, POINT_FS);
    this.tone = compile(gl, QUAD_VS, TONE_FS);
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, this.raw, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribIPointer(0, 2, gl.UNSIGNED_SHORT, 6, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribIPointer(1, 2, gl.UNSIGNED_BYTE, 6, 4);
    gl.bindVertexArray(null);
    this.colorTex = gl.createTexture();
    this.uploadColors();
    this.buildIndex();
    this.resize();
    addEventListener("resize", () => { this.resize(); this.dirty = true; });
  }

  uploadColors() {
    const gl = this.gl;
    const data = new Uint8Array(256 * 4);
    this.meta.categories.forEach((c, i) => {
      data.set([...hexRgb(c.color), this.enabled[i] ? 255 : 0], i * 4);
    });
    gl.bindTexture(gl.TEXTURE_2D, this.colorTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    this.dirty = true;
  }

  resize() {
    const gl = this.gl;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    this.w = Math.round(innerWidth * dpr);
    this.h = Math.round(innerHeight * dpr);
    this.dpr = dpr;
    this.canvas.width = this.w;
    this.canvas.height = this.h;
    const lc = $("labels");
    lc.width = this.w;
    lc.height = this.h;
    if (this.fbo) {
      gl.deleteFramebuffer(this.fbo);
      gl.deleteTexture(this.acc);
    }
    this.acc = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.acc);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA16F, this.w, this.h);
    this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.acc, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // [0,1] data square -> clip space, keeping the aspect ratio
  viewUniform() {
    const { cx, cy, zoom } = this.view;
    const s = Math.min(this.w, this.h) * zoom;
    const sx = (2 * s) / this.w, sy = (2 * s) / this.h;
    return [sx, sy, -cx * sx, -cy * sy];
  }

  toScreen(x, y) {
    const [sx, sy, ox, oy] = this.viewUniform();
    return [((x * sx + ox + 1) / 2) * this.w, ((1 - (y * sy + oy)) / 2) * this.h];
  }

  toData(px, py) {
    const [sx, sy, ox, oy] = this.viewUniform();
    return [((px / this.w) * 2 - 1 - ox) / sx, ((1 - (py / this.h) * 2) - oy) / sy];
  }

  draw() {
    const gl = this.gl;
    const t0 = performance.now();
    const z = this.view.zoom;
    const size = Math.min(1.6 * Math.pow(z, 0.45), 9) * this.dpr;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, this.w, this.h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(this.points.p);
    gl.uniform4fv(this.points.u.uView, this.viewUniform());
    gl.uniform1f(this.points.u.uSize, size);
    gl.uniform1ui(this.points.u.uYearMax, this.yearMax);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.colorTex);
    gl.uniform1i(this.points.u.uColors, 0);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.POINTS, 0, this.n);
    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(this.tone.p);
    gl.bindTexture(gl.TEXTURE_2D, this.acc);
    gl.uniform1i(this.tone.u.uAcc, 0);
    // exposure from the mean coverage of the data's footprint, so the picture
    // keeps its brightness as the zoom and point size change
    const footprint = Math.pow(Math.min(this.w, this.h) * z, 2) * 0.35;
    const coverage = (this.n * Math.PI * (size / 2) ** 2 * 0.5) / footprint;
    gl.uniform1f(this.tone.u.uExposure, 1.2 / Math.max(coverage, 0.05));
    gl.uniform3fv(this.tone.u.uBg, [11 / 255, 13 / 255, 18 / 255]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.drawLabels();
    this.frames.push(performance.now() - t0);
  }

  // ---- hover: a uniform grid over the quantized positions

  buildIndex() {
    const G = (this.G = 1024), n = this.n, p = this.pts;
    const cell = new Uint32Array(n);
    const start = new Uint32Array(G * G + 1);
    for (let i = 0; i < n; i++) {
      const c = (p[i * 3 + 1] >> 6) * G + (p[i * 3] >> 6);
      cell[i] = c;
      start[c + 1]++;
    }
    for (let c = 0; c < G * G; c++) start[c + 1] += start[c];
    const fill = start.slice(0, G * G);
    const items = new Uint32Array(n);
    for (let i = 0; i < n; i++) items[fill[cell[i]]++] = i;
    this.cellStart = start;
    this.cellItems = items;
  }

  visible(i) {
    return this.raw[i * 6 + 5] <= this.yearMax && this.enabled[this.raw[i * 6 + 4]];
  }

  nearest(px, py, radiusPx) {
    const [x, y] = this.toData(px, py);
    const s = Math.min(this.w, this.h) * this.view.zoom;
    const r = radiusPx / s;
    const G = this.G, p = this.pts;
    const c0 = Math.max(0, Math.floor((x - r) * G)), c1 = Math.min(G - 1, Math.floor((x + r) * G));
    const r0 = Math.max(0, Math.floor((y - r) * G)), r1 = Math.min(G - 1, Math.floor((y + r) * G));
    let best = -1, bd = r * r;
    for (let gy = r0; gy <= r1; gy++) {
      for (let gx = c0; gx <= c1; gx++) {
        const c = gy * G + gx;
        for (let k = this.cellStart[c]; k < this.cellStart[c + 1]; k++) {
          const i = this.cellItems[k];
          const dx = p[i * 3] / 65535 - x, dy = p[i * 3 + 1] / 65535 - y;
          const d = dx * dx + dy * dy;
          if (d < bd && this.visible(i)) { bd = d; best = i; }
        }
      }
    }
    return best;
  }

  async title(i) {
    const off = this.meta.offsets;
    let lo = 0, hi = off.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (off[mid] <= i) lo = mid; else hi = mid;
    }
    if (!this.tiles.has(lo)) this.tiles.set(lo, fetch(`${DATA}tiles/${lo}.json`).then((r) => r.json()));
    return (await this.tiles.get(lo))[i - off[lo]];
  }

  // ---- labels: bigger clusters first, skipped where they would overlap

  drawLabels() {
    const ctx = $("labels").getContext("2d");
    ctx.clearRect(0, 0, this.w, this.h);
    const z = this.view.zoom;
    const level = z < 2.5 ? 0 : z < 10 ? 1 : 2;
    const placed = [];
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const lb of this.labelsByLevel[level] || []) {
      const [sx, sy] = this.toScreen(lb.x, lb.y);
      if (sx < 0 || sy < 0 || sx > this.w || sy > this.h) continue;
      const fs = (level === 0 ? 15 : 12) * this.dpr;
      ctx.font = `${level === 0 ? 600 : 500} ${fs}px -apple-system, Segoe UI, Roboto, sans-serif`;
      const w = ctx.measureText(lb.text).width + 6 * this.dpr, h = fs * 1.3;
      const box = [sx - w / 2, sy - h / 2, sx + w / 2, sy + h / 2];
      if (placed.some((b) => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) continue;
      placed.push(box);
      ctx.lineWidth = 3 * this.dpr;
      ctx.strokeStyle = "rgba(11,13,18,0.85)";
      ctx.strokeText(lb.text, sx, sy);
      ctx.fillStyle = level === 0 ? "#ffffff" : "#d0d4da";
      ctx.fillText(lb.text, sx, sy);
    }
  }
}

async function main() {
  const meta = await (await fetch(`${DATA}meta.json`)).json();
  const t0 = performance.now();
  const buf = await (await fetch(`${DATA}points.bin`)).arrayBuffer();
  const loadMs = performance.now() - t0;
  const a = new Atlas(meta, buf);
  window.atlas = a;
  a.labelsByLevel = [0, 1, 2].map((l) => meta.labels.filter((x) => x.level === l).sort((p, q) => q.size - p.size));
  $("title").textContent = meta.name || "embedding map";
  $("subtitle").textContent = `${meta.n.toLocaleString()} points, ${(buf.byteLength / 1e6).toFixed(1)} MB in ${(loadMs / 1000).toFixed(1)} s`;

  // categories
  const cats = $("cats");
  meta.categories.forEach((c, i) => {
    if (!c.count) return;
    const row = document.createElement("div");
    row.className = "cat";
    row.dataset.i = i;
    row.innerHTML = `<span class="sw" style="background:${c.color}"></span><span></span><span class="n">${c.count.toLocaleString()}</span>`;
    row.children[1].textContent = c.name;
    row.onclick = (e) => {
      if (e.altKey || e.metaKey) a.enabled = a.enabled.map((_, j) => j === i);
      else a.enabled[i] = !a.enabled[i];
      cats.querySelectorAll(".cat").forEach((r) => r.classList.toggle("off", !a.enabled[+r.dataset.i]));
      a.uploadColors();
    };
    cats.appendChild(row);
  });

  // years
  const yr = $("year");
  yr.min = meta.year0;
  yr.max = meta.year1;
  yr.value = meta.year1;
  const setYear = (v) => { yr.value = v; $("yearv").textContent = v; a.yearMax = v - meta.year0; a.dirty = true; };
  setYear(meta.year1);
  yr.oninput = () => setYear(+yr.value);
  let playing = null;
  $("play").onclick = () => {
    if (playing) { clearInterval(playing); playing = null; $("play").textContent = "play"; return; }
    $("play").textContent = "stop";
    let v = meta.year0;
    playing = setInterval(() => {
      setYear(v);
      if (++v > meta.year1) { clearInterval(playing); playing = null; $("play").textContent = "play"; }
    }, 350);
  };

  // pan and zoom
  const cv = a.canvas;
  let drag = null;
  let down = null;
  cv.addEventListener("pointerdown", (e) => {
    drag = [e.clientX, e.clientY];
    down = [e.clientX, e.clientY];
    cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener("pointerup", async (e) => {
    drag = null;
    // a click, not a drag: open the point's page if the data has links
    if (meta.link && down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 4) {
      const i = a.nearest(e.clientX * a.dpr, e.clientY * a.dpr, 8 * a.dpr);
      if (i >= 0) {
        const entry = await a.title(i);
        if (Array.isArray(entry)) window.open(meta.link + entry[1], "_blank", "noopener");
      }
    }
  });
  cv.addEventListener("pointermove", (e) => {
    if (drag) {
      const s = Math.min(a.w, a.h) * a.view.zoom / a.dpr;
      a.view.cx -= (e.clientX - drag[0]) / s;
      a.view.cy += (e.clientY - drag[1]) / s;
      drag = [e.clientX, e.clientY];
      a.dirty = true;
      $("tip").style.display = "none";
      return;
    }
    hover(e.clientX, e.clientY);
  });
  cv.addEventListener("wheel", (e) => {
    e.preventDefault();
    const px = e.clientX * a.dpr, py = e.clientY * a.dpr;
    const [x0, y0] = a.toData(px, py);
    a.view.zoom = Math.min(Math.max(a.view.zoom * Math.exp(-e.deltaY * 0.0015), 0.5), 4000);
    const [x1, y1] = a.toData(px, py);
    a.view.cx += x0 - x1;
    a.view.cy += y0 - y1;
    a.dirty = true;
  }, { passive: false });

  let hoverSeq = 0;
  async function hover(cx, cy) {
    const i = a.nearest(cx * a.dpr, cy * a.dpr, 8 * a.dpr);
    const tip = $("tip");
    if (i < 0) { tip.style.display = "none"; return; }
    const seq = ++hoverSeq;
    const text = await a.title(i);
    if (seq !== hoverSeq) return;
    const c = meta.categories[a.raw[i * 6 + 4]];
    tip.innerHTML = `<div></div><div class="meta"></div>`;
    const [title, id] = Array.isArray(text) ? text : [text, null];
    tip.children[0].textContent = title;
    tip.children[1].textContent = `${c.name} · ${meta.year0 + a.raw[i * 6 + 5]}${id ? " · " + id + " · click to open" : ""}`;
    tip.style.display = "block";
    tip.style.left = Math.min(cx + 14, innerWidth - 380) + "px";
    tip.style.top = cy + 14 + "px";
  }

  // render loop: redraw only when something changed
  a.dirty = true;
  const frame = () => {
    if (a.dirty || a.bench) {
      a.dirty = false;
      a.draw();
      const f = a.frames.slice(-60);
      $("stats").textContent = `${(f.reduce((x, y) => x + y, 0) / f.length).toFixed(1)} ms cpu / frame · zoom ${a.view.zoom.toFixed(1)}x`;
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);

  // ?bench=1: zoom and pan continuously for a few seconds and report frame times
  if (params.get("bench")) {
    a.bench = true;
    const t1 = performance.now();
    const stamps = [];
    const step = (now) => {
      stamps.push(now);
      const t = (now - t1) / 1000;
      a.view.zoom = 1 + 9 * (0.5 - 0.5 * Math.cos(t * 1.2));
      a.view.cx = 0.5 + 0.15 * Math.sin(t * 0.7);
      a.view.cy = 0.5 + 0.15 * Math.cos(t * 0.5);
      if (t < 8) requestAnimationFrame(step);
      else {
        a.bench = false;
        const gaps = stamps.slice(1).map((s, k) => s - stamps[k]).sort((p, q) => p - q);
        window.benchResult = {
          points: a.n, width: a.w, height: a.h, frames: gaps.length,
          fps: +(1000 / (gaps.reduce((x, y) => x + y, 0) / gaps.length)).toFixed(1),
          p50_ms: +gaps[gaps.length >> 1].toFixed(1), p95_ms: +gaps[Math.floor(gaps.length * 0.95)].toFixed(1),
          load_ms: Math.round(loadMs), mb: +(buf.byteLength / 1e6).toFixed(1), renderer: rendererName(a.gl),
        };
        console.log("bench", JSON.stringify(window.benchResult));
      }
    };
    requestAnimationFrame(step);
  }
}

function rendererName(gl) {
  const ext = gl.getExtension("WEBGL_debug_renderer_info");
  return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
}

main().catch((e) => {
  $("err").style.display = "block";
  $("err").textContent = "Could not start: " + e.message;
  console.error(e);
});
