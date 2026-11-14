(() => {
'use strict';

const cfg = /*__CONFIG__*/null;
const sim = cfg.simulation;
const $ = id => document.getElementById(id);
const app = $('app');

/* =====================================================================
 * Constants & small helpers
 * ===================================================================== */
const EARTH_KM = 6371, MOON_KM = 1737.4;
const OBLIQUITY = 23.4393 * Math.PI / 180;
const ECLIPTIC_NORTH = new THREE.Vector3(0, -Math.sin(OBLIQUITY), Math.cos(OBLIQUITY));
const PHASE_COLORS = ['#36f5a6', '#39c6ff', '#ff9a3c', '#c99bff', '#ff6fae', '#f2e36b'];
const STAGE_NAMES = ['Boost', 'Core', 'ICPS'];
const CLASS_COLORS = ['#5ee6a8', '#5fd4ff', '#ffb35c', '#c99bff', '#ff6fae', '#f2e36b'];
const SITE_COLORS = ['#5fd4ff', '#ffb35c', '#5ee6a8', '#c99bff'];

const vec = a => new THREE.Vector3(+a[0], +a[1], +a[2]);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const setText = (id, text) => { $(id).textContent = text; };
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm3 = a => Math.hypot(a[0], a[1], a[2]);
const scale3 = (a, s) => [a[0] * s, a[1] * s, a[2] * s];

const fmt = {
  deg: v => `${Number(v).toFixed(1)}°`,
  days: v => `${Number(v).toFixed(2)} d`,
  km: v => `${Math.round(v).toLocaleString()} km`,
  latlon: (lat, lon) => `${Math.abs(lat).toFixed(2)}°${lat < 0 ? 'S' : 'N'} ${Math.abs(lon).toFixed(2)}°${lon < 0 ? 'W' : 'E'}`,
  seconds: s => Math.abs(s) < 120 ? `${s.toFixed(1)} s` : Math.abs(s) < 7200 ? `${(s / 60).toFixed(1)} min` : Math.abs(s) < 172800 ? `${(s / 3600).toFixed(1)} h` : `${(s / 86400).toFixed(2)} d`,
  length: km => km < 1 ? `${(km * 1000).toFixed(0)} m` : km < 100 ? `${km.toFixed(2)} km` : `${Math.round(km).toLocaleString()} km`,
  utc: iso => String(iso || '').replace('T', ' ').replace(/:\d\d(\.\d+)?Z$/, ' UTC').replace('Z', ' UTC'),
};

function bodyUnit(lat, lon) { // three.js sphere convention: +x = 0° lon, +y = north, -z = 90°E
  const a = lat * Math.PI / 180, b = lon * Math.PI / 180;
  return new THREE.Vector3(Math.cos(a) * Math.cos(b), Math.sin(a), -Math.cos(a) * Math.sin(b));
}
function greatCircleDeg(a, b) {
  const p = bodyUnit(+a.lat, +a.lon), q = bodyUnit(+b.lat, +b.lon);
  return Math.acos(clamp(p.dot(q), -1, 1)) * 180 / Math.PI;
}
function seededRandom(seed) { return () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296; }
function indexAt(xs, x) { // largest i with xs[i] <= x (clamped)
  let lo = 0, hi = xs.length - 1;
  if (x <= xs[0]) return 0;
  if (x >= xs[hi]) return Math.max(0, hi - 1);
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
  return lo;
}
function interpolate(xs, ys, x) {
  if (!xs.length) return NaN;
  const i = indexAt(xs, x), j = Math.min(i + 1, xs.length - 1);
  const f = xs[j] > xs[i] ? clamp((x - xs[i]) / (xs[j] - xs[i]), 0, 1) : 0;
  return ys[i] + (ys[j] - ys[i]) * f;
}
function disposeAll(list) {
  for (const object of list) {
    object.parent?.remove(object);
    object.geometry?.dispose();
    if (object.material && !object.material.userData?.shared) object.material.dispose();
  }
  list.length = 0;
}
/** IAU Earth orientation (precession ignored) at a UTC time in ms: body axes in ICRF. */
function earthFrameAt(ms) {
  const w = (190.147 + 360.9856235 * (ms / 86400000 + 2440587.5 - 2451545.0)) * Math.PI / 180;
  return {x: [-Math.sin(w), Math.cos(w), 0], y: [-Math.cos(w), -Math.sin(w), 0], z: [0, 0, 1]};
}
function orient(group, frame) { // frame axes are ICRF vectors; map to three.js body axes (x, north, -east)
  if (!frame) return;
  const x = vec(frame.x), y = vec(frame.y), z = vec(frame.z);
  group.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, z, y.negate()));
}

/* =====================================================================
 * Renderer & shared assets (textures, materials, stars)
 * ===================================================================== */
const renderer = new THREE.WebGLRenderer({antialias: true, logarithmicDepthBuffer: true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputEncoding = THREE.sRGBEncoding;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.domElement.className = 'viewport';
app.prepend(renderer.domElement);
const anisotropy = renderer.capabilities.getMaxAnisotropy();

const TEXTURES = 'https://threejs.org/examples/textures/planets/';
const textureLoader = new THREE.TextureLoader();
textureLoader.setCrossOrigin('anonymous');
function loadTexture(file, onLoad, isColor = true) {
  textureLoader.load(TEXTURES + file, texture => {
    if (isColor) texture.encoding = THREE.sRGBEncoding;
    texture.anisotropy = anisotropy;
    onLoad(texture);
  }, undefined, () => console.warn(`Texture unavailable: ${file}`));
}
function canvasTexture(size, paint, isColor = true) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  paint(canvas.getContext('2d'), size);
  const texture = new THREE.CanvasTexture(canvas);
  if (isColor) texture.encoding = THREE.sRGBEncoding;
  texture.anisotropy = anisotropy;
  return texture;
}
const shared = material => { material.userData.shared = true; return material; };

// Sun direction in view space, shared by the Earth night-lights and atmosphere shaders.
const sunDirView = {value: new THREE.Vector3(1, 0, 0)};

const earthMaterial = shared(new THREE.MeshStandardMaterial({color: 0x2c5f8f, roughness: 0.8, metalness: 0, emissive: 0xffc58a, emissiveIntensity: 0}));
earthMaterial.onBeforeCompile = shader => {
  shader.uniforms.sunDirView = sunDirView;
  shader.fragmentShader = 'uniform vec3 sunDirView;\n' + shader.fragmentShader.replace('#include <emissivemap_fragment>',
    '#include <emissivemap_fragment>\n  totalEmissiveRadiance *= smoothstep(0.1, -0.25, dot(normalize(vNormal), sunDirView));');
};
loadTexture('earth_atmos_2048.jpg', t => { earthMaterial.map = t; earthMaterial.color.set(0xffffff); earthMaterial.needsUpdate = true; });
loadTexture('earth_normal_2048.jpg', t => { earthMaterial.normalMap = t; earthMaterial.normalScale.set(0.55, 0.55); earthMaterial.needsUpdate = true; }, false);
loadTexture('earth_lights_2048.png', t => { earthMaterial.emissiveMap = t; earthMaterial.emissiveIntensity = 1.3; earthMaterial.needsUpdate = true; });

const cloudMaterial = shared(new THREE.MeshStandardMaterial({color: 0xffffff, transparent: true, opacity: 0, roughness: 1, depthWrite: false}));
loadTexture('earth_clouds_1024.png', t => { cloudMaterial.map = t; cloudMaterial.alphaMap = t; cloudMaterial.opacity = 0.85; cloudMaterial.needsUpdate = true; });

const atmosphereMaterial = shared(new THREE.ShaderMaterial({
  uniforms: {sunDirView, glowColor: {value: new THREE.Color(0x4fa8ff)}},
  vertexShader: `
    #include <common>
    #include <logdepthbuf_pars_vertex>
    varying vec3 vNormalV; varying vec3 vViewDir;
    void main() {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vNormalV = normalize(normalMatrix * normal);
      vViewDir = normalize(-mv.xyz);
      gl_Position = projectionMatrix * mv;
      #include <logdepthbuf_vertex>
    }`,
  fragmentShader: `
    #include <logdepthbuf_pars_fragment>
    uniform vec3 sunDirView; uniform vec3 glowColor;
    varying vec3 vNormalV; varying vec3 vViewDir;
    void main() {
      #include <logdepthbuf_fragment>
      vec3 n = normalize(vNormalV);
      float rim = pow(1.0 - max(dot(n, normalize(vViewDir)), 0.0), 3.0);
      float lit = smoothstep(-0.3, 0.45, dot(n, sunDirView));
      gl_FragColor = vec4(glowColor * rim * lit * 1.4, 1.0);
    }`,
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
}));

const moonMaterial = shared(new THREE.MeshStandardMaterial({color: 0x8d8f91, roughness: 1, metalness: 0}));
loadTexture('moon_1024.jpg', t => { moonMaterial.map = t; moonMaterial.bumpMap = t; moonMaterial.bumpScale = 0.005; moonMaterial.color.set(0xffffff); moonMaterial.needsUpdate = true; });

const glowTexture = canvasTexture(128, (x, s) => {
  const g = x.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(0.18, 'rgba(255,255,255,.6)'); g.addColorStop(1, 'rgba(255,255,255,0)');
  x.fillStyle = g; x.fillRect(0, 0, s, s);
});
function glowSprite(color, size, attenuate = true) {
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({map: glowTexture, color, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: attenuate, toneMapped: false}));
  sprite.scale.setScalar(size);
  return sprite;
}

const sphere = new THREE.SphereGeometry(1, 128, 96);
function makeEarth(radius) {
  const group = new THREE.Group();
  const surface = new THREE.Mesh(sphere, earthMaterial);
  const clouds = new THREE.Mesh(sphere, cloudMaterial);
  const atmosphere = new THREE.Mesh(sphere, atmosphereMaterial);
  surface.scale.setScalar(radius); clouds.scale.setScalar(radius * 1.006); atmosphere.scale.setScalar(radius * 1.03);
  group.add(surface, clouds, atmosphere);
  return {group, surface, clouds, radius};
}
function makeMoon(radius) {
  const group = new THREE.Group();
  const surface = new THREE.Mesh(sphere, moonMaterial);
  surface.scale.setScalar(radius);
  group.add(surface);
  return {group, surface, radius};
}
function makeStars(radius) {
  const rand = seededRandom(14321), positions = [], colors = [];
  for (let i = 0; i < 7000; i++) {
    const u = rand() * 2 - 1, a = rand() * Math.PI * 2, s = Math.sqrt(1 - u * u), tint = rand();
    positions.push(s * Math.cos(a) * radius, u * radius, s * Math.sin(a) * radius);
    const color = new THREE.Color(tint > 0.95 ? 0xffd2a1 : tint > 0.88 ? 0xa9d4ff : 0xdfe8f5).multiplyScalar(0.35 + 0.65 * rand() ** 3);
    colors.push(color.r, color.g, color.b);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  const points = new THREE.Points(geometry, new THREE.PointsMaterial({size: 1.5, sizeAttenuation: false, vertexColors: true, depthWrite: false}));
  points.frustumCulled = false;
  points.renderOrder = -1;
  return points;
}

/* =====================================================================
 * Screen labels (HTML tags projected from 3D)
 * ===================================================================== */
const labels = (() => {
  const layer = $('labels'), pool = new Map();
  let enabled = true;
  const projected = new THREE.Vector3();
  return {
    set enabled(value) { enabled = value; layer.hidden = !value; },
    render(items, camera) {
      const seen = new Set();
      if (enabled) {
        const width = renderer.domElement.clientWidth, height = renderer.domElement.clientHeight;
        for (const item of items) {
          projected.copy(item.at).project(camera);
          if (projected.z > 1 || Math.abs(projected.x) > 1.05 || Math.abs(projected.y) > 1.05) continue;
          let el = pool.get(item.key);
          if (!el) { el = document.createElement('div'); layer.appendChild(el); pool.set(item.key, el); }
          el.className = `tag ${item.kind || ''}`;
          if (el.textContent !== item.text) el.textContent = item.text;
          el.style.left = `${(projected.x + 1) / 2 * width}px`;
          el.style.top = `${(1 - projected.y) / 2 * height}px`;
          el.hidden = false;
          seen.add(item.key);
        }
      }
      for (const [key, el] of pool) if (!seen.has(key)) el.hidden = true;
    },
  };
})();
function cameraRight(camera) { return new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0); }

/* =====================================================================
 * Charts (speed chart in the details panel and the graphs dialog)
 * ===================================================================== */
function niceStep(span, count) {
  const raw = span / Math.max(1, count), power = 10 ** Math.floor(Math.log10(raw)), unit = raw / power;
  return (unit >= 5 ? 10 : unit >= 2 ? 5 : unit >= 1 ? 2 : 1) * power;
}
function drawChart(canvas, series, options = {}) {
  const dpr = window.devicePixelRatio || 1, width = canvas.clientWidth || 300, height = canvas.clientHeight || 150;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.font = '10px ui-monospace, Consolas, monospace';
  const xs = series.x, ys = series.y;
  if (!xs || xs.length < 2) {
    ctx.fillStyle = '#5d7083'; ctx.fillText(options.empty || 'No data', 12, height / 2); return;
  }
  const pad = {l: 46, r: 12, t: 10, b: 22};
  const x0 = xs[0], x1 = xs[xs.length - 1] > x0 ? xs[xs.length - 1] : x0 + 1;
  let y0 = Infinity, y1 = -Infinity;
  for (const y of ys) { if (y < y0) y0 = y; if (y > y1) y1 = y; }
  if (options.zero) y0 = Math.min(0, y0);
  if (y1 - y0 < 1e-9) { y1 += Math.abs(y1) * 0.05 + 1e-3; y0 -= Math.abs(y0) * 0.05 + 1e-3; }
  const step = niceStep(y1 - y0, 4);
  y0 = Math.floor(y0 / step) * step; y1 = Math.ceil(y1 / step) * step;
  const px = x => pad.l + (x - x0) / (x1 - x0) * (width - pad.l - pad.r);
  const py = y => height - pad.b - (y - y0) / (y1 - y0) * (height - pad.t - pad.b);
  const yFormat = options.yFormat || (v => Math.abs(step) >= 1 ? v.toLocaleString(undefined, {maximumFractionDigits: 0}) : v.toFixed(Math.min(4, Math.ceil(-Math.log10(step)))));
  const xFormat = options.xFormat || (v => v.toFixed(1));

  ctx.strokeStyle = 'rgba(170,205,235,.1)'; ctx.fillStyle = '#5d7083'; ctx.lineWidth = 1;
  for (let y = y0; y <= y1 + step / 2; y += step) {
    const yy = Math.round(py(y)) + 0.5;
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(width - pad.r, yy); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText(yFormat(y), pad.l - 6, yy + 3);
  }
  ctx.textAlign = 'left'; ctx.fillText(xFormat(x0), pad.l, height - 6);
  ctx.textAlign = 'right'; ctx.fillText(xFormat(x1), width - pad.r, height - 6);
  ctx.textAlign = 'center'; ctx.fillText(xFormat((x0 + x1) / 2), (pad.l + width - pad.r) / 2, height - 6);

  // Line, split into same-colour runs; decimate to about one point per pixel.
  const stride = Math.max(1, Math.floor(xs.length / (width * 1.5)));
  const colorAt = options.colorAt || (() => options.color || '#5fd4ff');
  ctx.lineWidth = 1.8; ctx.lineJoin = 'round';
  let runColor = null;
  for (let i = 0; i < xs.length; i += stride) {
    const color = colorAt(i), x = px(xs[i]), y = py(ys[i]);
    if (color !== runColor) {
      if (runColor !== null) { ctx.lineTo(x, y); ctx.stroke(); }
      ctx.strokeStyle = color; ctx.beginPath(); ctx.moveTo(x, y); runColor = color;
    } else ctx.lineTo(x, y);
    if (i + stride >= xs.length && i !== xs.length - 1) ctx.lineTo(px(xs[xs.length - 1]), py(ys[ys.length - 1]));
  }
  ctx.stroke();

  if (options.cursor != null && Number.isFinite(options.cursor)) {
    const x = px(clamp(options.cursor, x0, x1)), y = py(interpolate(xs, ys, options.cursor));
    ctx.strokeStyle = 'rgba(230,239,247,.35)'; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, height - pad.b); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = '#ffffff'; ctx.beginPath(); ctx.arc(x, y, 3.2, 0, Math.PI * 2); ctx.fill();
  }
}

/* =====================================================================
 * Planning view: heliocentric scene with exaggerated distances/sizes
 * ===================================================================== */
const P = (() => {
  const scene = new THREE.Scene();
  const system = new THREE.Group(); // ICRF-aligned, scaled scene units
  scene.add(system);
  const camera = new THREE.PerspectiveCamera(40, 1, 0.005, 8000);
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  Object.assign(controls, {enableDamping: true, dampingFactor: 0.085, enableZoom: false, rotateSpeed: 0.8, panSpeed: 0.8, minPolarAngle: 0.05, maxPolarAngle: Math.PI - 0.05});

  const position = id => vec(sim.scene_positions[id]);
  const sunPos = position('sun'), earthPos = position('earth'), moonPos = position('moon');
  const ER = +sim.scene_radii.earth, MR = +sim.scene_radii.moon, SR = +sim.scene_radii.sun;
  const earthMoonMid = earthPos.clone().lerp(moonPos, 0.5), separation = earthPos.distanceTo(moonPos);

  // Display rotation: ecliptic north is "up", the Sun-Earth line runs along +x.
  const sunToEarth = earthPos.clone().sub(sunPos);
  const xAxis = sunToEarth.sub(ECLIPTIC_NORTH.clone().multiplyScalar(sunToEarth.dot(ECLIPTIC_NORTH))).normalize();
  if (xAxis.lengthSq() > 0.5) system.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(xAxis, ECLIPTIC_NORTH, xAxis.clone().cross(ECLIPTIC_NORTH)).invert());

  const earth = makeEarth(ER), moon = makeMoon(MR);
  earth.group.position.copy(earthPos); moon.group.position.copy(moonPos);
  orient(earth.group, sim.body_frames?.earth); orient(moon.group, sim.body_frames?.moon);
  system.add(earth.group, moon.group);

  // One directional light along the true Sun direction lights Earth and Moon consistently (phases are correct).
  const sunLight = new THREE.DirectionalLight(0xfff4e6, 2.0);
  sunLight.position.copy(sunPos); sunLight.target.position.copy(earthMoonMid);
  system.add(sunLight, sunLight.target);
  scene.add(new THREE.AmbientLight(0x8fa8c8, 0.05));
  const sunMesh = new THREE.Mesh(new THREE.SphereGeometry(SR, 48, 32), new THREE.MeshBasicMaterial({color: 0xfff1cf, toneMapped: false}));
  const sunGlow = glowSprite(0xffc070, SR * 9);
  sunMesh.position.copy(sunPos); sunGlow.position.copy(sunPos); system.add(sunMesh, sunGlow);

  const stars = makeStars(3000);
  scene.add(stars);
  scene.updateMatrixWorld(true);

  // Other bodies supplied by the simulation input.
  for (const [id, body] of Object.entries(sim.bodies || {})) {
    if (['sun', 'earth', 'moon'].includes(id) || !sim.scene_positions[id]) continue;
    const marker = new THREE.Mesh(new THREE.SphereGeometry(+sim.scene_radii[id] || 0.2, 24, 16), new THREE.MeshStandardMaterial({color: 0xc9b2ff, roughness: 0.9}));
    marker.position.copy(position(id)); marker.userData.label = body.name || id; system.add(marker);
  }

  // Surface markers live in the body groups so they rotate with the body.
  const markers = {candidates: new Map(), historic: [], results: []};
  function addMarker(body, location, color, size) {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(size, 14, 10), new THREE.MeshBasicMaterial({color, toneMapped: false}));
    mesh.position.copy(bodyUnit(+location.lat, +location.lon).multiplyScalar(body.radius * 1.004));
    mesh.userData = {location, body: body === earth ? 'earth' : 'moon'};
    body.group.add(mesh);
    return mesh;
  }

  const surfaceDirection = (body, lat, lon) => bodyUnit(+lat, +lon).applyQuaternion(body.group.quaternion);
  const toWorld = local => system.localToWorld(local.clone());

  // ---------- Camera ----------
  function placeCamera(targetLocal, offsetLocal) {
    controls.target.copy(toWorld(targetLocal));
    camera.position.copy(toWorld(targetLocal.clone().add(offsetLocal)));
    controls.update();
  }
  const view = {overhead: false};
  function home() {
    if (view.overhead) { // Looking down on the ecliptic; "up" on screen is the Sun-Earth line.
      camera.up.set(0, 0, -1);
      placeCamera(earthMoonMid, ECLIPTIC_NORTH.clone().multiplyScalar(separation * 2.3));
      return;
    }
    // Look at the Earth-Moon pair from the day side, slightly above the ecliptic.
    camera.up.set(0, 1, 0);
    const toSun = sunPos.clone().sub(earthMoonMid).normalize();
    const direction = toSun.applyAxisAngle(ECLIPTIC_NORTH, -1.1).addScaledVector(ECLIPTIC_NORTH, 0.42).normalize();
    placeCamera(earthMoonMid, direction.multiplyScalar(separation * 1.75 * Math.max(1, 1.3 / camera.aspect)));
  }
  function setOverhead(on) {
    view.overhead = on;
    controls.enableRotate = !on;
    home();
  }
  // Fly the camera to a point on the Moon and mark it with a glow.
  const highlight = glowSprite(0xffffff, 0.045, false);
  highlight.visible = false;
  system.add(highlight);
  function focusMoonPoint(lat, lon) {
    const direction = surfaceDirection(moon, lat, lon);
    highlight.position.copy(moonPos).addScaledVector(direction, MR * 1.01);
    highlight.visible = true;
    view.overhead = false; controls.enableRotate = true; camera.up.set(0, 1, 0);
    placeCamera(moonPos, direction.clone().multiplyScalar(MR * 5.5).addScaledVector(ECLIPTIC_NORTH, MR * 0.8));
  }

  // ---------- Per-frame update ----------
  function update(dt) {
    earth.clouds.rotation.y += dt * 0.006;
    stars.position.copy(camera.position);
    sunDirView.value.copy(sunPos).sub(earthMoonMid).transformDirection(system.matrixWorld).transformDirection(camera.matrixWorldInverse);
  }

  function labelItems() {
    const right = cameraRight(camera), items = [];
    const bodyLabel = (key, text, center, radius) => items.push({key, text, kind: 'body', at: toWorld(center).addScaledVector(right, radius * 1.15)});
    bodyLabel('earth', 'Earth', earthPos, ER); bodyLabel('moon', 'Moon', moonPos, MR);
    if (sunMesh.visible) bodyLabel('sun', 'Sun', sunPos, SR * 1.6);
    return items;
  }

  return {
    scene, camera, controls, system, earth, moon, sunMesh, sunGlow, markers, addMarker,
    home, setOverhead, update, labelItems, view, focusMoonPoint,
    sceneFor: {earthPos, moonPos, ER, MR},
  };
})();

/* =====================================================================
 * Telemetry view: CSV trajectory at true scale (km) in its ECI frame — Earth fixed at the origin
 * ===================================================================== */
const T = (() => {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 1e-6, 5e8);
  let controls = null;
  const earth = makeEarth(EARTH_KM), moon = makeMoon(MOON_KM);
  orient(earth.group, sim.body_frames?.earth);
  scene.add(earth.group, moon.group);
  const earthHelio = sim.bodies?.earth?.position_km;
  const sunDirection = earthHelio ? vec(earthHelio).negate().normalize() : new THREE.Vector3(1, 0, 0);
  const sunLight = new THREE.DirectionalLight(0xfff4e6, 2.0);
  sunLight.position.copy(sunDirection); scene.add(sunLight, sunLight.target);
  scene.add(new THREE.AmbientLight(0x8fa8c8, 0.06));
  const stars = makeStars(1e8); scene.add(stars);
  const craft = glowSprite(0x7fe3ff, 0.05, false), craftCore = glowSprite(0xffffff, 0.016, false), targetMark = glowSprite(0xff6fd8, 0.032, false);
  scene.add(craft, craftCore, targetMark);
  const dropLine = new THREE.Line(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(6), 3)),
    new THREE.LineBasicMaterial({color: 0x7fe3ff, transparent: true, opacity: 0.45, toneMapped: false}));
  dropLine.frustumCulled = false; scene.add(dropLine);
  const dynamic = [];

  // Everything here is in the CSV's ECI frame: Earth's centre is the fixed origin and the Moon moves to
  // its moon_r position at each sample. Objects tied to the lunar surface carry a Moon-relative offset.
  const state = {data: null, times: [], focus: 'track', t: 0, playing: true, rate: 1,
    center: [0, 0, 0], extentKm: 1, localExtentKm: 1, nearSurface: false, groundRadius: MOON_KM, gridKm: 0,
    landingUp: new THREE.Vector3(0, 1, 0), up: new THREE.Vector3(0, 1, 0)};

  function sampleAt(t) {
    const s = state.data.samples, i = indexAt(state.times, t), j = Math.min(i + 1, s.length - 1);
    const f = s[j].t > s[i].t ? clamp((t - s[i].t) / (s[j].t - s[i].t), 0, 1) : 0;
    const mix = (a, b) => a.map((v, k) => v + (b[k] - v) * f);
    return {index: f > 0.5 ? j : i, r: mix(s[i].r, s[j].r), moon: mix(s[i].moon, s[j].moon), v: mix(s[i].v, s[j].v),
      target: s[i].target ? mix(s[i].target, s[j].target) : null, stage: s[i].stage,
      utc: Number.isFinite(s[i].utc) ? s[i].utc + (s[j].utc - s[i].utc) * f : NaN};
  }

  function rebuildControls() {
    const target = controls ? controls.target.clone() : new THREE.Vector3();
    controls?.dispose();
    camera.up.copy(state.up);
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    Object.assign(controls, {enableDamping: true, dampingFactor: 0.085, enableZoom: false, rotateSpeed: 0.8, panSpeed: 0.8, minPolarAngle: 0.02, maxPolarAngle: Math.PI - 0.02});
    controls.target.copy(target);
    controls.enabled = mode === 'telemetry';
  }

  /** Ground patch at the true local radius, so near-surface tracks sit on real curvature with a metric grid. */
  function buildGround(direction, radius, capKm, gridKm) {
    const c = direction.clone().normalize();
    const helper = Math.abs(c.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0);
    const e = helper.cross(c).normalize(), n = c.clone().cross(e);
    const rings = 90, segments = 128, alpha = capKm / radius, positions = [0, 0, 0], uvs = [0, 0], index = [];
    for (let i = 1; i <= rings; i++) {
      const rho = alpha * (i / rings) ** 2, sr = Math.sin(rho), cr = Math.cos(rho);
      for (let j = 0; j < segments; j++) {
        const theta = 2 * Math.PI * j / segments, ct = Math.cos(theta), st = Math.sin(theta);
        positions.push(radius * (c.x * (cr - 1) + (e.x * ct + n.x * st) * sr), radius * (c.y * (cr - 1) + (e.y * ct + n.y * st) * sr), radius * (c.z * (cr - 1) + (e.z * ct + n.z * st) * sr));
        uvs.push(radius * sr * ct / gridKm, radius * sr * st / gridKm);
      }
    }
    for (let j = 0; j < segments; j++) index.push(0, 1 + j, 1 + (j + 1) % segments);
    for (let i = 1; i < rings; i++) for (let j = 0; j < segments; j++) {
      const a = 1 + (i - 1) * segments + j, b = 1 + (i - 1) * segments + (j + 1) % segments, c2 = 1 + i * segments + j, d = 1 + i * segments + (j + 1) % segments;
      index.push(a, c2, b, b, c2, d);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geometry.setIndex(index); geometry.computeVertexNormals();
    const grid = canvasTexture(128, (x, s) => { x.fillStyle = '#55595d'; x.fillRect(0, 0, s, s); x.fillStyle = '#a9c6da'; x.fillRect(0, 0, s, 2); x.fillRect(0, 0, 2, s); });
    grid.wrapS = grid.wrapT = THREE.RepeatWrapping;
    const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({map: grid, emissive: 0x3a7390, emissiveMap: grid, emissiveIntensity: 0.55, roughness: 1, metalness: 0}));
    mesh.userData.moonOffset = scale3([c.x, c.y, c.z], radius);
    return mesh;
  }

  function lineFrom(points, colors, opacity = 1) {
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const p of points) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
    const origin = lo.map((v, k) => (v + hi[k]) / 2), array = new Float32Array(points.length * 3);
    points.forEach((p, i) => { array[i * 3] = p[0] - origin[0]; array[i * 3 + 1] = p[1] - origin[1]; array[i * 3 + 2] = p[2] - origin[2]; });
    const geometry = new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(array, 3));
    if (colors) geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({vertexColors: !!colors, color: colors ? 0xffffff : 0x8aa0b5, transparent: opacity < 1, opacity, toneMapped: false}));
    line.userData.origin = origin; line.frustumCulled = false;
    return {line, origin, extent: Math.max(1e-6, norm3(sub3(hi, lo)) / 2)};
  }

  const moonFixed = item => { item.line.userData.moonOffset = item.origin; delete item.line.userData.origin; return item.line; };

  function rebuild() {
    disposeAll(dynamic);
    const samples = state.data.samples, last = samples[samples.length - 1];
    const eci = samples.map(s => s.r), local = samples.map(s => sub3(s.r, s.moon));
    let minRadius = Infinity, maxRadius = 0;
    for (const p of local) { const r = norm3(p); minRadius = Math.min(minRadius, r); maxRadius = Math.max(maxRadius, r); }
    state.nearSurface = minRadius < MOON_KM + 150;
    // Long tracks (e.g. a whole Earth-Moon mission) are framed whole and drawn as lines only.
    state.longTrack = maxRadius > 30 * MOON_KM;
    let lowest = 0;
    local.forEach((p, i) => { if (norm3(p) < norm3(local[lowest])) lowest = i; });
    state.landingUp = vec(local[lowest]).normalize();
    // Near-surface part only, so the lunar ground view is not swamped by the rest of the mission.
    const nearIdx = local.map((p, i) => i).filter(i => norm3(local[i]) < MOON_KM + 200);
    const nearLocal = state.nearSurface ? nearIdx.map(i => local[i]) : local;

    // ECI track, coloured by stage: the 1 px line covers zoomed-out views, a thin tube per stage reads up close.
    const colors = [];
    for (const s of samples) { const c = new THREE.Color(stageColor(s.stage)); colors.push(c.r, c.g, c.b); }
    const track = lineFrom(eci, colors), relative = lineFrom(nearLocal.length > 1 ? nearLocal : local, null);
    state.center = track.origin; state.extentKm = track.extent; state.localExtentKm = relative.extent;
    dynamic.push(track.line);
    const tubeRadius = (state.nearSurface ? Math.min(state.extentKm, state.localExtentKm) : state.extentKm) * 0.005;
    let start = 0;
    for (let i = 1; i <= samples.length && !state.longTrack; i++) {
      if (i < samples.length && samples[i].stage === samples[start].stage) continue;
      const points = eci.slice(Math.max(0, start - 1), i).map(p => vec(sub3(p, track.origin)));
      if (points.length > 1) {
        const mesh = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points, false, 'centripetal'), Math.min(1200, points.length * 2), tubeRadius, 8, false),
          new THREE.MeshBasicMaterial({color: stageColor(samples[start].stage), toneMapped: false}));
        mesh.userData.origin = track.origin; dynamic.push(mesh);
      }
      start = i;
    }

    // The Moon's own ECI path over the same time span.
    const moonPath = samples.map(s => s.moon);
    if (norm3(sub3(moonPath[0], moonPath[moonPath.length - 1])) > 1) dynamic.push(lineFrom(moonPath, null, 0.35).line);

    if (state.nearSurface) {
      let ground = minRadius;
      const landing = samples[lowest];
      if (landing.target) ground = Math.min(ground, norm3(sub3(landing.target, landing.moon)));
      ground = clamp(ground, MOON_KM - 12, MOON_KM + 12);
      const maxHeight = Math.max(0.05, maxRadius - ground);
      const capKm = clamp(Math.max(Math.sqrt(2 * ground * maxHeight) * 1.6, state.localExtentKm * 6), 8, 600);
      const steps = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25];
      state.gridKm = steps.find(step => step >= state.localExtentKm / 6) || 25;
      state.groundRadius = ground;
      dynamic.push(buildGround(state.landingUp, ground, capKm, state.gridKm));
      // The descent as seen from the lunar surface (rides along with the Moon), plus its ground track.
      relative.line.material.color.set(0xdfefff); relative.line.material.transparent = true; relative.line.material.opacity = 0.5;
      dynamic.push(moonFixed(relative));
      const shadow = lineFrom((nearLocal.length > 1 ? nearLocal : local).map(p => scale3(p, (ground + 0.0003) / norm3(p))), null, 0.6);
      shadow.line.material.color.set(0x0b1a26);
      dynamic.push(moonFixed(shadow));
    } else {
      relative.line.geometry.dispose(); relative.line.material.dispose();
    }
    moon.surface.scale.setScalar(state.nearSurface ? state.groundRadius - 0.5 : MOON_KM);
    for (const object of dynamic) scene.add(object);
  }

  const upFor = focus => state.nearSurface && (focus === 'track' || focus === 'vehicle') ? state.landingUp.clone() : ECLIPTIC_NORTH.clone();
  function anchor(now) {
    if (state.focus === 'vehicle') return now.r;
    if (state.focus === 'moon') return now.moon;
    if (state.focus === 'earth') return [0, 0, 0];
    return state.center;
  }

  function frameCamera(distance) {
    state.up = upFor(state.focus);
    rebuildControls();
    const samples = state.data.samples, first = samples[0], last = samples[samples.length - 1];
    // Look across the direction of travel: over the surface for landings, through space otherwise.
    const motion = vec(state.nearSurface ? sub3(sub3(last.r, last.moon), sub3(first.r, first.moon)) : sub3(last.r, first.r));
    motion.addScaledVector(state.up, -motion.dot(state.up));
    const side = motion.lengthSq() > 1e-12 ? state.up.clone().cross(motion).normalize() : new THREE.Vector3(1, 0, 0).cross(state.up).normalize();
    controls.target.set(0, 0, 0);
    camera.position.copy(side.addScaledVector(state.up, 0.22).normalize().multiplyScalar(distance));
    controls.update();
  }
  function focusDistance(focus) {
    return {track: Math.max(state.extentKm * 3.6, 0.25), vehicle: Math.max((state.nearSurface ? state.localExtentKm : state.extentKm) * 3, 0.15),
      moon: MOON_KM * 3.2, earth: EARTH_KM * 3.2}[focus];
  }

  function load(data) {
    state.data = data;
    state.times = data.samples.map(s => s.t);
    state.t = state.times[0];
    state.playing = true;
    const first = data.samples[0];
    // Tidally locked Moon: IAU pole, prime meridian toward Earth at the first sample (libration ignored).
    const pole = vec(sim.body_frames?.moon?.z || [0, 0, 1]).normalize(), toEarth = vec(scale3(first.moon, -1));
    const x = toEarth.addScaledVector(pole, -toEarth.dot(pole)).normalize(), y = pole.clone().cross(x);
    orient(moon.group, {x: [x.x, x.y, x.z], y: [y.x, y.y, y.z], z: [pole.x, pole.y, pole.z]});
    rebuild();
    // Landings: follow the vehicle (the Moon carries it ~1 km/s through ECI); otherwise frame the whole track.
    if (data.sun) { sunDirection.copy(data.sun); sunLight.position.copy(data.sun); }
    state.focus = state.nearSurface && !state.longTrack ? 'vehicle' : 'track';
    frameCamera(focusDistance(state.focus));
    const span = state.times[state.times.length - 1] - state.times[0];
    state.rate = TELEMETRY_SPEEDS.reduce((best, r) => Math.abs(Math.log(span / r / 20)) < Math.abs(Math.log(span / best / 20)) ? r : best, 1);
  }
  function setFocus(focus) {
    if (!state.data) return;
    state.focus = focus;
    state.up = upFor(focus);
    const direction = camera.position.clone().sub(controls.target).normalize();
    rebuildControls();
    controls.target.set(0, 0, 0);
    camera.position.copy(direction.multiplyScalar(focusDistance(focus)));
    controls.update();
  }

  function update(dt) {
    if (!state.data) return;
    const t0 = state.times[0], t1 = state.times[state.times.length - 1];
    if (state.playing) {
      state.t += dt * state.rate;
      if (state.t >= t1) { state.t = t1; state.playing = false; }
    }
    const now = sampleAt(state.t), origin = anchor(now);
    const place = (object, p) => object.position.set(p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]);
    const onMoon = offset => [now.moon[0] + offset[0], now.moon[1] + offset[1], now.moon[2] + offset[2]];
    place(earth.group, [0, 0, 0]); // ECI: Earth's centre is the origin and never moves
    if (state.data.hasUtc) orient(earth.group, earthFrameAt(now.utc)); // ...but it spins, using the CSV's own clock
    place(moon.group, now.moon);
    for (const object of dynamic) place(object, object.userData.moonOffset ? onMoon(object.userData.moonOffset) : object.userData.origin);
    place(craft, now.r); place(craftCore, now.r);
    const target = now.target || state.data.samples[state.data.samples.length - 1].target;
    targetMark.visible = !!target;
    if (target) place(targetMark, target);
    dropLine.visible = state.nearSurface;
    if (state.nearSurface) {
      const local = sub3(now.r, now.moon), ground = onMoon(scale3(local, state.groundRadius / norm3(local)));
      dropLine.geometry.attributes.position.array.set([now.r[0] - origin[0], now.r[1] - origin[1], now.r[2] - origin[2], ground[0] - origin[0], ground[1] - origin[1], ground[2] - origin[2]]);
      dropLine.geometry.attributes.position.needsUpdate = true;
    }
    stars.position.copy(camera.position);
    sunDirView.value.copy(sunDirection).transformDirection(camera.matrixWorldInverse);
    state.now = {...now, origin, progress: (state.t - t0) / Math.max(1e-9, t1 - t0)};
  }

  function heightText(now) {
    const fromMoon = norm3(sub3(now.r, now.moon));
    if (state.nearSurface) return `h ${fmt.length(Math.max(0, fromMoon - state.groundRadius))}`;
    const fromEarth = norm3(now.r);
    return fromMoon < fromEarth ? `${fmt.length(fromMoon - MOON_KM)} above Moon` : `${fmt.length(fromEarth - EARTH_KM)} above Earth`;
  }

  function labelItems() {
    const now = state.now;
    if (!now) return [];
    const items = [], right = cameraRight(camera), at = p => vec(sub3(p, now.origin));
    const distanceTo = p => camera.position.distanceTo(at(p));
    if (distanceTo([0, 0, 0]) > EARTH_KM * 1.6) items.push({key: 't-earth', kind: 'body', text: 'Earth', at: at([0, 0, 0]).addScaledVector(right, EARTH_KM * 1.15)});
    if (distanceTo(now.moon) > MOON_KM * 1.6) items.push({key: 't-moon', kind: 'body', text: 'Moon', at: at(now.moon).addScaledVector(right, MOON_KM * 1.15)});
    const samples = state.data.samples;
    items.push({key: 't-start', text: 'Start', at: at(samples[0].r)});
    items.push({key: 't-end', text: 'End', at: at(samples[samples.length - 1].r)});
    if (targetMark.visible) items.push({key: 't-target', text: 'Target', at: targetMark.position.clone()});
    items.push({key: 't-craft', kind: 'craft', at: craft.position.clone(),
      text: `t ${fmt.seconds(state.t - state.times[0])} · ${heightText(now)} · ${norm3(now.v).toFixed(3)} km/s`});
    return items;
  }

  return {
    scene, camera, state, load, setFocus, update, labelItems,
    moonOrientation: () => moon.group.quaternion.clone(),
    get controls() { return controls; },
    home() { if (state.data) { state.focus = state.nearSurface && !state.longTrack ? 'vehicle' : 'track'; frameCamera(focusDistance(state.focus)); } },
    seek(t) { if (state.data) { state.t = clamp(t, state.times[0], state.times[state.times.length - 1]); } },
  };
})();

/* =====================================================================
 * Trajectory CSV parsing
 * ===================================================================== */
function stageColor(stage) { return PHASE_COLORS[((stage % PHASE_COLORS.length) + PHASE_COLORS.length) % PHASE_COLORS.length]; }
function stageName(stage) {
  const named = ui.telemetry?.stageNames?.[stage];
  if (named) return named;
  return STAGE_NAMES[stage] ? `${STAGE_NAMES[stage]} (stage ${stage})` : `Stage ${stage}`;
}

function parseCsvRecords(text) {
  const records = []; let record = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"' && field.length === 0) quoted = true;
    else if (ch === ',') { record.push(field); field = ''; }
    else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      record.push(field); records.push(record); record = []; field = '';
    } else field += ch;
  }
  if (quoted) throw new Error('CSV contains an unclosed quoted field.');
  if (field.length || record.length) { record.push(field); records.push(record); }
  return records.filter(row => row.some(value => value.trim() !== ''));
}

function parseTrajectory(text) {
  const rows = parseCsvRecords(text);
  if (rows.length < 3) throw new Error('CSV needs a header row and at least two samples.');
  const header = rows.shift().map(name => name.trim().replace(/^﻿/, '').toLowerCase());
  if (new Set(header).size !== header.length) throw new Error('CSV has duplicate column names.');
  const col = Object.fromEntries(header.map((name, i) => [name, i]));
  const missing = ['t', 'rx', 'ry', 'rz'].filter(name => !(name in col));
  if (missing.length) throw new Error(`Missing required columns: ${missing.join(', ')}.`);
  const group = names => {
    const present = names.filter(name => name in col);
    if (present.length && present.length !== names.length) throw new Error(`Columns ${names.join(', ')} must be supplied together.`);
    return present.length ? names : null;
  };
  const velocity = group(['vx', 'vy', 'vz']), moonCols = group(['moon_rx', 'moon_ry', 'moon_rz']), targetCols = group(['target_rx', 'target_ry', 'target_rz']);
  const accelCols = group(['ax', 'ay', 'az']), sunCols = group(['sun_rx', 'sun_ry', 'sun_rz']);
  const textColumn = name => name in col ? col[name] : null;
  const phaseCol = textColumn('phase'), utcCol = textColumn('utc');
  const stageNames = {};
  const optional = name => name in col ? name : null;
  const massCol = optional('mass'), thrustCol = optional('thrust_n'), stageCol = optional('stage_index'), altitudeCol = optional('altitude'), targetDistanceCol = optional('target_distance');
  const fallbackMoon = sub3(sim.bodies.moon.position_km, sim.bodies.earth.position_km);

  let previous = -Infinity;
  const samples = rows.map((row, i) => {
    const line = i + 2;
    if (row.length !== header.length) throw new Error(`Row ${line} has ${row.length} values; expected ${header.length}.`);
    const read = (name, allowEmpty = false) => {
      const raw = (row[col[name]] ?? '').trim();
      if (raw === '') { if (allowEmpty) return null; throw new Error(`Row ${line}: "${name}" is empty.`); }
      const value = Number(raw);
      if (!Number.isFinite(value)) throw new Error(`Row ${line}: "${name}" is not a number.`);
      return value;
    };
    const km = names => names.map(name => read(name) / 1000);
    const sample = {
      t: read('t'), r: km(['rx', 'ry', 'rz']),
      v: velocity ? km(velocity) : null,
      moon: moonCols ? km(moonCols) : fallbackMoon,
      target: targetCols ? km(targetCols) : null,
      stage: stageCol ? (read(stageCol, true) ?? 0) : 0,
      mass: massCol ? read(massCol, true) : null,
      thrust: thrustCol ? read(thrustCol, true) : null,
      altitude: altitudeCol ? read(altitudeCol, true) : null,
      mdot: 'mdot' in col ? read('mdot', true) : null,
      accel: accelCols ? accelCols.map(name => read(name)) : null,
      targetDistance: targetDistanceCol ? read(targetDistanceCol, true) : null,
      utc: utcCol !== null ? Date.parse((row[utcCol] || '').trim()) : NaN,
    };
    if (phaseCol !== null && row[phaseCol].trim() && stageNames[sample.stage] === undefined) stageNames[sample.stage] = row[phaseCol].trim();
    if (i === 0 && sunCols) sample.sun = sunCols.map(name => read(name));
    if (sample.t <= previous) throw new Error(`Row ${line}: time must increase (t=${sample.t}).`);
    if (!Number.isInteger(sample.stage) || sample.stage < 0) throw new Error(`Row ${line}: stage_index must be a whole number ≥ 0.`);
    if (norm3(sample.r) === 0) throw new Error(`Row ${line}: zero position vector.`);
    previous = sample.t;
    return sample;
  });
  if (!velocity) samples.forEach((s, i) => {
    const a = samples[Math.max(0, i - 1)], b = samples[Math.min(samples.length - 1, i + 1)];
    s.v = scale3(sub3(b.r, a.r), 1 / (b.t - a.t));
  });
  if (norm3(samples[0].moon) <= EARTH_KM) throw new Error('Moon position must lie outside the Earth.');
  // A "phase" column without stage_index: number the phases in order of appearance.
  if (phaseCol !== null && !stageCol) {
    const order = new Map();
    samples.forEach((sample, i) => {
      const name = rows[i][phaseCol].trim() || 'Flight';
      if (!order.has(name)) order.set(name, order.size);
      sample.stage = order.get(name);
      stageNames[sample.stage] = name;
    });
  }
  const hasUtc = samples.every(sample => Number.isFinite(sample.utc));
  return {samples, hasMoon: !!moonCols, hasVelocity: !!velocity, hasTarget: !!targetCols, stageNames, hasUtc,
    sun: samples[0].sun ? vec(samples[0].sun).normalize() : null};
}

/* =====================================================================
 * Application state & UI
 *
 * The site displays results produced elsewhere:
 *   - the landing-site classifier's output files (classifier/.../python/outputs) for Sites 1, 4 and 23;
 *   - CSV files you upload: trajectories (t, rx, ry, rz, ...) or result tables with lat/lon columns.
 * Nothing is computed here beyond what is needed to draw the data.
 * ===================================================================== */
let mode = 'scene';
const ui = {
  selected: null, telemetry: null, csvName: '',
  layers: {candidates: true, historic: true, results: true, labels: true, sun: true},
  mission: null, missionFilter: 'all', resultSets: [],
};
const isAnalysed = site => cfg.sites.some(s => s.id === site?.id);
const activeView = () => mode === 'telemetry' ? T : P;
const element = (tag, className, text) => { const el = document.createElement(tag); if (className) el.className = className; if (text != null) el.textContent = text; return el; };

// The LunAI server (when running) serves the classifier's figures and stored mission-path CSVs.
const api = (() => {
  let base = null;
  try { base = window.parent.location.origin; } catch { base = location.origin; }
  if (!base || base === 'null' || base.startsWith('file')) base = null;
  return {base, url: path => (base ? base + path : null)};
})();

// ---------- Plain-language helpers ----------
const GLOSSARY = {
  help: 'Explanations like this one appear wherever you see a ? mark.',
  slope: 'How steep the ground is. 0° is perfectly flat. Landers usually need less than about 8–10°.',
  roughness: 'How bumpy the ground is: the typical height difference between nearby points, in metres.',
  flat: 'How much of the area has slopes of 8° or less, gentle enough for a lander.',
  shelter: 'How much of the area sits in hollows at least 30 m below nearby ground. Hollows can stay cold and shadowed, which is where ice might hide. It is a terrain clue, not proof of ice.',
  nearShelter: 'Flat ground within 1 km of a sheltered hollow: a place to land near possible ice.',
  elevation: 'Height compared with the Moon’s average surface level.',
  coordinates: 'Latitude runs from −90 (south pole) to 90 (north pole). Longitude runs from −180 to 180. On the Moon, 0° longitude is the middle of the side facing Earth. You can type 89.5S or -89.5.',
  sunElevation: 'How high the Sun is above the horizon. Near the poles it always stays very low, which makes long shadows.',
  earthComms: 'Whether Earth is above the horizon. If it isn’t, a lander can’t radio Earth directly and needs a relay satellite.',
  solarPower: 'Electricity a 12 m² solar panel (28% efficient, tilted toward the Sun) would make right now.',
  speed: 'How fast the spacecraft is moving, measured from Earth’s centre (from the CSV’s velocity columns).',
  csv: 'Upload a CSV made by another program. Trajectories (t, rx, ry, rz …) replay at true scale; tables with lat/lon columns appear as spots on the Moon.',
  landingSpots: 'The classifier’s top-ranked spots in this region, as listed in its output file. Its score is 60% safe ground, 30% Earth in view (radio contact) and 10% nearness to possible ice, from 0 to 1.',
};
function verdictFor(location) {
  const given = location?.classification?.verdict;
  if (!given) return null;
  return {...given, mark: {good: '✓', care: '!', risky: '✕', unknown: '?'}[given.level] || '?'};
}
function renderVerdict(box, location) {
  const v = verdictFor(location);
  box.replaceChildren();
  box.className = v ? `verdict ${v.level}` : 'verdict';
  if (!v) return;
  box.append(element('div', 'mark', v.mark), element('strong', '', v.title), element('span', '', v.text));
}
/** Parse "89.5", "-89.5", "89.5S", "45°E", "45 W". */
function parseCoordinate(text, kind) {
  const raw = String(text ?? '').trim().toUpperCase().replace(/°/g, '');
  if (!raw) return null;
  const match = raw.match(/^([NSEW])?\s*([+-]?\d+(?:\.\d+)?)\s*([NSEW])?$/);
  if (!match) throw new Error(`“${text}” isn’t a ${kind} I understand. Try something like ${kind === 'latitude' ? '-89.5 or 89.5S' : '45 or 45E'}.`);
  const hemisphere = match[1] || match[3];
  let value = Number(match[2]);
  if (hemisphere) {
    if ((kind === 'latitude') !== 'NS'.includes(hemisphere)) throw new Error(`Use ${kind === 'latitude' ? 'N or S' : 'E or W'} for ${kind}.`);
    if ('SW'.includes(hemisphere)) value = -Math.abs(value);
  }
  const limit = kind === 'latitude' ? 90 : 180;
  if (Math.abs(value) > limit) throw new Error(`${kind[0].toUpperCase() + kind.slice(1)} must be between -${limit} and ${limit}.`);
  return value;
}
const coordKey = p => `${(+p.lat).toFixed(4)},${(+p.lon).toFixed(4)}`;

// ---------- Markers on the Moon ----------
const missions = cfg.missions?.missions || [];
for (const site of cfg.sites) P.markers.candidates.set(site.id, P.addMarker(P.moon, {...site, siteId: site.id}, 0xffcf77, 0.032));
for (const mission of missions) {
  if (!mission.site) continue;
  const color = mission.status === 'planned' ? 0xc99bff : mission.crewed ? 0xffffff : 0x9aa7b4;
  P.markers.historic.push(P.addMarker(P.moon, {...mission.site, name: mission.name, kind: mission.type, missionId: mission.id}, color, mission.crewed ? 0.024 : 0.018));
}
const pickable = () => [...P.markers.candidates.values(), ...P.markers.historic, ...P.markers.results].filter(m => m.visible).concat([P.earth.surface, P.moon.surface]);

// ---------- Selection ----------
/** Show a location in the Spot tab: an analysed site, one of its listed spots, an uploaded row, or a typed/clicked point. */
function select(location, {focus = false, open = true} = {}) {
  if (mode === 'telemetry') clearTelemetry();
  ui.selected = location;
  for (const [id, marker] of P.markers.candidates) marker.material.color.set(id === (location.siteId || location.parentSite) ? '#ff6fd8' : '#ffcf77');
  renderSites();
  renderSpot(location);
  showEnvironment();
  showPlacePhotos(location);
  updateHeader();
  if (focus) P.focusMoonPoint(location.lat, location.lon);
  if (open) openRight('site');
}
function updateHeader() {
  if (mode === 'telemetry') {
    setText('missionTitle', ui.csvName);
    setText('missionSubtitle', `${ui.telemetry.samples.length.toLocaleString()} samples · true scale`);
    return;
  }
  const site = ui.selected;
  setText('missionTitle', site ? site.name : 'Lunar South Pole');
  setText('missionSubtitle', site ? fmt.latlon(+site.lat, +site.lon) : 'Classifier results & uploaded CSVs');
}

// ---------- Left panel: classifier regions and uploaded result tables ----------
function siteCard({title, meta, sub, color, selected, onClick, onRemove}) {
  const card = element('button', `card${selected ? ' selected' : ''}`);
  card.type = 'button';
  card.style.setProperty('--card-color', color);
  const top = element('div', 'card-top');
  top.append(element('strong', '', title));
  if (onRemove) {
    const remove = element('span', 'link', 'Remove');
    remove.addEventListener('click', event => { event.stopPropagation(); onRemove(); });
    top.appendChild(remove);
  }
  card.append(top, element('div', 'card-meta', meta));
  if (sub) card.appendChild(element('div', 'card-route', sub));
  card.addEventListener('click', onClick);
  return card;
}
function renderSites() {
  const selectedRegion = ui.selected?.siteId || ui.selected?.parentSite;
  const cards = cfg.sites.map((site, i) => {
    const c = site.classification, best = c?.candidates?.[0];
    return siteCard({
      title: site.name, color: SITE_COLORS[i % SITE_COLORS.length], selected: selectedRegion === site.id,
      meta: c ? `${verdictFor(site)?.title || c.terrain_class} · ${c.candidates?.length || 0} spots listed` : 'No classifier results',
      sub: best ? `best score ${best.score.toFixed(2)} · ${fmt.latlon(best.lat, best.lon)}` : '',
      onClick: () => select({...site, siteId: site.id}, {focus: true}),
    });
  });
  const uploaded = ui.resultSets.map(set => siteCard({
    title: set.name, color: '#5fd4ff', selected: selectedRegion === set.id,
    meta: `Uploaded CSV · ${set.rows.length} spot${set.rows.length === 1 ? '' : 's'}`,
    sub: set.rows[0] ? `first: ${fmt.latlon(set.rows[0].lat, set.rows[0].lon)}` : '',
    onClick: () => select(set.summary, {focus: true}),
    onRemove: () => removeResultSet(set.id),
  }));
  $('candidateList').replaceChildren(...cards, ...uploaded);
}

// ---------- Details: Spot tab ----------
function helpIcon(term) { const icon = element('i', 'help', '?'); icon.dataset.term = term; icon.tabIndex = 0; return icon; }
function metricTile(label, value, sub, term) {
  const tile = element('div', 'metric'), l = element('span', '', label);
  if (term) l.appendChild(helpIcon(term));
  tile.append(l, element('strong', '', value));
  if (sub) tile.appendChild(element('small', '', sub));
  return tile;
}
function rowsFrom(list) {
  return list.map(({label, value}) => { const row = element('div', 'row'); row.append(element('span', '', label), element('strong', '', value)); return row; });
}
function renderSpot(location) {
  setText('siteName', location.name || 'Point on the Moon');
  setText('siteCoords', fmt.latlon(+location.lat, +location.lon));
  const region = cfg.sites.find(s => s.id === location.siteId);
  const classification = location.classification;
  renderVerdict($('siteVerdict'), location);

  // Classifier summary (score breakdown + details), exactly as in its output files.
  $('classBlock').hidden = !classification || !Object.keys(classification.class_scores || {}).length;
  if (!$('classBlock').hidden) {
    setText('siteClass', classification.terrain_class);
    setText('siteConfidence', '');
    const entries = Object.entries(classification.class_scores || {});
    $('classBar').replaceChildren(...entries.map(([name, score], i) => {
      const segment = element('i'); segment.style.width = `${score * 100}%`; segment.style.background = CLASS_COLORS[i % CLASS_COLORS.length]; segment.title = `${name} ${score.toFixed(2)}`; return segment;
    }));
    $('classLegend').replaceChildren(...entries.map(([name, score], i) => {
      const item = element('span'), dot = element('i'); dot.style.background = CLASS_COLORS[i % CLASS_COLORS.length]; item.append(dot, `${name} ${score.toFixed(2)}`); return item;
    }));
    setText('classModel', `${classification.score_label} · ${classification.model}`);
  }
  $('classDetails').replaceChildren(...rowsFrom(location.fields || classification?.details || []));
  $('classDetails').hidden = !$('classDetails').children.length;

  // LOLA terrain summary for the three analysed regions.
  const metrics = $('siteMetrics');
  metrics.replaceChildren();
  if (region) {
    metrics.append(
      metricTile('Height', `${(+region.median_elevation_m).toFixed(0)} m`, 'typical', 'elevation'),
      metricTile('Steepness', `${(+region.median_slope_deg).toFixed(1)}°`, region.p90_slope_deg != null ? `steepest 10%: ${(+region.p90_slope_deg).toFixed(0)}°` : 'typical', 'slope'),
      metricTile('Bumpiness', `${(+region.median_roughness_m).toFixed(1)} m`, 'typical', 'roughness'),
      metricTile('Flat ground', `${(+region.low_slope_pct).toFixed(0)}%`, 'of the area', 'flat'),
      metricTile('Shelter', `${(+region.shadow_proxy_pct).toFixed(0)}%`, 'of the area', 'shelter'),
      metricTile('Flat & sheltered', `${(+region.landable_near_shelter_pct).toFixed(0)}%`, 'of the area', 'nearShelter'),
    );
  }
  metrics.hidden = !metrics.children.length;

  renderSpotList(location);
  const nothing = !classification && !location.fields?.length;
  $('pointNote').hidden = !nothing;
  setText('pointNote', nothing ? 'No classifier results at this point. Results exist for Sites 1, 4 and 23, and for spots in CSVs you upload.' : '');
}
/** Spots listed for a region (the classifier's candidates) or an uploaded table, plus the classifier's own figure. */
function renderSpotList(location) {
  const regionId = location.siteId || location.parentSite;
  const region = cfg.sites.find(s => s.id === regionId);
  const set = ui.resultSets.find(r => r.id === regionId);
  const spots = region?.classification?.candidates || set?.rows || [];
  $('landingSpots').hidden = !spots.length;
  if (!spots.length) return;
  setText('spotsTitle', set ? `Spots in ${set.name}` : 'Best landing spots');
  const figure = region?.classification?.map?.figure;
  const figureUrl = figure ? api.url(`/api/classifier-figure/${figure}`) : null;
  $('figureWrap').hidden = !figureUrl;
  if (figureUrl && $('classifierFigure').dataset.src !== figureUrl) {
    $('classifierFigure').src = figureUrl;
    $('classifierFigure').dataset.src = figureUrl;
  }
  const pois = region?.classification?.map?.pois || [];
  setText('poiList', pois.length ? `Nearby named craters: ${pois.map(p => `${p.name} (${fmt.latlon(p.lat, p.lon)})`).join(' · ')}` : '');
  $('landingSpotList').replaceChildren(...spots.map((spot, i) => {
    const row = element('div', `spot${coordKey(spot) === coordKey(location) ? ' current' : ''}`);
    const text = element('div');
    const score = spot.score != null ? ` · score ${(+spot.score).toFixed(2)}` : '';
    const fields = (spot.details || spot.fields || []).slice(0, 5).map(d => `${d.label} ${d.value}`).join(' · ');
    text.append(element('b', '', `${spot.label || `Spot ${i + 1}`}${score}`), element('small', '', `${fmt.latlon(spot.lat, spot.lon)}${fields ? ` · ${fields}` : ''}`));
    const go = element('button', 'btn ghost small', 'Go');
    go.addEventListener('click', () => select({
      name: `${region?.name || set.name} · ${spot.label || `Spot ${i + 1}`}`, lat: spot.lat, lon: spot.lon, parentSite: regionId,
      fields: [...(spot.score != null ? [{label: 'Score', value: (+spot.score).toFixed(2)}] : []), ...(spot.details || spot.fields || [])],
    }, {focus: true}));
    row.append(element('span', 'rank', String(spot.rank ?? i + 1)), text, go);
    return row;
  }));
}
$('classifierFigure').addEventListener('click', () => { $('figureLarge').src = $('classifierFigure').src; $('figureModal').hidden = false; });
$('classifierFigure').addEventListener('error', () => { $('figureWrap').hidden = true; });

// ---------- Photos: LRO orbital mosaic (NASA Moon Trek) and the NASA Image Library ----------
const WAC = 'https://trek.nasa.gov/tiles/Moon/EQ/LRO_WAC_Mosaic_Global_303ppd_v02/1.0.0/default/default028mm';
/** Real orbital photo around a point: LRO Wide Angle Camera mosaic, ~100 m/pixel. Not usable near the poles. */
function drawOrbitalPhoto(lat, lon) {
  const canvas = $('orbitalPhoto');
  if (Math.abs(lat) > 80) {
    canvas.hidden = true;
    setText('orbitalNote', 'Orbital photos aren’t available this close to the pole (the map tiles don’t cover it well).');
    return;
  }
  canvas.hidden = false;
  const z = 7, tileDeg = 180 / 2 ** z, size = 512, spanDeg = 0.9;  // about 27 km tall
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.height = size;
  ctx.fillStyle = '#111'; ctx.fillRect(0, 0, size, size);
  const lonSpan = spanDeg / Math.max(0.2, Math.cos(lat * Math.PI / 180));
  const pxPerLat = size / spanDeg, pxPerLon = size / lonSpan;
  const rows = [Math.floor((90 - (lat + spanDeg / 2)) / tileDeg), Math.floor((90 - (lat - spanDeg / 2)) / tileDeg)];
  const cols = [Math.floor((lon - lonSpan / 2 + 180) / tileDeg), Math.floor((lon + lonSpan / 2 + 180) / tileDeg)];
  const key = coordKey({lat, lon});
  canvas.dataset.key = key;
  for (let r = rows[0]; r <= rows[1]; r++) for (let c = cols[0]; c <= cols[1]; c++) {
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.onload = () => {
      if (canvas.dataset.key !== key) return;
      const tileLat = 90 - r * tileDeg, tileLon = c * tileDeg - 180;
      const x = (tileLon - (lon - lonSpan / 2)) * pxPerLon, y = ((lat + spanDeg / 2) - tileLat) * pxPerLat;
      ctx.drawImage(image, x, y, tileDeg * pxPerLon, tileDeg * pxPerLat);
      ctx.strokeStyle = '#ff6fd8'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(size / 2 - 12, size / 2); ctx.lineTo(size / 2 + 12, size / 2); ctx.moveTo(size / 2, size / 2 - 12); ctx.lineTo(size / 2, size / 2 + 12); ctx.stroke();
    };
    image.src = `${WAC}/${z}/${r}/${((c % 2 ** (z + 1)) + 2 ** (z + 1)) % 2 ** (z + 1)}.jpg`;
  }
  setText('orbitalNote', 'Real orbital photo: LRO Wide Angle Camera mosaic, about 100 m per pixel, roughly 27 km across (NASA/GSFC/Arizona State University, via NASA Moon Trek). The pink cross marks the point.');
}
const photoCache = new Map();
async function nasaPhotos({q, year_start, year_end}) {
  const params = new URLSearchParams({q, media_type: 'image', page_size: '9'});
  if (year_start) { params.set('year_start', year_start); params.set('year_end', year_end || year_start); }
  const url = `https://images-api.nasa.gov/search?${params}`;
  if (!photoCache.has(url)) photoCache.set(url, fetch(url).then(r => r.json()).then(d => d.collection.items.map(item => ({
    id: item.data[0].nasa_id, title: item.data[0].title, date: (item.data[0].date_created || '').slice(0, 10), thumb: item.links?.[0]?.href,
  })).filter(p => p.thumb)).catch(() => []));
  return photoCache.get(url);
}
function renderPhotoGrid(grid, note, photos, label) {
  grid.replaceChildren(...photos.slice(0, 9).map(photo => {
    const link = element('a');
    link.href = `https://images.nasa.gov/details/${encodeURIComponent(photo.id)}`; link.target = '_blank'; link.rel = 'noopener';
    link.title = `${photo.title} (${photo.date})`;
    const image = element('img'); image.src = photo.thumb; image.alt = photo.title; image.loading = 'lazy';
    link.appendChild(image);
    return link;
  }));
  setText(note.id, photos.length ? `${label} From the NASA Image and Video Library; click a photo to open it.` : `No NASA Image Library photos found for ${label.toLowerCase().replace(/[.:]$/, '')}`);
}
/** Photos for a clicked location: the orbital view plus photos from missions that landed within ~300 km. */
async function showPlacePhotos(location) {
  drawOrbitalPhoto(+location.lat, +location.lon);
  const near = missions.filter(m => m.site && m.photo_query && greatCircleDeg(m.site, location) * 30.32 < 300);
  const region = cfg.sites.find(s => s.id === (location.siteId || location.parentSite));
  const queries = near.map(m => ({...m.photo_query, label: m.name}));
  if (region || +location.lat < -80) queries.push({q: 'lunar south pole', year_start: 2009, year_end: 2025, label: 'the lunar south pole'});
  const grid = $('photoGrid'), key = coordKey(location);
  grid.dataset.key = key;
  if (!queries.length) { grid.replaceChildren(); setText('photoNote', 'No NASA missions landed near here, so there are no mission photos.'); return; }
  setText('photoNote', 'Looking for photos…');
  const results = (await Promise.all(queries.map(nasaPhotos))).flat();
  if (grid.dataset.key !== key) return;
  const seen = new Set(), unique = results.filter(p => !seen.has(p.id) && seen.add(p.id));
  renderPhotoGrid(grid, $('photoNote'), unique, `Photos of ${queries.map(q => q.label).join(', ')}.`);
}
// ---------- Go to typed coordinates ----------
$('goButton').addEventListener('click', () => {
  try {
    const lat = parseCoordinate($('goLat').value, 'latitude'), lon = parseCoordinate($('goLon').value, 'longitude');
    if (lat === null || lon === null) throw new Error('Enter both a latitude and a longitude.');
    setText('goStatus', '');
    select({name: `Point at ${fmt.latlon(lat, lon)}`, lat, lon}, {focus: true});
  } catch (error) { setText('goStatus', error.message); }
});
for (const id of ['goLat', 'goLon']) $(id).addEventListener('keydown', event => { if (event.key === 'Enter') $('goButton').click(); });

// ---------- Sun & Earth summary (precomputed for the analysed regions) ----------
const TELEMETRY_SPEEDS = [1, 5, 20, 100, 1000, 10000, 50000, 100000];
const timelineFor = site => cfg.siteTimelines[site?.siteId || site?.parentSite] || null;
function showEnvironment() {
  const timeline = timelineFor(ui.selected);
  $('envSection').hidden = !timeline?.points?.length;
  if ($('envSection').hidden) return;
  const point = timeline.points[0];
  setText('sunElevation', `${(+point.sun_elevation_deg).toFixed(1)}° above horizon`.replace(/^-(.*) above/, '$1 below'));
  const hidden = elevation => elevation > 0 ? 'Hidden behind hills' : 'Below the horizon';
  setText('sunState', point.sun_visible ? 'Yes, sunlit' : `No: ${hidden(+point.sun_elevation_deg).toLowerCase()}`);
  setText('earthState', point.earth_visible ? 'Yes, can radio Earth' : `No: ${hidden(+point.earth_elevation_deg).toLowerCase()}`);
  setText('solarPower', `${Math.round(+point.power_w).toLocaleString()} W`);
  setText('envTotals', `☀ ${timeline.sun_visible_hours} h sun · ⊕ ${timeline.earth_visible_hours} h Earth`);
  setText('envTime', fmt.utc(point.timestamp));
  setText('envNote', `For the centre of this region at the scene time. ${point.ephemeris_source === 'skyfield' ? 'Sun and Earth positions from NASA JPL’s DE441 ephemeris.' : ''} ${cfg.horizonNote || ''}`);
}

// ---------- Uploaded CSVs ----------
function phaseRow(color, title, value, detail) {
  const row = element('div', 'phase'), bar = element('i'), b = element('b', '', title);
  bar.style.background = color; b.style.color = color;
  row.append(bar, b, element('em', '', value), element('span', '', detail));
  return row;
}
const LAT_NAMES = ['lat', 'lat_deg', 'latitude', 'lat_deg_n'], LON_NAMES = ['lon', 'lon_deg', 'longitude', 'long', 'lng'];
const SCORE_NAMES = ['composite_score', 'score'], LABEL_NAMES = ['name', 'label', 'site', 'id'];
/** A table of results with latitude/longitude columns (e.g. the classifier's *_candidates.csv). */
function parseResultTable(text, fileName) {
  const rows = parseCsvRecords(text);
  if (rows.length < 2) throw new Error('The file needs a header row and at least one row of data.');
  const header = rows.shift().map(h => h.trim().replace(/^﻿/, ''));
  const find = names => header.findIndex(h => names.includes(h.toLowerCase()));
  const latCol = find(LAT_NAMES), lonCol = find(LON_NAMES), scoreCol = find(SCORE_NAMES), labelCol = find(LABEL_NAMES), rankCol = find(['rank']);
  if (latCol < 0 || lonCol < 0) throw new Error('Not a trajectory (needs t, rx, ry, rz) and no latitude/longitude columns found (e.g. lat/lon or lat_deg/lon_deg).');
  const pretty = name => name.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());
  const parsed = rows.map((row, i) => {
    const lat = Number(row[latCol]), lon = Number(row[lonCol]);
    if (!Number.isFinite(lat) || !Number.isFinite(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180) throw new Error(`Row ${i + 2} has an invalid latitude/longitude.`);
    const fields = header.map((name, j) => ({name, j})).filter(({j}) => ![latCol, lonCol, scoreCol, labelCol, rankCol].includes(j) && (row[j] ?? '').trim() !== '')
      .map(({name, j}) => { const value = row[j].trim(), n = Number(value); return {label: pretty(name), value: Number.isFinite(n) && value.includes('.') ? String(+n.toPrecision(4)) : value}; });
    const score = scoreCol >= 0 ? Number(row[scoreCol]) : null;
    return {lat, lon, rank: rankCol >= 0 ? row[rankCol].trim() : i + 1, label: labelCol >= 0 ? row[labelCol].trim() : `Row ${i + 1}`,
      score: Number.isFinite(score) ? score : null, fields};
  });
  const id = `upload-${Date.now()}`;
  const centre = parsed.reduce((acc, r) => ({lat: acc.lat + r.lat / parsed.length, lon: acc.lon + r.lon / parsed.length}), {lat: 0, lon: 0});
  return {id, name: fileName, rows: parsed, summary: {name: fileName, ...(parsed[0] || centre), parentSite: id,
    fields: [{label: 'Rows', value: String(parsed.length)}, {label: 'Columns', value: header.join(', ')}]}};
}
function addResultSet(set) {
  set.markers = set.rows.map(row => {
    const marker = P.addMarker(P.moon, {...row, name: `${set.name} · ${row.label}`, resultSet: set.id}, 0x5fd4ff, 0.02);
    marker.visible = ui.layers.results;
    P.markers.results.push(marker);
    return marker;
  });
  ui.resultSets.push(set);
  select(set.summary, {focus: true});
}
function removeResultSet(id) {
  const set = ui.resultSets.find(r => r.id === id);
  if (!set) return;
  for (const marker of set.markers) { marker.parent?.remove(marker); marker.geometry.dispose(); marker.material.dispose(); }
  P.markers.results = P.markers.results.filter(m => !set.markers.includes(m));
  ui.resultSets = ui.resultSets.filter(r => r.id !== id);
  if ((ui.selected?.parentSite) === id) select({...cfg.sites[0], siteId: cfg.sites[0].id}, {open: false});
  else renderSites();
}
async function loadCsvText(text, name) { return loadCsvFile(new File([text], `${name}.csv`, {type: 'text/csv'}), name); }
async function loadCsvFile(file, displayName) {
  if (!file) return;
  try {
    const text = await file.text();
    const header = (text.split(/\r?\n/, 1)[0] || '').toLowerCase().replace(/^﻿/, '').split(',').map(h => h.trim());
    if (['t', 'rx', 'ry', 'rz'].every(name => header.includes(name))) {
      const data = parseTrajectory(text);
      const t0 = data.samples[0].t;
      data.speedX = data.samples.map(s => s.t - t0);
      data.speedY = data.samples.map(s => norm3(s.v));
      ui.telemetry = data; ui.csvName = displayName || file.name;
      T.load(data);
      setMode('telemetry');
      setText('csvStatus', `Showing ${file.name}: trajectory, ${data.samples.length.toLocaleString()} samples over ${fmt.seconds(data.samples[data.samples.length - 1].t - t0)}.`);
      openRight('flight');
      if (!$('graphsModal').hidden) renderGraphs();
    } else {
      const set = parseResultTable(text, displayName || file.name);
      addResultSet(set);
      setText('csvStatus', `Showing ${file.name}: ${set.rows.length} spots added to the Moon (blue dots) and the Landing spots list.`);
    }
  } catch (error) {
    setText('csvStatus', `Could not show ${file.name}: ${error.message}`);
  } finally {
    $('uploadCsv').value = '';
  }
}
$('uploadCsv').addEventListener('change', event => loadCsvFile(event.target.files?.[0]));
app.addEventListener('dragover', event => { event.preventDefault(); $('csvDrop').classList.add('dragging'); });
app.addEventListener('dragleave', event => { if (event.target === app || !app.contains(event.relatedTarget)) $('csvDrop').classList.remove('dragging'); });
app.addEventListener('drop', event => { event.preventDefault(); $('csvDrop').classList.remove('dragging'); loadCsvFile(event.dataTransfer?.files?.[0]); });

// ---------- Trajectory (telemetry) details ----------
function renderTelemetryDetails() {
  const data = ui.telemetry, samples = data.samples, first = samples[0], last = samples[samples.length - 1];
  const endLocal = vec(sub3(last.r, last.moon)).applyQuaternion(T.moonOrientation().invert()).normalize();
  const endLat = Math.asin(clamp(endLocal.y, -1, 1)) * 180 / Math.PI, endLon = Math.atan2(-endLocal.z, endLocal.x) * 180 / Math.PI;
  setText('telemetryLine', `${ui.csvName} · ${samples.length.toLocaleString()} samples · ends over ${fmt.latlon(endLat, endLon)}`);
  let path = 0, peak = 0, surfacePath = 0;
  const pathOf = (a, b) => norm3(sub3(b.r, a.r)); // ECI distance travelled
  for (let i = 1; i < samples.length; i++) {
    path += pathOf(samples[i - 1], samples[i]);
    surfacePath += norm3(sub3(sub3(samples[i].r, samples[i].moon), sub3(samples[i - 1].r, samples[i - 1].moon)));
  }
  for (const s of samples) peak = Math.max(peak, norm3(s.v));
  setText('tmDuration', fmt.seconds(last.t - first.t));
  setText('tmDistance', fmt.length(path));
  setText('tmMass', first.mass != null && last.mass != null ? `${Math.round(first.mass).toLocaleString()} → ${Math.round(last.mass).toLocaleString()} kg` : '—');
  setText('tmPeak', `${peak.toFixed(3)} km/s`);
  for (const b of $('focusButtons').children) b.classList.toggle('active', b.dataset.focus === T.state.focus);
  const stages = [];
  samples.forEach((s, i) => { if (!stages.length || stages[stages.length - 1].stage !== s.stage) stages.push({stage: s.stage, start: i, end: i}); else stages[stages.length - 1].end = i; });
  $('phaseList').replaceChildren(...stages.map(({stage, start, end}) => {
    let length = 0;
    for (let i = Math.max(1, start); i <= end; i++) length += pathOf(samples[i - 1], samples[i]);
    const a = samples[start], b = samples[end];
    return phaseRow(stageColor(stage), stageName(stage), fmt.seconds(b.t - a.t), `${fmt.length(length)}${a.mass != null ? ` · ${Math.round(a.mass).toLocaleString()} → ${Math.round(b.mass).toLocaleString()} kg` : ''}`);
  }));
  const notes = ['ECI frame at true scale: Earth’s centre is the fixed origin and all positions are the CSV’s own values.'];
  notes.push(data.hasMoon ? 'The Moon moves to the CSV moon_r position at every sample.' : 'No moon columns: the Moon is held at the scene date’s position.');
  if (T.state.nearSurface && T.state.longTrack) notes.push(`Near the Moon, a faint white line shows the final approach relative to the lunar surface, on a ground patch with a ${fmt.length(T.state.gridKm)} grid.`);
  else if (T.state.nearSurface) notes.push(`Faint white line: the same descent relative to the lunar surface (${fmt.length(surfacePath)}), on a ground patch with a ${fmt.length(T.state.gridKm)} grid.`);
  setText('routeNote', notes.join(' '));
  setText('speedNote', data.hasVelocity ? '|v| from the CSV velocity columns.' : 'Derived from positions (no velocity columns).');
}
function drawSpeedChart() {
  if (mode !== 'telemetry') return;
  const samples = ui.telemetry.samples, t0 = samples[0].t;
  drawChart($('speedChart'), {x: ui.telemetry.speedX, y: ui.telemetry.speedY}, {
    colorAt: i => stageColor(samples[i].stage), cursor: T.state.t - t0, xFormat: v => fmt.seconds(v), yFormat: v => v.toFixed(3)});
  setText('speedReadout', `${norm3(T.state.now?.v || [0, 0, 0]).toFixed(3)} km/s`);
}
function clearTelemetry() {
  ui.telemetry = null; ui.csvName = '';
  setMode('scene');
}
$('clearTrajectory').addEventListener('click', () => { clearTelemetry(); if (ui.selected) select(ui.selected, {open: false}); });
$('focusButtons').addEventListener('click', event => { const focus = event.target.dataset.focus; if (focus) { T.setFocus(focus); renderTelemetryDetails(); } });

// ---------- Graphs dialog ----------
function graphSeries() {
  const series = (sim.graphs || []).map(graph => ({
    title: graph.title, unit: graph.unit, x: graph.points.map(p => Date.parse(p.time)), y: graph.points.map(p => +p.value),
    xFormat: v => new Date(v).toISOString().slice(5, 10),
  }));
  if (ui.telemetry) {
    const s = ui.telemetry.samples, t0 = s[0].t, x = s.map(p => p.t - t0), add = (title, unit, values) => {
      if (values.every(Number.isFinite)) series.push({title, unit, x, y: values, xFormat: v => fmt.seconds(v)});
    };
    add('Inertial speed', 'km/s', s.map(p => norm3(p.v)));
    if (ui.telemetry.hasMoon) add('Height above mean lunar radius', 'km', s.map(p => norm3(sub3(p.r, p.moon)) - MOON_KM));
    add('Distance from Earth centre', 'km', s.map(p => norm3(p.r)));
    add('Altitude above Earth (CSV)', 'km', s.map(p => p.altitude == null ? NaN : p.altitude / 1000));
    add('Acceleration magnitude', 'm/s²', s.map(p => p.accel ? norm3(p.accel) : NaN));
    add('Mass flow rate', 'kg/s', s.map(p => p.mdot ?? NaN));
    add('Vehicle mass', 'kg', s.map(p => p.mass ?? NaN));
    add('Thrust', 'N', s.map(p => p.thrust ?? NaN));
    add('Target distance', 'km', s.map(p => p.targetDistance == null ? NaN : p.targetDistance / 1000));
  }
  return series;
}
function renderGraphs() {
  setText('graphsMeta', `${sim.source} · epoch ${fmt.utc(sim.epoch_utc)}`);
  setText('rawData', JSON.stringify({...sim, graphs: `${(sim.graphs || []).length} series`}, null, 2));
  const list = $('graphList');
  list.replaceChildren();
  const series = graphSeries();
  if (!series.length) { list.textContent = 'No graph series available.'; return; }
  series.forEach((s, i) => {
    const title = element('div', 'graph-title', `${s.title} (${s.unit})`), canvas = element('canvas', 'chart');
    canvas.style.height = '130px';
    list.append(title, canvas);
    drawChart(canvas, s, {color: PHASE_COLORS[(i + 1) % PHASE_COLORS.length], xFormat: s.xFormat});
  });
}

// ---------- Bottom bar (trajectory playback only) ----------
function configureBar() {
  $('bottomBar').hidden = mode !== 'telemetry';
  app.classList.toggle('bar-open', mode === 'telemetry');
  if (mode !== 'telemetry') return;
  $('barSpeed').replaceChildren(...TELEMETRY_SPEEDS.map(r => new Option(`${r.toLocaleString()}×`, String(r), false, r === T.state.rate)));
  syncBarButton();
}
function syncBarButton() {
  $('barPlay').textContent = T.state.playing ? '❚❚' : '▶';
  $('barPlay').setAttribute('aria-label', T.state.playing ? 'Pause' : 'Play');
}
function stepBar() {
  if (mode !== 'telemetry') return;
  const times = T.state.times;
  if (!times.length) return;
  const t0 = times[0], t1 = times[times.length - 1];
  if (document.activeElement !== $('barRange')) $('barRange').value = String(Math.round((T.state.t - t0) / Math.max(1e-9, t1 - t0) * 1000));
  const now = T.state.now;
  setText('barTime', ui.telemetry?.hasUtc && now?.utc ? `${new Date(now.utc).toISOString().slice(0, 16).replace('T', ' ')} UTC` : `${fmt.seconds(T.state.t - t0)} / ${fmt.seconds(t1 - t0)}`);
  syncBarButton();
}
$('barPlay').addEventListener('click', () => {
  const times = T.state.times;
  if (!T.state.playing && T.state.t >= times[times.length - 1]) T.seek(times[0]);
  T.state.playing = !T.state.playing;
  syncBarButton();
});
$('barRange').addEventListener('input', event => {
  const times = T.state.times;
  T.seek(times[0] + Number(event.target.value) / 1000 * (times[times.length - 1] - times[0]));
});
$('barSpeed').addEventListener('change', event => { T.state.rate = Number(event.target.value); });

// ---------- Mode switching (Moon view <-> trajectory view) ----------
function setMode(next) {
  mode = next;
  const scene = mode === 'scene';
  P.controls.enabled = scene;
  if (T.controls) T.controls.enabled = !scene;
  $('viewToggle').hidden = !scene;
  $('clearTrajectory').hidden = scene;
  $('flightTab').hidden = scene;
  if (scene && !document.querySelector('[data-tab-panel="flight"]').hidden) openRight('site');
  configureBar();
  if (scene) showEnvironment(); else renderTelemetryDetails();
  updateHeader();
}

// ---------- Panels, tabs, top bar ----------
const panelTitles = {sites: 'Landing spots', upload: 'Upload a CSV', missions: 'Moon missions', layers: 'What’s shown', data: 'About the data'};
let leftView = 'sites';
function showLeft(view) {
  const panel = $('leftPanel'), closing = view === leftView && !panel.classList.contains('closed');
  panel.classList.toggle('closed', closing);
  app.classList.toggle('left-open', !closing);
  leftView = view;
  for (const button of document.querySelectorAll('.rail [data-view]')) button.classList.toggle('active', !closing && button.dataset.view === view);
  for (const section of document.querySelectorAll('[data-panel]')) section.hidden = section.dataset.panel !== view;
  setText('leftTitle', panelTitles[view]);
  if (window.innerWidth < 720 && !closing) closeRight();
}
for (const button of document.querySelectorAll('.rail [data-view]')) button.addEventListener('click', () => showLeft(button.dataset.view));
$('leftClose').addEventListener('click', () => { $('leftPanel').classList.add('closed'); app.classList.remove('left-open'); for (const b of document.querySelectorAll('.rail [data-view]')) b.classList.remove('active'); });
function openRight(tab) {
  $('rightPanel').classList.remove('closed'); app.classList.remove('right-closed'); $('detailsToggle').setAttribute('aria-pressed', 'true');
  if (tab) for (const button of document.querySelectorAll('[data-tab]')) if (button.dataset.tab === tab) button.click();
  if (window.innerWidth < 720) { $('leftPanel').classList.add('closed'); app.classList.remove('left-open'); }
}
function closeRight() { $('rightPanel').classList.add('closed'); app.classList.add('right-closed'); $('detailsToggle').setAttribute('aria-pressed', 'false'); }
$('rightClose').addEventListener('click', closeRight);
$('detailsToggle').addEventListener('click', () => $('rightPanel').classList.contains('closed') ? openRight() : closeRight());
for (const button of document.querySelectorAll('[data-tab]')) button.addEventListener('click', () => {
  for (const b of document.querySelectorAll('[data-tab]')) b.classList.toggle('active', b === button);
  for (const section of document.querySelectorAll('[data-tab-panel]')) section.hidden = section.dataset.tabPanel !== button.dataset.tab;
});
const openGraphs = () => { $('graphsModal').hidden = false; renderGraphs(); };
$('graphsToggle').addEventListener('click', openGraphs);
$('openGraphs').addEventListener('click', openGraphs);
for (const button of document.querySelectorAll('[data-close]')) button.addEventListener('click', () => { $(button.dataset.close).hidden = true; });
for (const modal of document.querySelectorAll('.modal')) modal.addEventListener('click', event => { if (event.target === modal) modal.hidden = true; });
document.addEventListener('keydown', event => { if (event.key === 'Escape') for (const modal of document.querySelectorAll('.modal')) modal.hidden = true; });
$('viewToggle').addEventListener('click', () => {
  P.setOverhead(!P.view.overhead);
  $('viewToggle').textContent = P.view.overhead ? '3D' : '2D';
  $('viewToggle').setAttribute('aria-pressed', String(P.view.overhead));
});
$('fullscreenToggle').addEventListener('click', () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else app.requestFullscreen?.().catch(() => {});
});
$('homeView').addEventListener('click', () => activeView().home());
function zoom(factor) {
  const view = activeView(), controls = view.controls, camera = view.camera;
  const offset = camera.position.clone().sub(controls.target);
  const limits = mode === 'telemetry' ? [0.003, 3e6] : [0.6, 900];
  camera.position.copy(controls.target).add(offset.setLength(clamp(offset.length() * factor, ...limits)));
  controls.update();
}
$('zoomIn').addEventListener('click', () => zoom(0.78));
$('zoomOut').addEventListener('click', () => zoom(1.28));
renderer.domElement.addEventListener('wheel', event => { event.preventDefault(); zoom(Math.exp(event.deltaY * 0.0012)); }, {passive: false});
renderer.domElement.addEventListener('contextmenu', event => event.preventDefault());
for (const input of document.querySelectorAll('[data-layer]')) input.addEventListener('change', () => {
  const layer = input.dataset.layer, on = input.checked;
  ui.layers[layer] = on;
  if (layer === 'candidates') for (const marker of P.markers.candidates.values()) marker.visible = on;
  if (layer === 'historic') for (const marker of P.markers.historic) marker.visible = on;
  if (layer === 'results') for (const marker of P.markers.results) marker.visible = on;
  if (layer === 'labels') labels.enabled = on;
  if (layer === 'sun') P.sunMesh.visible = P.sunGlow.visible = on;
});

// ---------- Hover & click on the Moon ----------
const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2(), tooltip = $('tooltip');
function surfaceCoordinates(mesh, point) {
  const local = mesh.worldToLocal(point.clone()).normalize();
  return {lat: Math.asin(clamp(local.y, -1, 1)) * 180 / Math.PI, lon: Math.atan2(-local.z, local.x) * 180 / Math.PI};
}
function pick(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
  raycaster.setFromCamera(pointer, P.camera);
  const hit = raycaster.intersectObjects(pickable(), false)[0];
  if (!hit) return null;
  const known = hit.object.userData.location;
  const body = known ? hit.object.userData.body : hit.object === P.earth.surface ? 'earth' : 'moon';
  const coords = known ? {lat: +known.lat, lon: +known.lon} : surfaceCoordinates(hit.object, hit.point);
  return {known, body, coords, rect};
}
function hover(event) {
  const result = mode === 'scene' ? pick(event) : null;
  if (!result) { tooltip.style.display = 'none'; return; }
  const {known, body, coords, rect} = result;
  const mission = known?.missionId && missions.find(m => m.id === known.missionId);
  tooltip.replaceChildren(element('div', '', known ? known.name : body === 'earth' ? 'Earth' : 'Moon'),
    element('small', '', mission ? `${mission.status === 'planned' ? `Planned: ${mission.target}` : mission.launch.slice(0, 4)} · ${mission.type} · click for details`
      : `${fmt.latlon(coords.lat, coords.lon)}${body === 'moon' ? ' · click to show this point' : ''}`));
  tooltip.style.display = 'block';
  tooltip.style.left = `${Math.min(event.clientX - rect.left + 14, rect.width - 290)}px`;
  tooltip.style.top = `${event.clientY - rect.top + 14}px`;
}
function click(event) {
  const result = mode === 'scene' ? pick(event) : null;
  if (!result || result.body === 'earth') return;
  const {known, coords} = result;
  if (known?.missionId) { showMission(known.missionId); return; }
  if (known?.siteId) { select(cfg.sites.find(s => s.id === known.siteId) && {...cfg.sites.find(s => s.id === known.siteId), siteId: known.siteId}); return; }
  if (known?.resultSet) {
    select({name: known.name, lat: known.lat, lon: known.lon, parentSite: known.resultSet,
      fields: [...(known.score != null ? [{label: 'Score', value: (+known.score).toFixed(2)}] : []), ...(known.fields || [])]});
    return;
  }
  select({name: `Point at ${fmt.latlon(coords.lat, coords.lon)}`, lat: coords.lat, lon: coords.lon});
}
let pressed = null;
renderer.domElement.addEventListener('pointerdown', event => { if (event.button === 0) pressed = {x: event.clientX, y: event.clientY}; });
renderer.domElement.addEventListener('pointerup', event => { if (pressed && Math.hypot(event.clientX - pressed.x, event.clientY - pressed.y) < 6) click(event); pressed = null; });
renderer.domElement.addEventListener('pointermove', event => { if (!pressed) hover(event); });
renderer.domElement.addEventListener('pointerleave', () => { tooltip.style.display = 'none'; });

/* =====================================================================
 * Moon missions: list, statistics and details
 * ===================================================================== */
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const niceDate = iso => { if (!iso) return null; const [y, m, d] = iso.split('-').map(Number); return `${MONTHS[m - 1]} ${d}, ${y}`; };
const STATUS_TEXT = {success: 'Success', partial: 'Partly successful', failed: 'Failed', planned: 'Planned'};

function renderMissionStats() {
  const past = missions.filter(m => m.status !== 'planned');
  const walkers = past.flatMap(m => m.crew || []).filter(c => /walked on the Moon/i.test(c.role)).length;
  const kg = past.reduce((sum, m) => {
    const stat = (m.stats || []).find(s => /brought home|returned/i.test(s.label));
    if (!stat) return sum;
    const value = parseFloat(stat.value.replace(/[^\d.]/g, ''));
    return sum + (Number.isFinite(value) ? (/\bg\b/.test(stat.value) && !/kg/.test(stat.value) ? value / 1000 : value) : 0);
  }, 0);
  const tiles = [[past.length, 'missions so far'], [walkers, 'people have walked on the Moon'], [`${Math.round(kg)} kg`, 'of Moon rock brought to Earth'],
    [missions.length - past.length, 'NASA missions planned next']];
  $('missionStats').replaceChildren(...tiles.map(([value, label]) => { const tile = element('div'); tile.append(element('strong', '', String(value)), element('span', '', label)); return tile; }));
  setText('missionNote', cfg.missions?.note || '');
}
function renderMissionList() {
  const list = $('missionList'), filter = ui.missionFilter;
  const shown = missions.filter(m => filter === 'all' || (filter === 'past' ? m.status !== 'planned' : filter === 'planned' ? m.status === 'planned' : m.crewed));
  list.replaceChildren();
  let era = null;
  for (const mission of shown) {
    const group = mission.status === 'planned' ? 'Coming up (NASA)' : `${mission.launch.slice(0, 3)}0s`;
    if (group !== era) { list.appendChild(element('div', 'era', group)); era = group; }
    const card = element('button', `mission-card${ui.mission === mission.id ? ' selected' : ''}`);
    card.type = 'button';
    const top = element('div', 'top');
    top.append(element('strong', '', mission.name), element('time', '', mission.status === 'planned' ? mission.target : niceDate(mission.launch)));
    card.append(top, element('p', '', mission.summary), element('span', `status ${mission.status}`, STATUS_TEXT[mission.status]), element('span', 'status', mission.crewed ? `Crew of ${mission.crew?.length || '?'}` : 'Robotic'));
    card.addEventListener('click', () => showMission(mission.id));
    list.appendChild(card);
  }
  if (!shown.length) list.appendChild(element('p', 'foot', 'No missions match this filter.'));
}
function showMission(id) {
  const mission = missions.find(m => m.id === id);
  if (!mission) return;
  ui.mission = id;
  const box = $('missionDetail');
  box.replaceChildren();
  box.className = 'mission-detail';
  box.append(element('h3', '', mission.name), element('div', 'sub', `${mission.agency} · ${mission.type}`));
  const badges = element('div');
  badges.append(element('span', `status ${mission.status}`, STATUS_TEXT[mission.status]), element('span', 'status', mission.crewed ? 'With astronauts' : 'No crew (robotic)'));
  box.append(badges, element('p', 'summary', mission.summary), element('p', 'desc', mission.description));

  box.appendChild(element('div', 'section', 'When'));
  const rows = element('div', 'rows');
  const row = (label, value) => { if (!value) return; const r = element('div', 'row'); r.append(element('span', '', label), element('strong', '', value)); rows.appendChild(r); };
  if (mission.status === 'planned') { row('Target date', mission.target); row('Length', mission.duration); }
  else {
    row('Launched', niceDate(mission.launch));
    row(mission.type.includes('mpact') ? 'Hit the Moon' : 'Landed on the Moon', niceDate(mission.landing));
    row(mission.crewed ? 'Back on Earth' : 'Mission ended', niceDate(mission.end));
    row('How long', mission.duration);
  }
  if (mission.site) row('Where', `${mission.site.name} (${mission.site.approximate ? '≈ ' : ''}${fmt.latlon(mission.site.lat, mission.site.lon)})`);
  box.appendChild(rows);

  if (mission.crewed) {
    box.appendChild(element('div', 'section', 'Astronauts on board'));
    if (mission.crew?.length) {
      const crew = element('div', 'crew');
      for (const person of mission.crew) {
        const item = element('div'), text = element('div');
        const initials = person.name.split(/\s+/).map(part => part[0]).slice(0, 2).join('');
        text.append(element('b', '', person.name), element('small', '', person.role));
        item.append(element('span', 'avatar', initials), text);
        crew.appendChild(item);
      }
      box.appendChild(crew);
    } else box.appendChild(element('p', 'foot', 'The crew has not been announced yet.'));
  }
  if (mission.stats?.length) {
    box.appendChild(element('div', 'section', 'By the numbers'));
    const grid = element('div', 'metrics two');
    for (const stat of mission.stats) {
      const tile = element('div', 'metric');
      tile.append(element('span', '', stat.label), element('strong', 'text', stat.value));
      grid.appendChild(tile);
    }
    box.appendChild(grid);
  }
  const buttons = element('div', 'button-row');
  if (mission.site) {
    const look = element('button', 'btn ghost', 'Show on the Moon');
    look.addEventListener('click', () => { if (mode === 'telemetry') clearTelemetry(); P.focusMoonPoint(mission.site.lat, mission.site.lon); });
    buttons.append(look);
  }
  if (mission.status !== 'planned') {
    // Flight paths are CSV files you supply: data/mission_paths/<mission id>.csv, or uploaded here.
    if ((cfg.missionPaths || []).includes(mission.id) && api.base) {
      const watch = element('button', 'btn', 'Show the flight path');
      watch.addEventListener('click', async () => {
        try {
          const response = await fetch(api.url(`/api/mission-path/${mission.id}`));
          if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || `server error ${response.status}`);
          await loadCsvText(await response.text(), `${mission.name} flight path`);
        } catch (error) { setText('csvStatus', `Could not load the ${mission.name} path: ${error.message}`); showLeft('upload'); }
      });
      buttons.appendChild(watch);
    }
    const upload = element('label', 'btn ghost path-load', 'Upload a path CSV…');
    const input = element('input');
    input.type = 'file'; input.accept = '.csv,text/csv';
    input.addEventListener('change', () => { const file = input.files?.[0]; if (file) loadCsvFile(file, `${mission.name}: ${file.name}`); input.value = ''; });
    upload.appendChild(input);
    buttons.appendChild(upload);
  }
  box.appendChild(buttons);
  if (mission.status === 'planned') box.appendChild(element('p', 'foot', 'Planned dates often change. Check nasa.gov/artemis for the latest.'));
  if (mission.sources?.length) {
    const sources = element('div', 'sources');
    sources.appendChild(element('div', 'section', 'Sources'));
    for (const url of mission.sources) { const link = element('a', '', url.replace(/^https?:\/\/(www\.)?/, '')); link.href = url; link.target = '_blank'; link.rel = 'noopener'; sources.appendChild(link); }
    box.appendChild(sources);
  }
  if (mission.photo_query) {
    box.appendChild(element('div', 'section', 'Photos'));
    const grid = element('div', 'photos'), note = element('p', 'foot', 'Looking for photos…');
    note.id = 'missionPhotoNote';
    box.append(grid, note);
    nasaPhotos(mission.photo_query).then(photos => { if (ui.mission === mission.id) renderPhotoGrid(grid, note, photos, `${mission.name}: rocket, crew and mission photos.`); });
  }
  $('missionTab').hidden = false;
  openRight('missionInfo');
  renderMissionList();
  if (mission.site && mode === 'scene') P.focusMoonPoint(mission.site.lat, mission.site.lon);
}
$('missionFilters').addEventListener('click', event => {
  const filter = event.target.dataset.filter;
  if (!filter) return;
  ui.missionFilter = filter;
  for (const button of $('missionFilters').children) button.classList.toggle('active', button.dataset.filter === filter);
  renderMissionList();
});

/* =====================================================================
 * Help bubbles, welcome guide and "go to" shortcuts
 * ===================================================================== */
const bubble = $('helpBubble');
function showHelp(target) {
  const text = GLOSSARY[target.dataset.term];
  if (!text) return;
  bubble.textContent = text; bubble.hidden = false;
  const rect = target.getBoundingClientRect(), box = bubble.getBoundingClientRect();
  bubble.style.left = `${clamp(rect.left + rect.width / 2 - box.width / 2, 8, window.innerWidth - box.width - 8)}px`;
  bubble.style.top = `${rect.bottom + 8 + box.height > window.innerHeight ? rect.top - box.height - 8 : rect.bottom + 8}px`;
}
document.addEventListener('mouseover', event => { const help = event.target.closest?.('.help'); if (help) showHelp(help); });
document.addEventListener('mouseout', event => { if (event.target.closest?.('.help')) bubble.hidden = true; });
document.addEventListener('focusin', event => { const help = event.target.closest?.('.help'); if (help) showHelp(help); });
document.addEventListener('focusout', event => { if (event.target.closest?.('.help')) bubble.hidden = true; });
document.addEventListener('click', event => {
  const help = event.target.closest?.('.help');
  if (help) { event.preventDefault(); event.stopPropagation(); showHelp(help); return; }
  const goto = event.target.closest?.('[data-goto]');
  if (!goto) return;
  $('welcome').hidden = true;
  const target = goto.dataset.goto;
  if (leftView !== target || $('leftPanel').classList.contains('closed')) showLeft(target);
}, true);
for (const icon of document.querySelectorAll('.help')) icon.tabIndex = 0;
$('helpToggle').addEventListener('click', () => { $('welcome').hidden = false; });
function rememberWelcome() { try { localStorage.setItem('lunai-welcomed', '1'); } catch { /* storage may be blocked */ } }
for (const button of document.querySelectorAll('#welcome [data-close], #welcome [data-goto]')) button.addEventListener('click', rememberWelcome);

/* =====================================================================
 * Start-up & render loop
 * ===================================================================== */
renderMissionStats();
renderMissionList();
try { if (!localStorage.getItem('lunai-welcomed')) $('welcome').hidden = false; } catch { $('welcome').hidden = false; }
setText('classifierName', cfg.classifier?.name || '—');
setText('terrainSource', cfg.dataSource);
setText('ephemerisSource', sim.source);
setText('epochValue', fmt.utc(sim.epoch_utc));
setText('epochChip', fmt.utc(sim.epoch_utc));

function resize() {
  const width = app.clientWidth, height = app.clientHeight;
  renderer.setSize(width, height, false);
  for (const camera of [P.camera, T.camera]) { camera.aspect = width / height; camera.updateProjectionMatrix(); }
}
new ResizeObserver(resize).observe(app);
resize();
P.home();
if (window.innerWidth < 720) $('leftClose').click(); // one panel at a time on phones
setMode('scene');
select({...cfg.sites[0], siteId: cfg.sites[0].id}, {open: true});

let lastFrame = performance.now(), chartFrame = 0;
function frame(now) {
  const dt = Math.min(0.1, (now - lastFrame) / 1000);
  lastFrame = now;
  const view = activeView();
  view.update(dt);
  stepBar();
  view.controls.update();
  renderer.render(view.scene, view.camera);
  labels.render(mode === 'scene' ? P.labelItems() : T.labelItems(), view.camera);
  if (mode === 'telemetry' && !$('rightPanel').classList.contains('closed') && !document.querySelector('[data-tab-panel="flight"]').hidden && (chartFrame++ % 2 === 0)) drawSpeedChart();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// Exposed for automated checks and quick experiments from the browser console.
window.lunai = {cfg, P, T, ui, loadCsvText: (text, name = 'data.csv') => loadCsvFile(new File([text], name, {type: 'text/csv'}))};
})();
