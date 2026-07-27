// LeLamp renderer. Receives LampCommand messages over a WebSocket and eases
// the lamp toward the commanded pose/light. It knows nothing about why.

import * as THREE from 'three';

// ---------- scene ----------

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101014);

const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 100);
// Camera sits on the -X side: the arm's pitch plane bends toward -X, so this
// is a three-quarter FRONT view — the lamp leans/faces toward the viewer.
camera.position.set(-6.0, 4.6, 10.0);
camera.lookAt(0, 2.4, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

scene.add(new THREE.HemisphereLight(0x8890a8, 0x202028, 1.6));
const fill = new THREE.DirectionalLight(0xffffff, 1.2);
fill.position.set(5, 8, 4);
scene.add(fill);

const ground = new THREE.Mesh(
  new THREE.CircleGeometry(7, 48),
  new THREE.MeshStandardMaterial({ color: 0x1c1c22, roughness: 0.9 })
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// ---------- lamp ----------
// Hierarchy: base -> yaw -> shoulder(pitch) -> arm1 -> elbow(pitch) -> arm2
//            -> wrist(pitch) -> arm3 -> headYaw -> headPitch -> shade/bulb/spot.
// Every arm segment extends along its group's local +Y; pitches rotate about Z.

const metal = new THREE.MeshStandardMaterial({ color: 0xd8d8dc, metalness: 0.6, roughness: 0.35 });
const L1 = 1.6, L2 = 1.4, L3 = 0.9;

function segment(length) {
  const g = new THREE.Group();
  const joint = new THREE.Mesh(new THREE.SphereGeometry(0.13, 20, 20), metal);
  const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, length, 16), metal);
  rod.position.y = length / 2;
  joint.castShadow = rod.castShadow = true;
  g.add(joint, rod);
  return g;
}

const base = new THREE.Mesh(new THREE.CylinderGeometry(0.75, 0.95, 0.25, 32), metal);
base.position.y = 0.125;
base.castShadow = true;
scene.add(base);

const yaw = new THREE.Group();
yaw.position.y = 0.25;
scene.add(yaw);

const shoulder = segment(L1);
yaw.add(shoulder);

const elbow = segment(L2);
elbow.position.y = L1;
shoulder.add(elbow);

const wrist = segment(L3);
wrist.position.y = L2;
elbow.add(wrist);

const headYaw = new THREE.Group();
headYaw.position.y = L3;
wrist.add(headYaw);

const headPitch = new THREE.Group();
headYaw.add(headPitch);

// Shade: cone with apex at the wrist end, opening facing along local +Y.
const shadeMat = new THREE.MeshStandardMaterial({
  color: 0xd8d8dc, metalness: 0.4, roughness: 0.4,
  emissive: 0xfff2cc, emissiveIntensity: 0.0, side: THREE.DoubleSide,
});
const shade = new THREE.Mesh(new THREE.ConeGeometry(0.42, 0.75, 32, 1, true), shadeMat);
shade.rotation.x = Math.PI;
shade.position.y = 0.375;
shade.castShadow = true;
headPitch.add(shade);

const bulbMat = new THREE.MeshStandardMaterial({
  color: 0xfff8e0, emissive: 0xfff2cc, emissiveIntensity: 0.0,
});
const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.12, 20, 20), bulbMat);
bulb.position.y = 0.45;
headPitch.add(bulb);

const spot = new THREE.SpotLight(0xfff2cc, 0, 20, 0.55, 0.45, 1.0);
spot.position.y = 0.45;
spot.castShadow = true;
headPitch.add(spot);
const spotTarget = new THREE.Object3D();
spotTarget.position.y = 5;
headPitch.add(spotTarget);
spot.target = spotTarget;

// ---------- pose state + easing ----------

const REST = {
  base_yaw: 1.0, shoulder_pitch: 0.55, elbow_pitch: -1.1,
  wrist_pitch: 0.55, head_yaw: 0.0, head_pitch: 1.15,
};
const current = { ...REST };
const target = { ...REST };

const light = { intensity: 0.0, color: new THREE.Color(0xfff2cc) };
const lightTarget = { intensity: 0.4, color: new THREE.Color(0xfff2cc) };

function applyPose() {
  yaw.rotation.y = current.base_yaw;
  shoulder.rotation.z = current.shoulder_pitch;
  elbow.rotation.z = current.elbow_pitch;
  wrist.rotation.z = current.wrist_pitch;
  headYaw.rotation.y = current.head_yaw;
  headPitch.rotation.z = current.head_pitch;

  spot.intensity = light.intensity * 80;
  bulbMat.emissiveIntensity = light.intensity * 2.5;
  shadeMat.emissiveIntensity = light.intensity * 0.6;
  spot.color.copy(light.color);
  bulbMat.emissive.copy(light.color);
  shadeMat.emissive.copy(light.color);
}

// Critically damped spring per joint: stiff head, soft base. Response speed
// scales with how fast the target moves, so quick user motion reads as a
// quick head-flick while the body settles in behind it.
const STIFFNESS = {
  base_yaw: 30, shoulder_pitch: 40, elbow_pitch: 40,
  wrist_pitch: 50, head_yaw: 130, head_pitch: 130,
};
const velocity = {};
for (const k in REST) velocity[k] = 0;

const EASE = 4; // damping rate for light (color/intensity)
const clock = new THREE.Clock();
const headWorld = new THREE.Vector3();
const headScreen = { x: 0.5, y: 0.5 };
let lastCursor = null; // {x, y, over_lamp}
let lastSent = 0;

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  for (const k in current) {
    const stiffness = STIFFNESS[k];
    const accel = stiffness * (target[k] - current[k]) - 2 * Math.sqrt(stiffness) * velocity[k];
    velocity[k] += accel * dt;          // semi-implicit Euler: stable at our dt
    current[k] += velocity[k] * dt;
  }
  light.intensity = THREE.MathUtils.damp(light.intensity, lightTarget.intensity, EASE, dt);
  light.color.lerp(lightTarget.color, 1 - Math.exp(-EASE * dt));
  applyPose();

  // Head position in normalized screen coords, for hover detection + QA.
  headPitch.getWorldPosition(headWorld).project(camera);
  headScreen.x = (headWorld.x + 1) / 2;
  headScreen.y = (1 - headWorld.y) / 2;
  window.__lamp = { head: { ...headScreen }, cursor: lastCursor };

  renderer.render(scene, camera);
}
animate();

// ---------- cursor reporting (M1 attention scaffold) ----------
// Hovering on/near the lamp head = "attending". The page only reports; all
// engagement logic lives in Python.

function sendCursor() {
  if (ws && ws.readyState === WebSocket.OPEN && lastCursor) {
    ws.send(JSON.stringify({ type: 'cursor', ...lastCursor }));
  }
}

addEventListener('mousemove', (e) => {
  const x = e.clientX / innerWidth;
  const y = e.clientY / innerHeight;
  const over = Math.hypot(x - headScreen.x, y - headScreen.y) < 0.16;
  lastCursor = { x, y, over_lamp: over };
  const now = performance.now();
  if (now - lastSent > 33) {
    lastSent = now;
    sendCursor();
  }
});

document.addEventListener('mouseleave', () => { lastCursor = null; });

// Heartbeat so a resting (but present) cursor keeps signaling presence.
setInterval(sendCursor, 100);

// ---------- sound cues ----------

let audio = null;

function blip(t0, from, to) {
  const o = audio.createOscillator();
  const g = audio.createGain();
  o.frequency.setValueAtTime(from, t0);
  o.frequency.exponentialRampToValueAtTime(to, t0 + 0.1);
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.exponentialRampToValueAtTime(0.15, t0 + 0.02);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.12);
  o.connect(g).connect(audio.destination);
  o.start(t0);
  o.stop(t0 + 0.15);
}

// ---------- chat (M4 v1: text in, text out) ----------

const chatlog = document.getElementById('chatlog');
const chatbox = document.getElementById('chatbox');
let pendingLine = null;

function addLine(who, text) {
  const div = document.createElement('div');
  div.className = who;
  div.textContent = (who === 'you' ? 'you: ' : 'lamp: ') + text;
  chatlog.appendChild(div);
  chatlog.scrollTop = chatlog.scrollHeight;
  return div;
}

function chatReply(text) {
  if (pendingLine) {
    pendingLine.textContent = 'lamp: ' + text;
    pendingLine = null;
  } else {
    addLine('lamp', text);
  }
  chatlog.scrollTop = chatlog.scrollHeight;
}

function showTranscript(text) {
  addLine('you', text + ' 🎤');
  pendingLine = addLine('lamp', '…');
}

// ---------- push-to-talk (M4 v2) ----------

const micBtn = document.getElementById('micbtn');
let recorder = null;

async function startPTT() {
  if (recorder) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      recorder = null;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ptt', state: 'stop' }));
        const buf = await new Blob(chunks).arrayBuffer();
        if (buf.byteLength > 2000) ws.send(buf); // ignore accidental taps
      }
    };
    recorder.start();
    micBtn.classList.add('recording');
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'ptt', state: 'start' }));
    }
  } catch (err) {
    addLine('lamp', '(mic unavailable: ' + err.message + ')');
  }
}

function stopPTT() {
  micBtn.classList.remove('recording');
  if (recorder && recorder.state === 'recording') recorder.stop();
}

micBtn.addEventListener('mousedown', startPTT);
micBtn.addEventListener('mouseup', stopPTT);
micBtn.addEventListener('mouseleave', stopPTT);

chatbox.addEventListener('keydown', (e) => {
  e.stopPropagation();
  if (e.key !== 'Enter') return;
  const text = chatbox.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  addLine('you', text);
  pendingLine = addLine('lamp', '…');
  ws.send(JSON.stringify({ type: 'chat', text }));
  chatbox.value = '';
});

function cue(name) {
  try {
    audio ??= new AudioContext();
    if (audio.state === 'suspended') audio.resume(); // needs a user gesture first
    if (name === 'chirp') {
      blip(audio.currentTime, 700, 1300);
      blip(audio.currentTime + 0.15, 900, 1600);
    }
  } catch { /* audio unavailable; stay silent */ }
}

// ---------- websocket ----------

const status = document.getElementById('status');
let ws = null;

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { status.textContent = 'connected'; };
  ws.onclose = () => {
    status.textContent = 'disconnected — retrying…';
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'chat_reply') {
      chatReply(msg.text);
      return;
    }
    if (msg.type === 'transcript') {
      showTranscript(msg.text);
      return;
    }
    if (msg.type !== 'command') return;
    if (msg.joints) Object.assign(target, msg.joints);
    if (msg.light) {
      if (msg.light.intensity !== undefined) lightTarget.intensity = msg.light.intensity;
      if (msg.light.color !== undefined) lightTarget.color.set(msg.light.color);
    }
    if (msg.sound) cue(msg.sound);
  };
}
connect();
