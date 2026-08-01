// 1. Backend e Configuração
tf.setBackend('cpu');
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// Configuração de Tempo Limite
const CLONE_DURATION = 10000; // 10 segundos em milissegundos
let clonesTriggered = false;
let cloneStartTime = null;

let mask = null;
let gestureModel = null;
let rasenganRotation = 0;

// Imagem do Rasengan
const imgRasengan = new Image();
imgRasengan.src = 'assets/rasengan.png';

// 2. Carregamento do Modelo Multiclasse
async function loadGestureModel() {
  try {
    gestureModel = await tf.loadLayersModel("gesture-model.json");
    console.log("🔥 Modelo Multiclasse V2 carregado!");
  } catch (e) { console.error("❌ Erro ao carregar modelo:", e); }
}
loadGestureModel();

// 3. Inicialização do MediaPipe (Holistic & Selfie)
const selfie = new SelfieSegmentation({ locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${f}` });
selfie.setOptions({ modelSelection: 1 });
selfie.onResults((r) => (mask = r.segmentationMask));

const holistic = new Holistic({ locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}` });
// Reduzindo complexidade para melhorar FPS na CPU enquanto usa resolução alta
holistic.setOptions({ modelComplexity: 0, smoothLandmarks: true });

const camera = new Camera(video, { 
  width: 1920, 
  height: 1080, 
  onFrame: async () => {
    await selfie.send({ image: video });
    await holistic.send({ image: video });
  }
});
camera.start();

// 4. Funções Auxiliares de Processamento
function getEmptyHand() { return new Array(63).fill(0); }

function normalizeHand(lm) {
  const w = lm[0], mcp = lm[9];
  const scale = Math.sqrt((mcp.x - w.x)**2 + (mcp.y - w.y)**2 + (mcp.z - w.z)**2) || 1;
  const out = [];
  for (let i = 0; i < 21; i++) {
    out.push((lm[i].x - w.x) / scale);
    out.push((lm[i].y - w.y) / scale);
    out.push((lm[i].z - w.z) / scale);
  }
  return out;
}

function predictGesture(right, left) {
  if (!gestureModel || (!right && !left)) return { label: "none", prob: 0 };
  const rD = right ? normalizeHand(right) : getEmptyHand();
  const lD = left ? normalizeHand(left) : getEmptyHand();
  const input = tf.tensor2d([[...rD, ...lD]]);
  const prediction = gestureModel.predict(input).dataSync();
  input.dispose();
  const maxIdx = prediction.indexOf(Math.max(...prediction));
  const labels = ["none", "clone", "rasengan"];
  return { label: labels[maxIdx], prob: prediction[maxIdx] };
}

// 5. Desenho do Rasengan (Com Offset e Brilho)
function drawRasengan(handLandmarks) {
  if (!imgRasengan.complete || imgRasengan.naturalWidth === 0) return;

  const palm = handLandmarks[9]; 
  const wrist = handLandmarks[0];

  const offsetX = (palm.x - wrist.x) * 100; 
  const offsetY = (palm.y - wrist.y) * 100;

  const x = palm.x * canvas.width + offsetX;
  const y = palm.y * canvas.height + offsetY;
  const size = 450; 

  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rasenganRotation);
  ctx.globalCompositeOperation = "lighter"; 
  ctx.drawImage(imgRasengan, -size/2, -size/2, size, size);
  ctx.restore();
  rasenganRotation += 0.3;
}

// 6. Loop Principal com Gerenciamento de Tempo
holistic.onResults((res) => {
  if (!mask) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const person = grabPerson();
  const prediction = predictGesture(res.rightHandLandmarks, res.leftHandLandmarks);
  
  const confidenceEl = document.querySelector(".confidence");
  if (confidenceEl) {
    confidenceEl.textContent = `${prediction.label.toUpperCase()} (${(prediction.prob * 100).toFixed(1)}%)`;
  }

  const activeHand = res.rightHandLandmarks || res.leftHandLandmarks;

  // Lógica de Renderização dos Clones com Temporizador
  if (clonesTriggered) {
    const now = performance.now();
    const elapsed = now - cloneStartTime;

    // VERIFICAÇÃO: Se passaram 10 segundos, os clones somem
    if (elapsed > CLONE_DURATION) {
      clonesTriggered = false;
      cloneStartTime = null;
      // Resetamos a fumaça para permitir novo disparo
      customClones.forEach(cl => cl.smokeSpawned = false);
      console.log("💨 Os clones sumiram após 10 segundos!");
    } else {
      // Continua desenhando se estiver no tempo
      customClones.forEach((cl) => {
        if (!cl.smokeSpawned && elapsed >= cl.delay) {
          cl.smokeSpawned = true;
          const centerX = cl.x + canvas.width / 2;
          const centerY = cl.y + canvas.height / 2 - 40;
          spawnSmoke(centerX - 15, centerY, cl.scale);
          spawnSmoke(centerX + 15, centerY, cl.scale);
        }
      });
      drawClones(person);
      drawSmokes();
    }
  } else {
    ctx.drawImage(person, 0, 0);
  }

  // Lógica de Gatilho e Rasengan
  if (prediction.label === "rasengan" && prediction.prob > 0.85 && activeHand) {
      drawRasengan(activeHand);
  } 
  else if (prediction.label === "clone" && prediction.prob > 0.99 && !clonesTriggered) {
      clonesTriggered = true;
      cloneStartTime = performance.now();
      console.log("🔥 JUTSU: Kage Bunshin!");
  }
});

// 7. Funções de Suporte (Grab, Smoke, Clones)
function grabPerson() {
  const offscreen = document.createElement("canvas");
  offscreen.width = canvas.width; offscreen.height = canvas.height;
  const tempCtx = offscreen.getContext("2d");
  tempCtx.drawImage(mask, 0, 0, canvas.width, canvas.height);
  tempCtx.globalCompositeOperation = "source-in";
  tempCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return offscreen;
}

const customClones = [
  // Lado Esquerdo (Mais próximo de você)
  { x: -350, y: 80,  scale: 0.85, delay: 1000, smokeSpawned: false },
  
  // Lado Direito (Mais próximo de você)
  { x: 350,  y: 80,  scale: 0.85, delay: 1200, smokeSpawned: false },
  
  // Lado Esquerdo (Mais longe e mais "atrás")
  { x: -650, y: 150, scale: 0.7,  delay: 1400, smokeSpawned: false },
  
  // Lado Direito (Mais longe e mais "atrás")
  { x: 650,  y: 150, scale: 0.7,  delay: 1650, smokeSpawned: false },
  
  // Clone de Fundo (Centralizado mas pequeno)
  { x: 0,    y: -50, scale: 0.5,  delay: 2000, smokeSpawned: false },
];

function drawClones(person) {
  const now = performance.now();
  customClones.forEach((cl) => {
    if (now - cloneStartTime >= cl.delay) {
      ctx.save();
      ctx.translate(cl.x + canvas.width * (1 - cl.scale) / 2, cl.y);
      ctx.scale(cl.scale, cl.scale);
      ctx.drawImage(person, 0, 0);
      ctx.restore();
    }
  });
  ctx.drawImage(person, 0, 0);
}

const SMOKE_FOLDERS = ["smoke_1", "smoke_2", "smoke_3"];
const activeSmokes = [];
function spawnSmoke(x, y, scale) {
  const folder = SMOKE_FOLDERS[Math.floor(Math.random() * SMOKE_FOLDERS.length)];
  const frames = [];
  for (let i = 1; i <= 5; i++) {
    const img = new Image();
    img.src = `assets/${folder}/${i}.png`;
    frames.push(img);
  }
  activeSmokes.push({ x, y, scale: scale * 1.2, start: performance.now(), frames });
}

function drawSmokes() {
  const now = performance.now();
  for (let i = activeSmokes.length - 1; i >= 0; i--) {
    const s = activeSmokes[i];
    const elapsed = now - s.start;
    const frameIndex = Math.floor(elapsed / 120);
    if (frameIndex >= s.frames.length) { activeSmokes.splice(i, 1); continue; }
    const img = s.frames[frameIndex];
    if (img.complete) { ctx.save(); ctx.translate(s.x, s.y); ctx.scale(s.scale, s.scale); ctx.drawImage(img, -img.width / 2, -img.height / 2); ctx.restore(); }
  }
}

window.onload = () => { clonesTriggered = false; };