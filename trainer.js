// 1. Força o uso da CPU para estabilidade
tf.setBackend('cpu').then(() => {
    console.log("Trainer: Backend definido para CPU");
});

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const holistic = new Holistic({
  locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}`
});
holistic.setOptions({ modelComplexity: 1, smoothLandmarks: true });

const cam = new Camera(video, {
  width: 640, height: 480,
  onFrame: async () => await holistic.send({ image: video })
});
cam.start();

let samples = { clone_sign: [], rasengan_sign: [], not_sign: [] };
let recording = null;
let model = null;
const labels = ["Outros", "Clone", "Rasengan"];
const statusEl = document.getElementById("train-status");

// FUNÇÃO AUXILIAR: Cria uma "mão vazia" (63 zeros) para quando apenas uma mão estiver visível
function getEmptyHand() {
  return new Array(63).fill(0);
}

holistic.onResults(res => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
  ctx.restore();

  if (res.rightHandLandmarks) drawHand(res.rightHandLandmarks);
  if (res.leftHandLandmarks) drawHand(res.leftHandLandmarks);

  // MUDANÇA: Agora tentamos capturar mesmo que uma das mãos seja nula
  captureFrame(res.rightHandLandmarks, res.leftHandLandmarks);

  // Live inference ajustada para lidar com mãos ausentes (importante para testar o Rasengan)
  if (model && (res.rightHandLandmarks || res.leftHandLandmarks)) {
    const rD = res.rightHandLandmarks ? normalize(res.rightHandLandmarks) : getEmptyHand();
    const lD = res.leftHandLandmarks ? normalize(res.leftHandLandmarks) : getEmptyHand();
    
    const input = tf.tensor2d([[...rD, ...lD]]);
    const prediction = model.predict(input).dataSync();
    input.dispose();
    
    const maxIdx = prediction.indexOf(Math.max(...prediction));
    updateConf(prediction[maxIdx], labels[maxIdx]);
  }
});

function drawHand(lm) {
  const segs = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]];
  ctx.strokeStyle = "#22c55e"; ctx.lineWidth = 2;
  for (const seg of segs) {
    ctx.beginPath();
    seg.forEach((i, idx) => {
      const x = canvas.width - lm[i].x * canvas.width;
      const y = lm[i].y * canvas.height;
      idx === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}

function normalize(lm) {
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

function captureFrame(right, left) {
  // MUDANÇA: Agora gravamos se houver PELO MENOS uma mão. 
  // Se ambas forem nulas, não grava nada.
  if (!recording || (!right && !left)) return;

  const rightData = right ? normalize(right) : getEmptyHand();
  const leftData = left ? normalize(left) : getEmptyHand();
  
  samples[recording].push([...rightData, ...leftData]);

  // Atualiza os contadores na tela
  document.getElementById("count-clone").textContent = samples.clone_sign.length;
  document.getElementById("count-rasengan").textContent = samples.rasengan_sign.length;
  document.getElementById("count-other").textContent = samples.not_sign.length;
}

function startCountdown(label) {
  stopRec();
  const badge = document.getElementById("rec-badge");
  let rem = 3;
  badge.classList.add("active");
  const inter = setInterval(() => {
    badge.textContent = `PREPARAR… ${rem}`;
    if (rem <= 0) { clearInterval(inter); startRec(label); }
    rem--;
  }, 1000);
}

function startRec(label) {
  recording = label;
  const badge = document.getElementById("rec-badge");
  let rem = 4;
  const inter = setInterval(() => {
    badge.textContent = `● REC ${rem}s`;
    if (rem <= 0) { stopRec(); badge.classList.remove("active"); statusEl.textContent = "Fim da captura!"; clearInterval(inter); }
    rem--;
  }, 1000);
}

function stopRec() { recording = null; }

document.getElementById("btn-rec-clone").onclick = () => startCountdown("clone_sign");
document.getElementById("btn-rec-rasengan").onclick = () => startCountdown("rasengan_sign");
document.getElementById("btn-rec-other").onclick = () => startCountdown("not_sign");

document.getElementById("btn-train").onclick = async () => {
  const xs = [], ys = [];
  samples.not_sign.forEach(s => { xs.push(s); ys.push(0); });
  samples.clone_sign.forEach(s => { xs.push(s); ys.push(1); });
  samples.rasengan_sign.forEach(s => { xs.push(s); ys.push(2); });

  if (xs.length === 0) { statusEl.textContent = "Grave amostras antes!"; return; }

  const xT = tf.tensor2d(xs);
  const yT = tf.tensor1d(ys, 'float32'); 

  if (model) model.dispose();
  model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [126], units: 64, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: 32, activation: "relu" }));
  model.add(tf.layers.dense({ units: 3, activation: "softmax" }));

  model.compile({ 
    optimizer: "adam", 
    loss: "sparseCategoricalCrossentropy", 
    metrics: ["accuracy"] 
  });

  statusEl.textContent = "Iniciando treinamento...";
  document.getElementById("btn-train").disabled = true;

  await model.fit(xT, yT, { 
    epochs: 60, 
    batchSize: 16, 
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        statusEl.textContent = `Treinando: Epoch ${epoch + 1}/60 | Precisão: ${(logs.acc * 100).toFixed(1)}%`;
      }
    }
  });
  
  statusEl.textContent = "Modelo Pronto!";
  document.getElementById("btn-train").disabled = false;
  xT.dispose(); yT.dispose();
};

function updateConf(prob, label) {
  document.getElementById("conf-fill").style.width = `${(prob * 100)}%`;
  document.getElementById("conf-label").textContent = `${label} (${(prob * 100).toFixed(0)}%)`;
}

document.getElementById("btn-save-model").onclick = async () => {
  if (!model) { statusEl.textContent = "Treine o modelo primeiro!"; return; }
  await model.save("downloads://gesture-model");
  statusEl.textContent = "Modelo baixado! Coloque os arquivos na pasta do projeto.";
};

// Funções de Export/Import/Clear mantidas
document.getElementById("btn-export-data").onclick = () => {
  const blob = new Blob([JSON.stringify(samples)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "gesture-data.json";
  a.click();
};

document.getElementById("btn-import-data").onclick = () => document.getElementById("import-data-input").click();
document.getElementById("import-data-input").onchange = e => {
  const reader = new FileReader();
  reader.onload = ev => {
    const data = JSON.parse(ev.target.result);
    samples.not_sign = data.not_sign || [];
    samples.clone_sign = data.clone_sign || [];
    samples.rasengan_sign = data.rasengan_sign || [];
    document.getElementById("count-clone").textContent = samples.clone_sign.length;
    document.getElementById("count-rasengan").textContent = samples.rasengan_sign.length;
    document.getElementById("count-other").textContent = samples.not_sign.length;
    statusEl.textContent = "Dados importados!";
  };
  reader.readAsText(e.target.files[0]);
};

document.getElementById("btn-clear-data").onclick = () => {
  samples = { clone_sign: [], rasengan_sign: [], not_sign: [] };
  document.getElementById("count-clone").textContent = "0";
  document.getElementById("count-rasengan").textContent = "0";
  document.getElementById("count-other").textContent = "0";
  statusEl.textContent = "Dados limpos.";
};