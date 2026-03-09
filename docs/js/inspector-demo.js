/**
 * inspector-demo.js — ONNX inference, anomaly detection, heatmap visualization
 *
 * Loads SALT ViT model via ONNX Runtime Web, runs inference on 224×224 images,
 * computes per-patch anomaly scores against normal references, and renders heatmaps.
 */
(function () {
  'use strict';

  let ort = null;
  let session = null;
  let selectedImageData = null;
  let selectedCategory = '';
  let referenceData = null;
  let samplesData = null;

  const IMG_SIZE = 224;
  const GRID_SIZE = 14;
  const NUM_PATCHES = 196;
  const EMBED_DIM = 192;

  // ImageNet normalization
  const MEAN = [0.485, 0.456, 0.406];
  const STD = [0.229, 0.224, 0.225];

  // DOM elements
  const sampleGrid = document.getElementById('sampleGrid');
  const categorySelect = document.getElementById('categorySelect');
  const runBtn = document.getElementById('runBtn');
  const previewCanvas = document.getElementById('previewCanvas');
  const heatmapCanvas = document.getElementById('heatmapCanvas');
  const resultsPlaceholder = document.getElementById('demoResults');
  const resultsContent = document.getElementById('demoResultsContent');
  const scoreValue = document.getElementById('scoreValue');
  const scoreFill = document.getElementById('scoreFill');
  const scoreVerdict = document.getElementById('scoreVerdict');
  const inferenceTimeEl = document.getElementById('inferenceTime');
  const detectedCategoryEl = document.getElementById('detectedCategory');
  const embeddingShapeEl = document.getElementById('embeddingShape');
  const knnResultsEl = document.getElementById('knnResults');
  const uploadZone = document.getElementById('uploadZone');
  const fileInput = document.getElementById('fileInput');

  // ---- Load ONNX Runtime ----
  function loadOrtScript() {
    return new Promise((resolve, reject) => {
      if (window.ort) { ort = window.ort; resolve(); return; }
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js';
      script.onload = () => { ort = window.ort; resolve(); };
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  // ---- Load JSON helper ----
  async function loadJSON(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to load ${url}: ${resp.status}`);
    return resp.json();
  }

  // ---- Load sample images ----
  async function loadSamples() {
    if (samplesData) return samplesData;
    try {
      samplesData = await loadJSON('data/mvtec-samples.json');
    } catch (e) {
      console.warn('Sample data not found:', e);
      samplesData = {};
    }
    return samplesData;
  }

  // ---- Load reference embeddings ----
  async function loadReferences() {
    if (referenceData) return referenceData;
    try {
      referenceData = await loadJSON('data/reference-embeddings.json');
    } catch (e) {
      console.warn('Reference data not found:', e);
      referenceData = null;
    }
    return referenceData;
  }

  // ---- Populate category select and sample grid ----
  async function populateSamples() {
    const rawSamples = await loadSamples();
    // Support both formats: {samples: {cat: [...]}} and {cat: [...]}
    const samples = rawSamples.samples || rawSamples;
    const categories = Object.keys(samples).filter(k => Array.isArray(samples[k]));

    if (categories.length === 0) {
      sampleGrid.innerHTML = '<div class="sample-placeholder">No samples available yet. Train the model first.</div>';
      return;
    }

    // Populate category select
    categorySelect.innerHTML = '<option value="all">All Categories</option>';
    categories.forEach((cat) => {
      const opt = document.createElement('option');
      opt.value = cat;
      opt.textContent = cat.replace(/_/g, ' ');
      categorySelect.appendChild(opt);
    });

    categorySelect.addEventListener('change', () => renderSampleGrid(samples));
    renderSampleGrid(samples);
  }

  function renderSampleGrid(samples) {
    const filter = categorySelect.value;
    sampleGrid.innerHTML = '';

    const categories = filter === 'all' ? Object.keys(samples) : [filter];

    categories.forEach((cat) => {
      if (!samples[cat]) return;
      samples[cat].forEach((sample) => {
        // Normalize fields: support both old format (base64/is_defective/defect_type)
        // and new Colab format (image/label)
        const imageData = sample.image || sample.base64;
        const isDefective = sample.is_defective !== undefined ? sample.is_defective : (sample.label !== 'good');
        const defectType = sample.defect_type || sample.label || 'unknown';
        sample.category = sample.category || cat;

        const wrapper = document.createElement('div');
        wrapper.className = 'sample-item';

        const img = document.createElement('img');
        img.className = 'sample-img' + (isDefective ? ' defective' : '');
        img.src = 'data:image/jpeg;base64,' + imageData;
        img.alt = cat + (isDefective ? ' (defective)' : ' (normal)');
        img.title = cat + ' — ' + defectType;
        img.addEventListener('click', () => selectSampleImage(img, sample));

        const label = document.createElement('span');
        label.className = 'sample-label';
        label.textContent = defectType;

        wrapper.appendChild(img);
        wrapper.appendChild(label);
        sampleGrid.appendChild(wrapper);
      });
    });
  }

  // ---- Select sample ----
  function selectSampleImage(imgEl, sample) {
    document.querySelectorAll('.sample-img').forEach((el) => el.classList.remove('selected'));
    imgEl.classList.add('selected');

    selectedCategory = sample.category;

    const img = new Image();
    img.onload = () => {
      // Draw preview
      const ctx = previewCanvas.getContext('2d');
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE);

      // Store raw pixel data for inference
      const tmpCanvas = document.createElement('canvas');
      tmpCanvas.width = IMG_SIZE;
      tmpCanvas.height = IMG_SIZE;
      const tctx = tmpCanvas.getContext('2d');
      tctx.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE);
      selectedImageData = tctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
    };
    img.src = 'data:image/jpeg;base64,' + (sample.image || sample.base64);

    runBtn.disabled = false;
  }

  // ---- Upload handling ----
  if (uploadZone && fileInput) {
    uploadZone.addEventListener('click', () => fileInput.click());
    uploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadZone.classList.add('dragover');
    });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
    uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadZone.classList.remove('dragover');
      if (e.dataTransfer.files.length) handleFileUpload(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', () => {
      if (fileInput.files.length) handleFileUpload(fileInput.files[0]);
    });
  }

  function handleFileUpload(file) {
    if (!file.type.startsWith('image/')) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const ctx = previewCanvas.getContext('2d');
        ctx.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE);

        const tmpCanvas = document.createElement('canvas');
        tmpCanvas.width = IMG_SIZE;
        tmpCanvas.height = IMG_SIZE;
        const tctx = tmpCanvas.getContext('2d');
        tctx.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE);
        selectedImageData = tctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);

        // Auto-detect category (will be determined by k-NN)
        selectedCategory = '';
        document.querySelectorAll('.sample-img').forEach((el) => el.classList.remove('selected'));
        runBtn.disabled = false;
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  // ---- Allow camera-demo.js to set image data ----
  window.setInspectorImage = function (imageData, category) {
    selectedImageData = imageData;
    selectedCategory = category || '';

    const ctx = previewCanvas.getContext('2d');
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = imageData.width;
    tmpCanvas.height = imageData.height;
    tmpCanvas.getContext('2d').putImageData(imageData, 0, 0);
    ctx.drawImage(tmpCanvas, 0, 0, IMG_SIZE, IMG_SIZE);

    document.querySelectorAll('.sample-img').forEach((el) => el.classList.remove('selected'));
    runBtn.disabled = false;
  };

  // ---- Preprocess image for ONNX ----
  function preprocessImage(imageData) {
    const { data, width, height } = imageData;
    const tensor = new Float32Array(1 * 3 * height * width);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const r = data[idx] / 255.0;
        const g = data[idx + 1] / 255.0;
        const b = data[idx + 2] / 255.0;
        tensor[0 * height * width + y * width + x] = (r - MEAN[0]) / STD[0];
        tensor[1 * height * width + y * width + x] = (g - MEAN[1]) / STD[1];
        tensor[2 * height * width + y * width + x] = (b - MEAN[2]) / STD[2];
      }
    }
    return tensor;
  }

  // ---- Run inference ----
  async function runInference() {
    if (!selectedImageData) return;

    runBtn.disabled = true;
    runBtn.classList.add('loading');
    runBtn.innerHTML = '<span class="spinner"></span> Loading model...';

    try {
      if (!ort) await loadOrtScript();

      if (!session) {
        runBtn.innerHTML = '<span class="spinner"></span> Loading SALT Inspector...';
        try {
          session = await ort.InferenceSession.create('models/salt-inspector.onnx', {
            executionProviders: ['wasm'],
          });
        } catch (e) {
          console.warn('ONNX model not found, using simulated inference');
          showSimulatedResults();
          return;
        }
      }

      runBtn.innerHTML = '<span class="spinner"></span> Inspecting...';

      // Load references if needed
      await loadReferences();

      // Preprocess
      const tensor = preprocessImage(selectedImageData);
      const input = new ort.Tensor('float32', tensor, [1, 3, IMG_SIZE, IMG_SIZE]);

      // Run inference
      const t0 = performance.now();
      const results = await session.run({ image: input });
      const elapsed = performance.now() - t0;

      // Extract output: (1, 196, 192)
      const embeddings = results.embeddings;
      const data = embeddings.data;
      const shape = embeddings.dims;

      displayResults(data, shape, elapsed);
    } catch (err) {
      console.error('Inference failed:', err);
      showSimulatedResults();
    } finally {
      runBtn.disabled = false;
      runBtn.classList.remove('loading');
      runBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 2l10 6-10 6V2z" fill="currentColor"/></svg>
        Run Inspection
      `;
    }
  }

  // ---- Compute anomaly detection ----
  function computeAnomaly(patchEmbeddings, category) {
    if (!referenceData || !referenceData.references) {
      return { scores: new Float32Array(NUM_PATCHES).fill(0.5), bestCategory: category, similarities: {} };
    }

    // Normalize patch embeddings
    const normalizedPatches = [];
    for (let p = 0; p < NUM_PATCHES; p++) {
      const patch = [];
      let norm = 0;
      for (let d = 0; d < EMBED_DIM; d++) {
        const v = patchEmbeddings[p * EMBED_DIM + d];
        patch.push(v);
        norm += v * v;
      }
      norm = Math.sqrt(norm) + 1e-8;
      normalizedPatches.push(patch.map((v) => v / norm));
    }

    // Global embedding (mean pool + normalize)
    const globalEmb = new Array(EMBED_DIM).fill(0);
    for (let p = 0; p < NUM_PATCHES; p++) {
      for (let d = 0; d < EMBED_DIM; d++) {
        globalEmb[d] += normalizedPatches[p][d] / NUM_PATCHES;
      }
    }
    let globalNorm = 0;
    for (let d = 0; d < EMBED_DIM; d++) globalNorm += globalEmb[d] * globalEmb[d];
    globalNorm = Math.sqrt(globalNorm) + 1e-8;
    for (let d = 0; d < EMBED_DIM; d++) globalEmb[d] /= globalNorm;

    // Find best matching category by global similarity
    let bestCategory = category;
    let bestSim = -1;
    const similarities = {};

    for (const [cat, ref] of Object.entries(referenceData.references)) {
      let dot = 0;
      for (let d = 0; d < Math.min(EMBED_DIM, ref.global_embedding.length); d++) {
        dot += globalEmb[d] * ref.global_embedding[d];
      }
      similarities[cat] = dot;
      if (dot > bestSim) {
        bestSim = dot;
        bestCategory = cat;
      }
    }

    // Use the best category (or specified one) for patch-level scoring
    const refCat = category && referenceData.references[category] ? category : bestCategory;
    const refPatches = referenceData.references[refCat]?.patch_embeddings;

    let scores;
    if (refPatches && refPatches.length === NUM_PATCHES) {
      scores = new Float32Array(NUM_PATCHES);
      for (let p = 0; p < NUM_PATCHES; p++) {
        let dot = 0;
        for (let d = 0; d < Math.min(EMBED_DIM, refPatches[p].length); d++) {
          dot += normalizedPatches[p][d] * refPatches[p][d];
        }
        // Anomaly = 1 - similarity (higher = more anomalous)
        scores[p] = Math.max(0, 1.0 - dot);
      }
    } else {
      scores = new Float32Array(NUM_PATCHES).fill(0.5);
    }

    return { scores, bestCategory, similarities };
  }

  // ---- Display results ----
  function displayResults(data, shape, elapsed) {
    resultsPlaceholder.style.display = 'none';
    resultsContent.style.display = 'block';

    // Compute anomaly detection
    const { scores, bestCategory, similarities } = computeAnomaly(data, selectedCategory);

    // Render heatmap
    renderHeatmap(scores);

    // Anomaly score (mean of top-10% patches for robustness)
    const sorted = Array.from(scores).sort((a, b) => b - a);
    const top10 = sorted.slice(0, Math.max(1, Math.floor(NUM_PATCHES * 0.1)));
    const anomalyScore = top10.reduce((a, b) => a + b, 0) / top10.length;
    const scorePercent = Math.round(anomalyScore * 100);

    scoreValue.textContent = scorePercent + '%';

    // Score bar color
    let barColor, verdict, verdictClass;
    if (scorePercent < 20) {
      barColor = '#ef4444';
      verdict = 'PASS — No anomaly detected';
      verdictClass = 'pass';
    } else if (scorePercent < 45) {
      barColor = '#f59e0b';
      verdict = 'WARNING — Possible anomaly detected';
      verdictClass = 'warning';
    } else {
      barColor = '#ef4444';
      verdict = 'FAIL — Anomaly detected';
      verdictClass = 'fail';
    }

    scoreFill.style.width = scorePercent + '%';
    scoreFill.style.background = barColor;
    scoreVerdict.textContent = verdict;
    scoreVerdict.className = 'score-verdict ' + verdictClass;

    // Metadata
    inferenceTimeEl.textContent = elapsed.toFixed(1) + ' ms';
    detectedCategoryEl.textContent = bestCategory.replace(/_/g, ' ');
    embeddingShapeEl.textContent = shape.join(' × ');

    // k-NN: top-5 most similar categories
    const sortedCats = Object.entries(similarities)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);

    knnResultsEl.innerHTML = sortedCats
      .map(
        ([cat, sim], i) => `
      <div class="result-item">
        <span class="result-label">${i + 1}. ${cat.replace(/_/g, ' ')}</span>
        <span class="result-value${i === 0 ? ' highlight' : ''}">${sim.toFixed(3)}</span>
      </div>
    `
      )
      .join('');
  }

  // ---- Render anomaly heatmap ----
  function renderHeatmap(scores) {
    const ctx = heatmapCanvas.getContext('2d');

    // First draw the original image
    ctx.drawImage(previewCanvas, 0, 0, IMG_SIZE, IMG_SIZE);

    // Then overlay the heatmap
    const heatmapData = ctx.createImageData(IMG_SIZE, IMG_SIZE);
    const patchW = IMG_SIZE / GRID_SIZE;
    const patchH = IMG_SIZE / GRID_SIZE;

    // Normalize scores for visualization
    const maxScore = Math.max(...scores);
    const minScore = Math.min(...scores);
    const range = maxScore - minScore + 1e-8;

    for (let y = 0; y < IMG_SIZE; y++) {
      for (let x = 0; x < IMG_SIZE; x++) {
        const patchY = Math.min(Math.floor(y / patchH), GRID_SIZE - 1);
        const patchX = Math.min(Math.floor(x / patchW), GRID_SIZE - 1);
        const patchIdx = patchY * GRID_SIZE + patchX;
        const normalized = (scores[patchIdx] - minScore) / range;

        // Colormap: green (0) → yellow (0.5) → red (1.0)
        let r, g, b;
        if (normalized < 0.5) {
          const t = normalized * 2;
          r = Math.round(34 + (245 - 34) * t);   // green to yellow
          g = Math.round(197 + (158 - 197) * t);
          b = Math.round(94 + (11 - 94) * t);
        } else {
          const t = (normalized - 0.5) * 2;
          r = Math.round(245 + (239 - 245) * t);  // yellow to red
          g = Math.round(158 + (68 - 158) * t);
          b = Math.round(11 + (68 - 11) * t);
        }

        const alpha = 0.15 + normalized * 0.45; // More transparent for normal regions

        const idx = (y * IMG_SIZE + x) * 4;
        heatmapData.data[idx] = r;
        heatmapData.data[idx + 1] = g;
        heatmapData.data[idx + 2] = b;
        heatmapData.data[idx + 3] = Math.round(alpha * 255);
      }
    }

    // Overlay heatmap on the original image
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = IMG_SIZE;
    tempCanvas.height = IMG_SIZE;
    const tctx = tempCanvas.getContext('2d');
    tctx.putImageData(heatmapData, 0, 0);

    ctx.globalCompositeOperation = 'source-over';
    ctx.drawImage(tempCanvas, 0, 0);
  }

  // ---- Simulated results (when model isn't loaded yet) ----
  function showSimulatedResults() {
    resultsPlaceholder.style.display = 'none';
    resultsContent.style.display = 'block';

    inferenceTimeEl.textContent = 'Simulated';
    detectedCategoryEl.textContent = selectedCategory || 'unknown';
    embeddingShapeEl.textContent = '196 × 192';

    const fakeScores = new Float32Array(NUM_PATCHES);
    for (let i = 0; i < NUM_PATCHES; i++) fakeScores[i] = Math.random() * 0.6;

    renderHeatmap(fakeScores);

    const avgScore = Math.round(fakeScores.reduce((a, b) => a + b) / NUM_PATCHES * 100);
    scoreValue.textContent = avgScore + '%';
    scoreFill.style.width = avgScore + '%';
    scoreFill.style.background = avgScore < 30 ? '#ef4444' : '#f59e0b';
    scoreVerdict.textContent = 'Simulated — train model for real results';
    scoreVerdict.className = 'score-verdict warning';

    knnResultsEl.innerHTML = '<div class="result-item"><span class="result-label">Train model to see real results</span></div>';

    runBtn.disabled = false;
    runBtn.classList.remove('loading');
    runBtn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 2l10 6-10 6V2z" fill="currentColor"/></svg>
      Run Inspection
    `;
  }

  // ---- Event listeners ----
  runBtn.addEventListener('click', runInference);

  // Load on section visibility
  window.addEventListener('section-visible', (e) => {
    if (e.detail.id === 'demo') {
      populateSamples();
      loadOrtScript().catch(() => {});
      loadReferences().catch(() => {});
    }
  });

  // Also load if already visible
  const demoSection = document.getElementById('demo');
  if (demoSection) {
    const rect = demoSection.getBoundingClientRect();
    if (rect.top < window.innerHeight) {
      populateSamples();
    }
  }
})();
