/**
 * camera-demo.js — Live webcam feed with real-time anomaly detection overlay
 */
(function () {
  'use strict';

  const video = document.getElementById('cameraVideo');
  const overlay = document.getElementById('cameraOverlay');
  const startBtn = document.getElementById('cameraStartBtn');
  const stopBtn = document.getElementById('cameraStopBtn');
  const captureBtn = document.getElementById('cameraCaptureBtn');
  const statusEl = document.getElementById('cameraStatus');

  let stream = null;
  let animFrame = null;
  let isRunning = false;

  const IMG_SIZE = 224;

  if (!startBtn) return;

  startBtn.addEventListener('click', startCamera);
  stopBtn.addEventListener('click', stopCamera);
  captureBtn.addEventListener('click', captureFrame);

  async function startCamera() {
    try {
      statusEl.textContent = 'Starting camera...';
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
      });
      video.srcObject = stream;
      await video.play();

      startBtn.style.display = 'none';
      stopBtn.style.display = 'inline-flex';
      captureBtn.style.display = 'inline-flex';
      statusEl.textContent = 'Camera active — point at an object';
      isRunning = true;
    } catch (err) {
      console.error('Camera error:', err);
      statusEl.textContent = 'Camera access denied or unavailable';
    }
  }

  function stopCamera() {
    isRunning = false;
    if (animFrame) cancelAnimationFrame(animFrame);
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    video.srcObject = null;

    startBtn.style.display = 'inline-flex';
    stopBtn.style.display = 'none';
    captureBtn.style.display = 'none';
    statusEl.textContent = 'Camera stopped';
  }

  function captureFrame() {
    if (!video.videoWidth) return;

    const canvas = document.createElement('canvas');
    canvas.width = IMG_SIZE;
    canvas.height = IMG_SIZE;
    const ctx = canvas.getContext('2d');

    // Center-crop the video to square, then resize to 224×224
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const side = Math.min(vw, vh);
    const sx = (vw - side) / 2;
    const sy = (vh - side) / 2;

    ctx.drawImage(video, sx, sy, side, side, 0, 0, IMG_SIZE, IMG_SIZE);
    const imageData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);

    // Pass to inspector-demo.js via the global function
    if (typeof window.setInspectorImage === 'function') {
      window.setInspectorImage(imageData, '');
      statusEl.textContent = 'Frame captured — click "Run Inspection"';
    }
  }
})();
