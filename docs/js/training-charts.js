/**
 * training-charts.js — Render training loss curves from training-metrics.json
 */
(function () {
  'use strict';

  const stage1Container = document.getElementById('stage1Chart');
  const stage2Container = document.getElementById('stage2Chart');
  const metaContainer = document.getElementById('trainingMeta');

  async function loadAndRender() {
    let metrics;
    try {
      const resp = await fetch('data/training-metrics.json');
      if (!resp.ok) throw new Error('Not found');
      metrics = await resp.json();
    } catch (e) {
      // Show placeholder
      if (stage1Container) stage1Container.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:40px">Train the model to see real loss curves</p>';
      if (stage2Container) stage2Container.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:40px">Train the model to see real loss curves</p>';
      return;
    }

    if (metrics.stage1 && stage1Container) {
      renderChart(stage1Container, metrics.stage1, '#ef4444');
    }

    if (metrics.stage2 && stage2Container) {
      renderChart(stage2Container, metrics.stage2, '#f87171');
    }

    if (metaContainer && metrics.config) {
      const c = metrics.config;
      const totalTime = metrics.total_time_seconds;
      const timeStr = totalTime > 3600
        ? `${(totalTime / 3600).toFixed(1)}h`
        : `${Math.round(totalTime / 60)}min`;

      metaContainer.innerHTML = `
        <span class="meta-pill">Image: ${c.img_size}×${c.img_size}px</span>
        <span class="meta-pill">Patches: ${c.num_patches} (${c.grid_size}×${c.grid_size})</span>
        <span class="meta-pill">Embed: ${c.embed_dim}d × ${c.depth}L</span>
        <span class="meta-pill">Epochs: ${c.stage1_epochs} + ${c.stage2_epochs}</span>
        ${totalTime ? `<span class="meta-pill">Training: ${timeStr}</span>` : ''}
      `;
    }
  }

  function renderChart(container, steps, color) {
    container.innerHTML = '';

    if (!steps || steps.length === 0) return;

    // Downsample if too many points
    const maxBars = 200;
    let data = steps;
    if (data.length > maxBars) {
      const stride = Math.ceil(data.length / maxBars);
      data = data.filter((_, i) => i % stride === 0);
    }

    const values = data.map((d) => d.loss);
    const maxVal = Math.max(...values);
    const minVal = Math.min(...values);
    const range = maxVal - minVal || 1;

    data.forEach((d) => {
      const bar = document.createElement('div');
      bar.className = 'chart-bar';
      const normalized = (d.loss - minVal) / range;
      const height = 10 + normalized * 90; // 10% to 100% of container
      bar.style.height = height + '%';
      bar.style.background = color;
      bar.title = `Step ${d.step}: ${d.loss}`;
      container.appendChild(bar);
    });
  }

  // Load when training section is visible
  window.addEventListener('section-visible', (e) => {
    if (e.detail.id === 'training') {
      loadAndRender();
    }
  });

  // Also check if already visible
  const section = document.getElementById('training');
  if (section) {
    const rect = section.getBoundingClientRect();
    if (rect.top < window.innerHeight) {
      loadAndRender();
    }
  }
})();
