/* =====================================================
   AQI Predictor — Frontend Logic
   ===================================================== */

(function () {
  'use strict';

  const form = document.getElementById('predict-form');
  const btnText = document.querySelector('.btn-text');
  const btnLoader = document.querySelector('.btn-loader');
  const predictBtn = document.getElementById('predict-btn');
  const resultsSection = document.getElementById('results-section');
  const aqiValue = document.getElementById('aqi-value');
  const aqiCategory = document.getElementById('aqi-category');
  const adviceContent = document.getElementById('advice-content');
  const gaugeNeedle = document.getElementById('gauge-needle');

  // ---- Form Submission ----
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    setLoading(true);

    const payload = {
      pm2_5: parseFloat(form.pm2_5.value),
      pm10: parseFloat(form.pm10.value),
      co: parseFloat(form.co.value),
      no2: parseFloat(form.no2.value),
      so2: parseFloat(form.so2.value),
      o3: parseFloat(form.o3.value),
      temperature: parseFloat(form.temperature.value),
      humidity: parseFloat(form.humidity.value),
      health_condition: form.health_condition.value,
      activity: form.activity.value,
    };

    const ageVal = form.age.value.trim();
    if (ageVal) payload.age = parseInt(ageVal, 10);

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      displayResults(data);
    } catch (err) {
      showError(err.message || 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  });

  // ---- Display Results ----
  function displayResults(data) {
    resultsSection.hidden = false;
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Animate AQI value (count up)
    animateCounter(aqiValue, data.aqi);

    // Color the value
    aqiValue.style.color = data.color;

    // Category badge
    aqiCategory.textContent = data.category;
    aqiCategory.style.background = data.color + '22';
    aqiCategory.style.color = data.color;
    aqiCategory.style.border = `1px solid ${data.color}44`;

    // Rotate gauge needle: AQI 0 → -90°, AQI 500 → +90°
    const angle = -90 + (Math.min(data.aqi, 500) / 500) * 180;
    gaugeNeedle.style.transform = `rotate(${angle}deg)`;
    gaugeNeedle.setAttribute('transform', `rotate(${angle}, 100, 100)`);

    // Health advice — render markdown-ish bullet points
    adviceContent.innerHTML = formatAdvice(data.health_advice);
  }

  // ---- Counter Animation ----
  function animateCounter(el, target) {
    const duration = 1200;
    const start = performance.now();
    const from = 0;

    function tick(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      // easeOutExpo
      const eased = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
      const current = from + (target - from) * eased;
      el.textContent = current.toFixed(1);
      if (progress < 1) requestAnimationFrame(tick);
    }

    requestAnimationFrame(tick);
  }

  // ---- Format Advice Text ----
  function formatAdvice(text) {
    if (!text) return '<p>No advice available.</p>';

    // Convert markdown-style bullets to HTML list
    const lines = text.split('\n').filter((l) => l.trim());
    const listItems = lines
      .map((line) => {
        // Strip leading bullet chars: •, -, *, numbered
        const clean = line.replace(/^[\s]*[-•*]\s*/, '').replace(/^\d+\.\s*/, '');
        // Handle **bold** markdown
        const formatted = clean.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        return `<li>${formatted}</li>`;
      })
      .join('');

    return `<ul>${listItems}</ul>`;
  }

  // ---- Loading State ----
  function setLoading(isLoading) {
    predictBtn.disabled = isLoading;
    btnText.hidden = isLoading;
    btnLoader.hidden = !isLoading;
  }

  // ---- Error Toast ----
  function showError(msg) {
    // Remove existing toast
    const existing = document.querySelector('.error-toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.textContent = '⚠ ' + msg;
    document.body.appendChild(toast);

    // Trigger reflow then show
    requestAnimationFrame(() => {
      toast.classList.add('visible');
    });

    setTimeout(() => {
      toast.classList.remove('visible');
      setTimeout(() => toast.remove(), 400);
    }, 5000);
  }
})();
