const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const facesDiv = document.getElementById('faces');
const analysisStatus = document.getElementById('analysisStatus');
let ctx = null;
if (canvas) {
  ctx = canvas.getContext('2d');
}
let lastFrameDataUrl = null;

// Continuous analysis variables
let isAnalyzing = false;
let analysisInterval = null;
let analysisFrameCount = 0;
const ANALYSIS_INTERVAL_MS = 1000; // Analyze every 1 second

// Helper function to create enhanced emotion display
function createEmotionDisplay(data) {
  const confidence = Math.round(data.confidence * 100);
  const emotionClass = data.emotion.toLowerCase();
  
  return `
    <div class="emotion-result">
      <div class="emotion-badge ${emotionClass}">
        <span style="font-size: 24px;">${data.emoji}</span>
        <span>${data.emotion.toUpperCase()}</span>
      </div>
      <div class="confidence-meter">
        <div class="confidence-label">
          <span>Confidence Level</span>
          <span>${confidence}%</span>
        </div>
        <div class="confidence-bar">
          <div class="confidence-fill" style="width: ${confidence}%"></div>
        </div>
      </div>
      <div style="margin-top: 16px; color: var(--muted); font-size: 14px;">
        ${data.description}
      </div>
    </div>
  `;
}

async function initCamera() {
  if (!video) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
  } catch (e) {
    if (facesDiv) {
      facesDiv.innerHTML = `<div class="face">Camera access denied: ${e.message}</div>`;
    }
  }
}

function captureFrame() {
  if (!video || !canvas || !ctx) return null;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);
  return canvas.toDataURL('image/png');
}

async function analyzeSingleFrame(showLoading = true) {
  if (!facesDiv) return;
  
  if (showLoading) {
    facesDiv.innerHTML = `
      <div class="emotion-result">
        <div class="loading-skeleton loading-badge"></div>
        <div class="confidence-meter">
          <div class="loading-skeleton loading-text"></div>
          <div class="loading-skeleton confidence-bar"></div>
        </div>
      </div>
    `;
  }
  
  const dataUrl = captureFrame();
  if (!dataUrl) {
    if (showLoading) {
      facesDiv.innerHTML = '<div class="emotion-result"><div class="face">📷 Camera not ready</div></div>';
    }
    return;
  }
  
  lastFrameDataUrl = dataUrl;
  
  try {
    const res = await fetch('/api/analyze_frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl }),
    });
    const json = await res.json();
    
    if (!res.ok) throw new Error(json.error || 'Failed');

    if (!json.faces || json.faces.length === 0) {
      facesDiv.innerHTML = '<div class="emotion-result"><div class="face">👤 No faces detected</div></div>';
      return;
    }

    facesDiv.innerHTML = '';
    json.faces.forEach(f => {
      const emotionDisplay = createEmotionDisplay(f);
      const div = document.createElement('div');
      div.innerHTML = emotionDisplay;
      facesDiv.appendChild(div);
      // Contextual wellness CTA
      const cta = document.createElement('div');
      cta.className = 'muted';
      cta.style.marginTop = '6px';
      cta.innerHTML = 'Need a breather? <a href="/wellness">Open Wellness</a>';
      div.appendChild(cta);
    });
    
    // Update analysis status for continuous mode
    if (isAnalyzing && analysisStatus) {
      analysisFrameCount++;
      analysisStatus.textContent = `Continuous analysis active • Frame ${analysisFrameCount} • ${json.faces.length} face(s) detected`;
    }
    
  } catch (e) {
    if (showLoading) {
      facesDiv.innerHTML = `<div class="emotion-result"><div class="face error">❌ Error: ${e.message}</div></div>`;
    }
    
    // Update analysis status with error
    if (isAnalyzing && analysisStatus) {
      analysisStatus.textContent = `Analysis error: ${e.message}`;
    }
  }
}

function startContinuousAnalysis() {
  if (isAnalyzing) return;
  
  isAnalyzing = true;
  analysisFrameCount = 0;
  
  // Update UI
  const startBtn = document.getElementById('startAnalysis');
  if (startBtn) {
    startBtn.textContent = 'Stop Analyzing';
    startBtn.classList.add('analyzing');
  }
  
  if (analysisStatus) {
    analysisStatus.textContent = 'Starting continuous analysis...';
  }
  
  // Start continuous analysis
  analysisInterval = setInterval(() => {
    analyzeSingleFrame(false); // Don't show loading for continuous mode
  }, ANALYSIS_INTERVAL_MS);
  
  // Analyze first frame immediately
  analyzeSingleFrame(false);
}

function stopContinuousAnalysis() {
  if (!isAnalyzing) return;
  
  isAnalyzing = false;
  
  // Clear interval
  if (analysisInterval) {
    clearInterval(analysisInterval);
    analysisInterval = null;
  }
  
  // Update UI
  const startBtn = document.getElementById('startAnalysis');
  if (startBtn) {
    startBtn.textContent = 'Start Analyzing';
    startBtn.classList.remove('analyzing');
  }
  
  if (analysisStatus) {
    analysisStatus.textContent = `Analysis stopped • Processed ${analysisFrameCount} frames`;
  }
}

function toggleContinuousAnalysis() {
  if (isAnalyzing) {
    stopContinuousAnalysis();
  } else {
    startContinuousAnalysis();
  }
}

// Start/Stop continuous analysis button
const startAnalysisBtn = document.getElementById('startAnalysis');
if (startAnalysisBtn) {
  startAnalysisBtn.addEventListener('click', toggleContinuousAnalysis);
}

// Single frame capture button
const snapBtn = document.getElementById('snap');
if (snapBtn) {
  snapBtn.addEventListener('click', async () => {
    // Stop continuous analysis if running
    if (isAnalyzing) {
      stopContinuousAnalysis();
    }
    
    // Analyze single frame with loading
    await analyzeSingleFrame(true);
    
    if (analysisStatus) {
      analysisStatus.textContent = 'Single frame analyzed';
    }
  });
}

const saveBtn = document.getElementById('saveShot');
if (saveBtn) {
  saveBtn.addEventListener('click', async () => {
    if (!lastFrameDataUrl) {
      if (facesDiv) facesDiv.innerHTML = '<div class="face">Capture a frame first (Analyze Frame) before saving.</div>';
      return;
    }
    if (facesDiv) facesDiv.innerHTML = 'Saving screenshot...';
    try {
      const res = await fetch('/api/save_screenshot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: lastFrameDataUrl })
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || 'Failed');
      const link = json.url ? `<a href="${json.url}" target="_blank">Open saved image</a>` : '';
      if (facesDiv) facesDiv.innerHTML = `<div class="face">Saved to ${json.path || 'screenshots'} ${link ? '• ' + link : ''}</div>`;
    } catch (e) {
      if (facesDiv) facesDiv.innerHTML = `<div class="face">Save failed: ${e.message}</div>`;
    }
  });
}

const analyzeTextBtn = document.getElementById('analyzeText');
if (analyzeTextBtn) {
  analyzeTextBtn.addEventListener('click', async () => {
    const textEl = document.getElementById('text');
    const textResult = document.getElementById('textResult');
    if (!textEl || !textResult) return;
    const text = textEl.value;
    
    // Show loading state with skeleton
    textResult.innerHTML = `
      <div class="emotion-result">
        <div class="loading-skeleton loading-badge"></div>
        <div class="confidence-meter">
          <div class="loading-skeleton loading-text"></div>
          <div class="loading-skeleton confidence-bar"></div>
        </div>
      </div>
    `;
    
    try {
      const res = await fetch('/api/analyze_text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || 'Failed');
      
      // Create enhanced emotion display
      textResult.innerHTML = createEmotionDisplay(json) + '<div class="muted" style="margin-top:6px;">Need a breather? <a href="/wellness">Open Wellness</a></div>';
    } catch (e) {
      textResult.innerHTML = `<div class="emotion-result"><div class="face">❌ Error: ${e.message}</div></div>`;
    }
  });
}

// Audio drag & drop functionality
const audioDropZone = document.getElementById('audioDropZone');
const audioFileInput = document.getElementById('audioFile');
const audioPreview = document.getElementById('audioPreview');

if (audioDropZone && audioFileInput) {
  // Click to browse
  audioDropZone.addEventListener('click', () => {
    audioFileInput.click();
  });

  // File input change
  audioFileInput.addEventListener('change', handleAudioFileSelect);

  // Drag & drop events
  audioDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    audioDropZone.classList.add('drag-over');
  });

  audioDropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    audioDropZone.classList.remove('drag-over');
  });

  audioDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    audioDropZone.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('audio/')) {
      // Simulate file input selection
      const dt = new DataTransfer();
      dt.items.add(files[0]);
      audioFileInput.files = dt.files;
      updateAudioDropZone(files[0]);
      const analyzeBtn = document.getElementById('analyzeAudio');
      if (analyzeBtn) analyzeBtn.disabled = false;
    }
  });
}

function handleAudioFileSelect(e) {
  const file = e.target.files[0];
  if (file) {
    updateAudioDropZone(file);
    const analyzeBtn = document.getElementById('analyzeAudio');
    if (analyzeBtn) analyzeBtn.disabled = false;
  }
}

function updateAudioDropZone(file) {
  if (audioDropZone) {
    audioDropZone.innerHTML = `
      <div style="font-size: 32px; margin-bottom: 12px;">✅</div>
      <p style="margin: 8px 0; font-size: 16px; color: var(--accent1);">${file.name}</p>
      <p style="margin: 4px 0; color: var(--muted);">Ready to analyze</p>
    `;
  }
  // Preview selected audio
  if (audioPreview && file) {
    const url = URL.createObjectURL(file);
    audioPreview.src = url;
    try { audioPreview.load(); } catch {}
  }
}

const analyzeAudioBtn = document.getElementById('analyzeAudio');
if (analyzeAudioBtn) {
  analyzeAudioBtn.addEventListener('click', async () => {
    const fileInput = document.getElementById('audioFile');
    const audioResult = document.getElementById('audioResult');
    if (!fileInput || !audioResult) return;
    if (!fileInput.files.length) {
      audioResult.innerHTML = '<div class="emotion-result"><div class="face">🎵 Choose an audio file first.</div></div>';
      return;
    }
    
    // Show loading state
    audioResult.innerHTML = `
      <div class="emotion-result">
        <div class="loading-skeleton loading-badge"></div>
        <div class="confidence-meter">
          <div class="loading-skeleton loading-text"></div>
          <div class="loading-skeleton confidence-bar"></div>
        </div>
      </div>
    `;
    
    const form = new FormData();
    form.append('file', fileInput.files[0]);
    try {
      const res = await fetch('/api/analyze_audio', { method: 'POST', body: form });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || 'Failed');
      
      // Create enhanced emotion display
      audioResult.innerHTML = createEmotionDisplay(json) + '<div class="muted" style="margin-top:6px;">Need a breather? <a href="/wellness">Open Wellness</a></div>';
    } catch (e) {
      audioResult.innerHTML = `<div class="emotion-result"><div class="face">❌ Error: ${e.message}</div></div>`;
    }
  });
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
  if (isAnalyzing) {
    stopContinuousAnalysis();
  }
});

// Handle visibility change (tab switching)
document.addEventListener('visibilitychange', () => {
  if (document.hidden && isAnalyzing) {
    // Pause analysis when tab is not visible to save resources
    if (analysisInterval) {
      clearInterval(analysisInterval);
      analysisInterval = null;
    }
    if (analysisStatus) {
      analysisStatus.textContent = 'Analysis paused (tab not visible)';
    }
  } else if (!document.hidden && isAnalyzing && !analysisInterval) {
    // Resume analysis when tab becomes visible
    analysisInterval = setInterval(() => {
      analyzeSingleFrame(false);
    }, ANALYSIS_INTERVAL_MS);
    if (analysisStatus) {
      analysisStatus.textContent = 'Analysis resumed';
    }
  }
});

if (video) {
  initCamera();
}

// --- Audio Recording (WAV) and Analyze ---
const recordBtn = document.getElementById('recordAudio');
const stopBtn = document.getElementById('stopRecording');
const recordStatus = document.getElementById('recordStatus');

let audioCtx = null;
let mediaStream = null;
let mediaSource = null;
let processor = null;
let recordedBuffers = [];
let recording = false;
let recordStartTs = 0;
let recordTimer = null;
const MAX_RECORD_MS = 15000; // auto-stop after 15s

function updateRecordUI(state, message) {
  if (recordBtn) recordBtn.disabled = state === 'recording';
  if (stopBtn) stopBtn.disabled = state !== 'recording';
  if (recordStatus) recordStatus.textContent = message || '';
}

function floatTo16BitPCM(float32Array) {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);
  let offset = 0;
  for (let i = 0; i < float32Array.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return view;
}

function writeWavHeader(view, sampleRate, numSamples, numChannels = 1) {
  function writeString(view, offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = numSamples * bytesPerSample;
  // RIFF chunk descriptor
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');
  // fmt subchunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);        // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true);         // AudioFormat (1 = PCM)
  view.setUint16(22, numChannels, true);// NumChannels
  view.setUint32(24, sampleRate, true);// SampleRate
  view.setUint32(28, byteRate, true);  // ByteRate
  view.setUint16(32, blockAlign, true);// BlockAlign
  view.setUint16(34, 16, true);        // BitsPerSample
  // data subchunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);
}

function encodeWav(buffers, sampleRate) {
  // Merge mono buffers
  const length = buffers.reduce((sum, b) => sum + b.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  for (const b of buffers) { merged.set(b, offset); offset += b.length; }

  const pcmView = floatTo16BitPCM(merged);
  const wavBuffer = new ArrayBuffer(44 + pcmView.byteLength);
  const view = new DataView(wavBuffer);
  writeWavHeader(view, sampleRate, merged.length, 1);
  // PCM data
  const bytes = new Uint8Array(wavBuffer, 44);
  for (let i = 0; i < pcmView.byteLength; i++) bytes[i] = pcmView.getUint8(i);
  // Return Blob from the underlying ArrayBuffer to avoid DataView compatibility quirks
  return new Blob([wavBuffer], { type: 'audio/wav' });
}

async function startRecording() {
  if (recording) return;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    updateRecordUI('idle', `Microphone access denied: ${e.message}`);
    return;
  }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  mediaSource = audioCtx.createMediaStreamSource(mediaStream);
  // Use ScriptProcessor (deprecated but widely supported) for simplicity
  const bufferSize = 4096;
  processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);
  recordedBuffers = [];
  processor.onaudioprocess = (e) => {
    const channel = e.inputBuffer.getChannelData(0);
    recordedBuffers.push(new Float32Array(channel));
  };
  mediaSource.connect(processor);
  processor.connect(audioCtx.destination);
  recording = true;
  recordStartTs = Date.now();
  updateRecordUI('recording', 'Recording… click Stop to analyze');
  // Start UI timer and auto-stop
  if (recordTimer) clearInterval(recordTimer);
  recordTimer = setInterval(() => {
    const elapsed = Date.now() - recordStartTs;
    const sec = Math.floor(elapsed / 1000);
    if (recordStatus) recordStatus.textContent = `Recording… ${sec}s (auto-stops at ${Math.floor(MAX_RECORD_MS/1000)}s)`;
    if (elapsed >= MAX_RECORD_MS) {
      stopRecordingAndAnalyze();
    }
  }, 250);
}

async function stopRecordingAndAnalyze() {
  if (!recording) return;
  recording = false;
  try {
    mediaSource && mediaSource.disconnect();
    processor && processor.disconnect();
  } catch {}
  try {
    mediaStream && mediaStream.getTracks().forEach(t => t.stop());
  } catch {}
  if (recordTimer) { clearInterval(recordTimer); recordTimer = null; }
  const sampleRate = audioCtx ? audioCtx.sampleRate : 44100;
  if (audioCtx) { try { await audioCtx.close(); } catch {} }

  updateRecordUI('idle', 'Encoding…');
  // Encode WAV and send to server
  try {
    if (!recordedBuffers || recordedBuffers.length === 0) {
      updateRecordUI('idle', 'No audio captured. Please try again.');
      return;
    }
    const wavBlob = encodeWav(recordedBuffers, sampleRate);
    // Preview recorded audio
    if (audioPreview) {
      const url = URL.createObjectURL(wavBlob);
      audioPreview.src = url;
      try { audioPreview.load(); } catch {}
    }
    const form = new FormData();
    form.append('file', wavBlob, 'recording.wav');
    const audioResult = document.getElementById('audioResult');
    if (audioResult) audioResult.innerHTML = 'Analyzing recorded audio...';
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000);
    const res = await fetch('/api/analyze_audio', { method: 'POST', body: form, signal: controller.signal });
    clearTimeout(timeout);
    const json = await res.json();
    if (!res.ok) throw new Error(json.error || 'Failed');
    if (audioResult) {
      audioResult.innerHTML = createEmotionDisplay(json);
    }
    updateRecordUI('idle', 'Ready');
  } catch (e) {
    const audioResult = document.getElementById('audioResult');
    if (audioResult) audioResult.innerHTML = `<div class="face">Record analyze error: ${e.name === 'AbortError' ? 'Request timed out' : e.message}</div>`;
    updateRecordUI('idle', `Record analyze error: ${e.name === 'AbortError' ? 'Request timed out' : e.message}`);
  }
}

if (recordBtn && stopBtn) {
  updateRecordUI('idle', 'Ready');
  recordBtn.addEventListener('click', startRecording);
  stopBtn.addEventListener('click', stopRecordingAndAnalyze);
}

// Initialize enhancements on page load
document.addEventListener('DOMContentLoaded', function() {
  // Add page transition class to main content
  const main = document.querySelector('main');
  if (main) {
    main.classList.add('page-transition');
  }
  
  // Initialize camera on vision page
  if (video) {
    initCamera();
  }
  
  // Add smooth scrolling to navigation links
  const navLinks = document.querySelectorAll('.nav a');
  navLinks.forEach(link => {
    link.addEventListener('click', function(e) {
      // Add loading effect
      this.style.opacity = '0.7';
      setTimeout(() => {
        this.style.opacity = '1';
      }, 300);
    });
  });
  
  // Add hover effects to cards
  const cards = document.querySelectorAll('.card');
  cards.forEach(card => {
    card.addEventListener('mouseenter', function() {
      this.style.transform = 'translateY(-8px)';
    });
    
    card.addEventListener('mouseleave', function() {
      this.style.transform = 'translateY(0)';
    });
  });
  
  // Enhanced dropdown interactions
  const selectElements = document.querySelectorAll('select');
  selectElements.forEach(select => {
    // Add smooth transitions for select elements
    select.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
    
    select.addEventListener('mouseenter', function() {
      this.style.borderColor = 'var(--accent1)';
      this.style.boxShadow = '0 4px 12px rgba(0, 188, 212, 0.15)';
    });
    
    select.addEventListener('mouseleave', function() {
      if (!this.matches(':focus')) {
        this.style.borderColor = 'var(--border)';
        this.style.boxShadow = 'none';
      }
    });
    
    select.addEventListener('focus', function() {
      this.style.borderColor = 'var(--accent1)';
      this.style.boxShadow = '0 0 0 3px rgba(0, 188, 212, 0.2)';
    });
    
    select.addEventListener('blur', function() {
      this.style.borderColor = 'var(--border)';
      this.style.boxShadow = 'none';
    });
  });
  
  // Add ripple effect to dropdown selections
  window.addRippleEffect = function(element) {
    element.addEventListener('click', function(e) {
      const ripple = document.createElement('span');
      const rect = this.getBoundingClientRect();
      const size = Math.max(rect.width, rect.height);
      const x = e.clientX - rect.left - size / 2;
      const y = e.clientY - rect.top - size / 2;
      
      ripple.style.cssText = `
        position: absolute;
        border-radius: 50%;
        background: rgba(0, 188, 212, 0.3);
        transform: scale(0);
        animation: ripple 0.6s linear;
        width: ${size}px;
        height: ${size}px;
        left: ${x}px;
        top: ${y}px;
        pointer-events: none;
      `;
      
      this.style.position = 'relative';
      this.style.overflow = 'hidden';
      this.appendChild(ripple);
      
      setTimeout(() => {
        ripple.remove();
      }, 600);
    });
  };

  // ===== Scroll Reveal (IntersectionObserver) =====
  try {
    const revealEls = document.querySelectorAll('[data-reveal], [data-stagger]');
    if (revealEls.length) {
      const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      if (!prefersReduced && 'IntersectionObserver' in window) {
        const io = new IntersectionObserver((entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              entry.target.classList.add('is-revealed');
              io.unobserve(entry.target);
            }
          });
        }, { rootMargin: '0px 0px -10% 0px', threshold: 0.15 });

        revealEls.forEach((el) => {
          if (el.hasAttribute('data-stagger')) {
            const children = Array.from(el.children);
            children.forEach((child, idx) => {
              child.style.setProperty('--stagger-index', idx.toString());
            });
          }
          io.observe(el);
        });
      } else {
        // immediately reveal when motion is reduced or IO unsupported
        revealEls.forEach((el) => el.classList.add('is-revealed'));
      }
    }
  } catch {}

  // ===== Subtle Parallax for hero blobs and header brand =====
  try {
    const parallaxTargets = [
      ...document.querySelectorAll('.blob'),
      ...document.querySelectorAll('.brand__logo')
    ];
    parallaxTargets.forEach((el) => el.classList.add('parallax-layer'));

    let rafId = null;
    const onMove = (evt) => {
      if (rafId) cancelAnimationFrame(rafId);
      const { clientX, clientY } = evt.touches ? evt.touches[0] : evt;
      const cx = window.innerWidth / 2;
      const cy = window.innerHeight / 2;
      const dx = (clientX - cx) / cx; // -1..1
      const dy = (clientY - cy) / cy;
      rafId = requestAnimationFrame(() => {
        parallaxTargets.forEach((el, i) => {
          const depth = (i % 3 + 1) * 3; // 3,6,9 px range
          el.style.setProperty('--px', `${dx * depth}px`);
          el.style.setProperty('--py', `${dy * depth}px`);
        });
      });
    };
    window.addEventListener('mousemove', onMove, { passive: true });
    window.addEventListener('touchmove', onMove, { passive: true });
  } catch {}

  // ===== Enable ripple on primary buttons =====
  try {
    const rippleButtons = document.querySelectorAll('.btn, button');
    rippleButtons.forEach((btn) => {
      btn.classList.add('ripple-enabled');
      btn.addEventListener('click', (e) => {
        const host = e.currentTarget;
        const rect = host.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const ripple = document.createElement('span');
        ripple.className = 'ripple';
        ripple.style.width = ripple.style.height = `${size}px`;
        ripple.style.left = `${e.clientX - rect.left}px`;
        ripple.style.top = `${e.clientY - rect.top}px`;
        host.appendChild(ripple);
        setTimeout(() => ripple.remove(), 600);
      });
    });
  } catch {}

  // ===== Wellness page JS =====
  try {
    // Breathing coach
    const circle = document.getElementById('breathCircle');
    const toggle = document.getElementById('breathToggle');
    const pattern = document.getElementById('breathPattern');
    const label = document.getElementById('breathLabel');
    let running = false;
    function applyPattern(name) {
      if (!circle) return;
      circle.style.animation = 'none';
      // force reflow
      void circle.offsetWidth;
      if (name === 'box') {
        circle.style.animation = 'breatheBox 16s infinite ease-in-out';
        if (label) label.textContent = 'Box: In 4 • Hold 4 • Out 4 • Hold 4';
      } else if (name === '478') {
        circle.style.animation = 'breathe478 12s infinite ease-in-out';
        if (label) label.textContent = '4-7-8: In 4 • Hold 7 • Out 8';
      } else {
        circle.style.animation = 'breatheCalm 6s infinite ease-in-out';
        if (label) label.textContent = 'Calm: In 4 • Out 4';
      }
    }
    if (pattern) {
      applyPattern(pattern.value);
      pattern.addEventListener('change', () => { if (running) applyPattern(pattern.value); });
    }
    if (toggle && circle) {
      toggle.addEventListener('click', () => {
        running = !running;
        if (running) {
          toggle.textContent = 'Pause';
          applyPattern(pattern ? pattern.value : 'calm');
        } else {
          toggle.textContent = 'Start';
          circle.style.animation = 'none';
          if (label) label.textContent = 'Paused';
        }
      });
    }

    // Grounding prompt
    const gStart = document.getElementById('groundingStart');
    const gOut = document.getElementById('groundingPrompt');
    if (gStart && gOut) {
      const prompts = [
        'Look around and name 5 things you can see.',
        'Touch 4 objects and notice their textures.',
        'Listen for 3 different sounds around you.',
        'Identify 2 scents you can smell.',
        'Focus on 1 taste you can notice.'
      ];
      let idx = 0;
      gStart.addEventListener('click', () => {
        gOut.textContent = prompts[idx % prompts.length];
        idx += 1;
      });
    }

    // Mood check tips
    const moodButtons = document.querySelectorAll('#mood [data-mood]');
    const moodTip = document.getElementById('moodTip');
    const moodMap = {
      stressed: 'Try box breathing (4-4-4-4) for 1 minute and unclench your jaw.',
      anxious: 'Use 4-7-8 breathing, then name 3 sounds you can hear right now.',
      low: 'Sit up, roll your shoulders, drink some water, and step into light.',
      angry: 'Inhale through the nose, exhale longer than inhale. Take a short walk.',
      overwhelmed: 'Write one small next step. Do only that. Then reassess.',
    };
    moodButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const m = btn.getAttribute('data-mood');
        if (moodTip && m && moodMap[m]) moodTip.textContent = moodMap[m];
      });
    });

    // 3-minute routine timer
    const rStart = document.getElementById('routineStart');
    const rStop = document.getElementById('routineStop');
    const rStatus = document.getElementById('routineStatus');
    let rTimer = null;
    let rEnd = 0;
    function fmt(ms){ const s = Math.max(0, Math.ceil(ms/1000)); const m = Math.floor(s/60); const r = s%60; return `${m}:${r.toString().padStart(2,'0')}`; }
    function tick(){
      const ms = rEnd - Date.now();
      if (ms <= 0) {
        clearInterval(rTimer); rTimer = null;
        if (rStatus) rStatus.textContent = 'Done — nice work.';
        if (rStop) rStop.disabled = true;
        if (rStart) rStart.disabled = false;
        return;
      }
      if (rStatus) rStatus.textContent = `Time left: ${fmt(ms)}`;
    }
    if (rStart && rStop) {
      rStart.addEventListener('click', () => {
        rEnd = Date.now() + 3*60*1000;
        if (rStatus) rStatus.textContent = 'Time left: 3:00';
        if (rTimer) clearInterval(rTimer);
        rTimer = setInterval(tick, 250);
        rStart.disabled = true; rStop.disabled = false;
      });
      rStop.addEventListener('click', () => {
        if (rTimer) clearInterval(rTimer); rTimer = null;
        if (rStatus) rStatus.textContent = 'Stopped';
        rStart.disabled = false; rStop.disabled = true;
      });
    }

    // Affirmations
    const affirmation = document.getElementById('affirmation');
    const nextAff = document.getElementById('nextAffirmation');
    const lines = [
      'You are doing your best with the tools you have.',
      'This feeling is temporary. You are more than this moment.',
      'Small steps count. Progress, not perfection.',
      'You deserve care and patience from yourself.',
      'Breathe in calm, breathe out tension.',
    ];
    if (affirmation && nextAff) {
      nextAff.addEventListener('click', () => {
        const i = Math.floor(Math.random() * lines.length);
        affirmation.textContent = lines[i];
      });
    }
  } catch {}
});
