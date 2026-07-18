<script setup>
import { onMounted, onUnmounted, ref } from 'vue'
import { api } from './api'
import StatsPanel from './components/StatsPanel.vue'
import PromptEditor from './components/PromptEditor.vue'
import ClassFilter from './components/ClassFilter.vue'

const health = ref(null)
const stats = ref(null)
const videos = ref([])
const classNames = ref({})
const activeClasses = ref(null)
const prompts = ref([])
const everyN = ref(15)

const running = ref(false)
const streamSrc = ref('')
const source = ref('0')
const error = ref('')
const uploading = ref(false)

let poller = null

async function refreshStats() {
  try {
    stats.value = await api.stats()
    running.value = stats.value.running
    // the stream ends server-side when a file finishes; drop the <img> so it doesn't hang
    if (!running.value && streamSrc.value) streamSrc.value = ''
    // ...and re-attach if the backend is already streaming when the page loads or reloads,
    // otherwise a refresh mid-stream shows "no stream" next to live stats.
    if (running.value && !streamSrc.value) streamSrc.value = `/api/stream?t=${Date.now()}`
  } catch (e) {
    error.value = e.message
  }
}

async function start() {
  error.value = ''
  try {
    await api.start(source.value)
    // cache-buster: without it the browser reuses the dead MJPEG response
    streamSrc.value = `/api/stream?t=${Date.now()}`
    running.value = true
  } catch (e) {
    error.value = e.message
  }
}

async function stop() {
  await api.stop()
  streamSrc.value = ''
  running.value = false
}

async function onUpload(evt) {
  const file = evt.target.files?.[0]
  if (!file) return
  uploading.value = true
  error.value = ''
  try {
    const { saved } = await api.upload(file)
    videos.value = (await api.videos()).videos
    source.value = saved
  } catch (e) {
    error.value = e.message
  } finally {
    uploading.value = false
    evt.target.value = ''
  }
}

async function applyPrompts(list) {
  const res = await api.setPrompts(list)
  prompts.value = res.prompts
}

async function applyClasses(list) {
  const res = await api.setClasses(list)
  activeClasses.value = res.active
}

onMounted(async () => {
  try {
    health.value = await api.health()
    videos.value = (await api.videos()).videos
    const c = await api.classes()
    classNames.value = c.names
    activeClasses.value = c.active
    const p = await api.prompts()
    prompts.value = p.prompts
    everyN.value = p.clip_every_n
    if (!health.value.webcam_supported && videos.value.length) source.value = videos.value[0]
  } catch (e) {
    error.value = `backend unreachable: ${e.message}`
  }
  poller = setInterval(refreshStats, 500)
})

onUnmounted(() => clearInterval(poller))
</script>

<template>
  <div class="app">
    <header class="masthead">
      <div class="brand">
        <span class="brand__dot" :class="{ live: running }" aria-hidden="true" />
        <h1>VisionStream</h1>
      </div>
      <p class="tagline">
        Real-time video understanding —
        <span class="pipeline">YOLOv8 · ByteTrack · CLIP zero-shot</span>
      </p>
      <span v-if="health" class="chip" :class="health.device === 'mps' ? 'chip--mps' : 'chip--cpu'">
        <span class="num">{{ health.device }}</span>
      </span>
    </header>

    <div v-if="health && !health.webcam_supported" class="banner">
      <strong>Docker mode</strong> — webcam is unavailable (no host-camera passthrough on
      Apple Silicon) and inference is on CPU. File input only, and slower than native.
    </div>

    <p v-if="error" class="error">{{ error }}</p>

    <main class="layout">
      <section class="stage">
        <div class="controls">
          <select v-model="source" :disabled="running" class="select">
            <option v-if="health?.webcam_supported" value="0">Webcam · device 0</option>
            <option v-for="v in videos" :key="v" :value="v">{{ v }}</option>
          </select>

          <label class="upload" :class="{ busy: uploading }">
            {{ uploading ? 'Uploading…' : 'Upload video' }}
            <input type="file" accept="video/mp4,video/quicktime,video/x-msvideo" @change="onUpload" />
          </label>

          <button v-if="!running" class="btn btn--go" @click="start">Start</button>
          <button v-else class="btn btn--stop" @click="stop">Stop</button>
        </div>

        <!-- the video sits on a dark ink WELL: a bright MJPEG frame would wash out on cream -->
        <div class="well">
          <img v-if="streamSrc" :src="streamSrc" alt="annotated stream" class="well__img" />
          <div v-else class="well__empty">
            <span class="well__mark" aria-hidden="true">◉</span>
            <p class="well__title">No stream running</p>
            <p class="well__hint">Pick a source and press Start.</p>
          </div>
        </div>

        <div class="tools">
          <PromptEditor :model-value="prompts" :every-n="everyN" @apply="applyPrompts" />
          <ClassFilter :names="classNames" :active="activeClasses" @apply="applyClasses" />
        </div>
      </section>

      <aside class="rail">
        <StatsPanel :stats="stats" />
      </aside>
    </main>
  </div>
</template>

<!-- global design tokens + base reset (unscoped, so :root and html/body apply document-wide) -->
<style>
:root {
  color-scheme: light;

  /* Paper / ink — cool-pastel band (hue 258) */
  --color-paper-0:  oklch(98.4% 0.005 258);
  --color-paper-1:  oklch(96.2% 0.010 258);
  --color-paper-2:  oklch(93.0% 0.015 258);
  --color-paper-3:  oklch(89.0% 0.020 258);
  --color-ink-0:    oklch(18.0% 0.030 258);
  --color-ink-1:    oklch(35.0% 0.025 258);
  --color-ink-2:    oklch(52.0% 0.018 258);
  --color-ink-3:    oklch(70.0% 0.012 258);

  /* One accent — indigo. Everything else is neutral or a functional signal. */
  --color-accent:      oklch(54.0% 0.220 268);
  --color-accent-deep: oklch(46.0% 0.220 268);
  --color-accent-soft: oklch(72.0% 0.140 268);
  --color-accent-tint: oklch(94.0% 0.040 268);

  /* Functional signals — used sparingly, never as brand colour */
  --color-live:     oklch(82.0% 0.180 130);
  --color-success:  oklch(60.0% 0.150 145);
  --color-warning:  oklch(64.0% 0.170 55);
  --color-danger:   oklch(58.0% 0.200 25);
  --color-danger-tint:  oklch(94.0% 0.035 25);
  --color-success-tint: oklch(94.0% 0.030 145);
  --color-warning-tint: oklch(94.0% 0.040 60);

  --color-focus:    oklch(46.0% 0.220 268);

  /* Cool graphite video well — the one dark surface */
  --color-well:     oklch(22.0% 0.020 258);
  --color-well-2:   oklch(28.0% 0.022 258);
  --color-on-well:      oklch(94.0% 0.010 258);
  --color-on-well-mute: oklch(70.0% 0.014 258);

  /* Type — Geist + Geist Mono, self-hosted via @fontsource */
  --font-display: "Geist", ui-sans-serif, system-ui, sans-serif;
  --font-body:    "Geist", ui-sans-serif, system-ui, sans-serif;
  --font-mono:    "Geist Mono", ui-monospace, "SF Mono", Menlo, monospace;

  --text-xs:   0.75rem;
  --text-sm:   0.875rem;
  --text-base: 1rem;
  --text-md:   1.125rem;
  --text-lg:   1.375rem;
  --text-xl:   1.75rem;
  --text-2xl:  2.25rem;

  --tracking-tight: -0.02em;
  --tracking-label:  0.08em;

  --lh-tight:  1.1;
  --lh-snug:   1.25;
  --lh-normal: 1.55;

  --space-2xs: 0.25rem;
  --space-xs:  0.5rem;
  --space-sm:  0.75rem;
  --space-md:  1rem;
  --space-lg:  1.5rem;
  --space-xl:  2rem;
  --space-2xl: 3rem;

  --page-max:    82rem;
  --page-gutter: clamp(1rem, 4vw, 2.5rem);

  /* Tight radii — modern, ruler-drawn */
  --radius-sm:   6px;
  --radius-md:   10px;
  --radius-lg:   14px;
  --radius-pill: 999px;

  /* Hairlines do the structural work; shadow is a whisper, not a lift */
  --rule-hair: 1px solid color-mix(in oklch, var(--color-ink-0) 9%, transparent);
  --rule-soft: 1px solid color-mix(in oklch, var(--color-ink-0) 15%, transparent);
  --shadow-card: 0 1px 2px oklch(18% 0.03 258 / 0.06);
  --shadow-well: 0 8px 28px -14px oklch(18% 0.03 258 / 0.30);

  --ease-out: cubic-bezier(0.22, 0.61, 0.36, 1);
  --dur-fast: 140ms;
  --dur-mid:  240ms;
  --dur-slow: 420ms;
}

*, *::before, *::after { box-sizing: border-box; }

html, body {
  margin: 0;
  overflow-x: clip; /* never hidden — gate 34 */
}

body {
  background: var(--color-paper-0);
  color: var(--color-ink-0);
  font-family: var(--font-body);
  font-size: var(--text-base);
  line-height: var(--lh-normal);
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

#app { min-height: 100vh; }

h1, h2, h3 {
  font-family: var(--font-display);
  font-style: normal; /* roman only — no italic headers */
  letter-spacing: var(--tracking-tight);
  line-height: var(--lh-snug);
}

/* every live number is tabular so the panel doesn't jitter as values update */
.mono, .num {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
}

:focus-visible {
  outline: 2px solid var(--color-focus);
  outline-offset: 2px;
  border-radius: 4px;
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.001ms !important;
    transition-duration: 0.001ms !important;
  }
}
</style>

<style scoped>
.app {
  max-width: var(--page-max);
  margin: 0 auto;
  padding: var(--space-lg) var(--page-gutter) var(--space-2xl);
}

/* ---- masthead ---- */
.masthead {
  display: flex;
  align-items: center;
  gap: var(--space-md);
  flex-wrap: wrap;
  padding: var(--space-sm) 0 var(--space-lg);
  border-bottom: var(--rule-hair);
  margin-bottom: var(--space-lg);
}
.brand { display: flex; align-items: center; gap: var(--space-xs); }
.brand__dot {
  width: 10px; height: 10px; border-radius: var(--radius-pill);
  background: var(--color-ink-3); flex: none;
  transition: background var(--dur-mid) var(--ease-out);
}
.brand__dot.live {
  background: var(--color-live);
  box-shadow: 0 0 0 4px color-mix(in oklch, var(--color-live) 30%, transparent);
  animation: pulse 1.8s var(--ease-out) infinite;
}
@keyframes pulse { 50% { box-shadow: 0 0 0 7px color-mix(in oklch, var(--color-live) 0%, transparent); } }
h1 { margin: 0; font-size: var(--text-xl); font-weight: 600; }
.tagline { margin: 0; color: var(--color-ink-2); font-size: var(--text-sm); }
.pipeline {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  letter-spacing: var(--tracking-label);
  color: var(--color-ink-1);
}
.chip {
  margin-left: auto;
  padding: 0.25rem 0.7rem;
  border-radius: var(--radius-sm);
  border: var(--rule-soft);
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: var(--tracking-label);
}
.chip--mps { color: var(--color-accent-deep); border-color: var(--color-accent-soft); background: var(--color-accent-tint); }
.chip--cpu { color: var(--color-ink-2); }

.banner {
  background: var(--color-warning-tint);
  border: 1px solid color-mix(in oklch, var(--color-warning) 45%, transparent);
  color: var(--color-ink-1);
  border-radius: var(--radius-md);
  padding: var(--space-sm) var(--space-md);
  font-size: var(--text-sm);
  margin-bottom: var(--space-md);
}
.banner strong { color: var(--color-ink-0); }

.error {
  background: var(--color-danger-tint);
  border: 1px solid color-mix(in oklch, var(--color-danger) 45%, transparent);
  color: var(--color-danger);
  border-radius: var(--radius-md);
  padding: var(--space-sm) var(--space-md);
  font-size: var(--text-sm);
}

/* ---- layout ---- */
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: var(--space-lg);
  align-items: start;
}
@media (max-width: 980px) { .layout { grid-template-columns: minmax(0, 1fr); } }

/* ---- controls ---- */
.controls { display: flex; gap: var(--space-xs); margin-bottom: var(--space-md); flex-wrap: wrap; }
.select {
  background: var(--color-paper-1);
  border: var(--rule-soft);
  color: var(--color-ink-0);
  border-radius: var(--radius-sm);
  padding: 0.5rem 0.75rem;
  font: inherit;
  font-size: var(--text-sm);
}
.upload {
  display: inline-flex; align-items: center;
  background: var(--color-paper-1);
  border: 1px dashed color-mix(in oklch, var(--color-ink-0) 22%, transparent);
  color: var(--color-ink-1);
  border-radius: var(--radius-sm);
  padding: 0.5rem 0.85rem;
  font-size: var(--text-sm);
  cursor: pointer;
  transition: border-color var(--dur-mid), color var(--dur-mid);
}
.upload:hover { border-color: var(--color-accent); color: var(--color-accent-deep); }
.upload.busy { opacity: 0.55; }
.upload input { display: none; }

.btn {
  font: inherit; font-weight: 600; font-size: var(--text-sm);
  border: 1px solid transparent; border-radius: var(--radius-sm);
  padding: 0.5rem 1.2rem; cursor: pointer;
  transition: background var(--dur-mid) var(--ease-out), border-color var(--dur-mid);
}
.btn--go { background: var(--color-accent); color: oklch(98% 0.01 258); }
.btn--go:hover { background: var(--color-accent-deep); }
.btn--stop {
  background: transparent; color: var(--color-danger);
  border-color: color-mix(in oklch, var(--color-danger) 50%, transparent);
}
.btn--stop:hover { background: var(--color-danger-tint); }

/* ---- video well (the one dark surface) ---- */
.well {
  background: radial-gradient(120% 80% at 50% -10%, var(--color-well-2), var(--color-well));
  border-radius: var(--radius-lg);
  overflow: hidden;
  aspect-ratio: 16 / 9;
  display: flex; align-items: center; justify-content: center;
  box-shadow: var(--shadow-well);
}
.well__img { width: 100%; height: 100%; object-fit: contain; display: block; }
.well__empty { text-align: center; color: var(--color-on-well-mute); }
.well__mark { font-size: 1.75rem; color: var(--color-accent-soft); opacity: 0.7; }
.well__title { margin: var(--space-xs) 0 0.2rem; color: var(--color-on-well); font-weight: 600; }
.well__hint { margin: 0; font-size: var(--text-sm); }

/* ---- tools under the video (fills the left column so it balances the tall stats rail) ---- */
.stage { display: flex; flex-direction: column; }
.tools {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: var(--space-md);
  margin-top: var(--space-md);
}
@media (max-width: 620px) { .tools { grid-template-columns: minmax(0, 1fr); } }

/* ---- rail ---- */
.rail { display: flex; flex-direction: column; gap: var(--space-md); }
</style>
