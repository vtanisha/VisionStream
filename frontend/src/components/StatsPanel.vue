<script setup>
defineProps({ stats: { type: Object, default: null } })

const ms = (v) => (v == null ? '—' : `${v.toFixed(1)}`)
</script>

<template>
  <div class="card" v-if="stats">
    <p class="eyebrow">Live stats</p>

    <div class="tiles">
      <div class="tile" :class="stats.timings.fps >= 25 ? 'tile--ok' : 'tile--warn'">
        <span class="tile__label">FPS</span>
        <span class="tile__value num">{{ stats.timings.fps.toFixed(1) }}</span>
      </div>
      <div class="tile">
        <span class="tile__label">Objects tracked</span>
        <span class="tile__value num">{{ stats.objects_tracked }}</span>
      </div>
      <div class="tile">
        <span class="tile__label">Unique IDs seen</span>
        <span class="tile__value num">{{ stats.unique_ids }}</span>
      </div>
      <div class="tile">
        <span class="tile__label">Device</span>
        <span class="tile__value num tile__value--sm">{{ stats.device }}</span>
      </div>
    </div>

    <p class="eyebrow eyebrow--sub">Top CLIP prompt</p>
    <div v-if="stats.top_prompt" class="topprompt">
      <p class="topprompt__text">{{ stats.top_prompt.prompt }}</p>
      <div class="bar-row">
        <div class="bar"><div class="bar__fill" :style="{ width: stats.top_prompt.score * 100 + '%' }" /></div>
        <span class="num bar__pct">{{ (stats.top_prompt.score * 100).toFixed(1) }}%</span>
      </div>
    </div>
    <p v-else class="muted">No prompts scored yet.</p>

    <p class="eyebrow eyebrow--sub">All prompt scores</p>
    <ul class="rows">
      <li v-for="s in stats.clip_scores" :key="s.prompt">
        <span class="rows__key">{{ s.prompt }}</span>
        <span class="rows__val num">{{ (s.score * 100).toFixed(1) }}%</span>
      </li>
    </ul>

    <p class="eyebrow eyebrow--sub">Latency · p50 / p95 (ms)</p>
    <table class="timings">
      <tbody>
        <tr v-for="k in ['detect', 'clip', 'encode', 'end_to_end']" :key="k">
          <td class="timings__name">{{ k }}</td>
          <td class="timings__val num">{{ ms(stats.timings[k].p50_ms) }}</td>
          <td class="timings__val num">{{ ms(stats.timings[k].p95_ms) }}</td>
        </tr>
      </tbody>
    </table>
    <p class="note">
      end_to_end = capture → JPEG ready to send. Browser decode + paint not included.
    </p>

    <p class="eyebrow eyebrow--sub">Frame drops</p>
    <div class="drops">
      <span class="drop">capture→detect <b class="num">{{ stats.frames_dropped_capture }}</b></span>
      <span class="drop">detect→CLIP <b class="num">{{ stats.frames_dropped_clip }}</b></span>
    </div>
    <p class="note">
      Drops are the policy working: the queue is depth-1 and overwrites, so a slow stage
      loses frames instead of accumulating latency.
    </p>

    <p class="eyebrow eyebrow--sub">Class counts</p>
    <ul class="rows">
      <li v-for="(count, label) in stats.class_counts" :key="label">
        <span class="rows__key">{{ label }}</span>
        <span class="rows__val num">{{ count }}</span>
      </li>
      <li v-if="!Object.keys(stats.class_counts).length" class="muted">Nothing in frame.</li>
    </ul>
  </div>

  <div class="card card--wait" v-else>
    <p class="muted">Waiting for the backend…</p>
  </div>
</template>

<style scoped>
.card {
  background: var(--color-paper-1);
  border: var(--rule-hair);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-card);
  padding: var(--space-md);
}
.card--wait { color: var(--color-ink-2); }

.eyebrow {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: var(--tracking-label);
  color: var(--color-ink-2);
  margin: 0 0 var(--space-sm);
}
.eyebrow--sub { margin-top: var(--space-lg); }

.tiles { display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-xs); }
.tile {
  background: var(--color-paper-0);
  border: var(--rule-hair);
  border-radius: var(--radius-md);
  padding: var(--space-sm);
  display: flex; flex-direction: column; gap: 0.15rem;
}
/* one accent stays indigo; FPS carries a functional green/amber left border only */
.tile--ok   { border-left: 3px solid var(--color-success); }
.tile--warn { border-left: 3px solid var(--color-warning); }
.tile--ok .tile__value   { color: var(--color-success); }
.tile--warn .tile__value { color: var(--color-warning); }
.tile__label { font-size: var(--text-xs); color: var(--color-ink-2); }
.tile__value { font-size: var(--text-2xl); font-weight: 600; line-height: 1; color: var(--color-ink-0); }
.tile__value--sm { font-size: var(--text-lg); text-transform: uppercase; }

.topprompt { background: var(--color-paper-0); border: var(--rule-hair); border-radius: var(--radius-md); padding: var(--space-sm); }
.topprompt__text { margin: 0 0 var(--space-xs); font-size: var(--text-sm); font-weight: 500; }
.bar-row { display: flex; align-items: center; gap: var(--space-xs); }
.bar { flex: 1; height: 6px; background: var(--color-paper-3); border-radius: var(--radius-pill); overflow: hidden; }
.bar__fill {
  height: 100%; background: var(--color-accent); border-radius: var(--radius-pill);
  transition: width var(--dur-slow) var(--ease-out);
}
.bar__pct { font-size: var(--text-sm); min-width: 3.4rem; text-align: right; color: var(--color-ink-1); }

.rows { list-style: none; padding: 0; margin: 0; }
.rows li {
  display: flex; justify-content: space-between; gap: var(--space-sm);
  font-size: var(--text-sm); padding: 0.35rem 0;
  border-bottom: var(--rule-hair);
}
.rows li:last-child { border-bottom: none; }
.rows__key { color: var(--color-ink-1); }
.rows__val { color: var(--color-ink-2); }

.timings { width: 100%; border-collapse: collapse; font-size: var(--text-sm); }
.timings__name { color: var(--color-ink-1); padding: 0.3rem 0; }
.timings__val { text-align: right; color: var(--color-ink-2); }

.drops { display: flex; flex-wrap: wrap; gap: var(--space-sm); }
.drop {
  background: var(--color-paper-0); border: var(--rule-hair); border-radius: var(--radius-sm);
  padding: 0.25rem 0.6rem; font-size: var(--text-xs); color: var(--color-ink-1);
}
.drop b { color: var(--color-ink-0); margin-left: 0.25rem; }

.muted { color: var(--color-ink-2); font-size: var(--text-sm); margin: 0.2rem 0; }
.note { color: var(--color-ink-2); font-size: var(--text-xs); line-height: 1.5; margin: var(--space-xs) 0 0; }
</style>
