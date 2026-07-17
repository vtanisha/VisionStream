<script setup>
import { computed, ref } from 'vue'

const props = defineProps({
  names: { type: Object, default: () => ({}) },
  active: { type: Array, default: null },
})
const emit = defineEmits(['apply'])

const query = ref('')
const selected = ref(new Set(props.active || []))

// COCO has 80 classes; a filter box beats an 80-item scroll
const filtered = computed(() => {
  const q = query.value.trim().toLowerCase()
  return Object.entries(props.names)
    .filter(([, name]) => !q || name.includes(q))
    .slice(0, 40)
})

function toggle(id) {
  const n = Number(id)
  selected.value.has(n) ? selected.value.delete(n) : selected.value.add(n)
  selected.value = new Set(selected.value)
}

const applyFilter = () => emit('apply', selected.value.size ? [...selected.value] : null)

function clear() {
  selected.value = new Set()
  emit('apply', null)
}
</script>

<template>
  <div class="card">
    <p class="eyebrow">Class filter</p>
    <p class="note">Nothing selected = all 80 COCO classes.</p>

    <input v-model="query" class="search" placeholder="Search classes…" />

    <div class="chips">
      <button
        v-for="[id, name] in filtered"
        :key="id"
        class="chip"
        :class="{ on: selected.has(Number(id)) }"
        @click="toggle(id)"
      >
        {{ name }}
      </button>
    </div>

    <div class="actions">
      <button class="ghost" @click="clear">Clear</button>
      <button class="apply" @click="applyFilter">
        Apply · <span class="num">{{ selected.size || 'all' }}</span>
      </button>
    </div>
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
.eyebrow {
  font-family: var(--font-mono); font-size: var(--text-xs);
  text-transform: uppercase; letter-spacing: var(--tracking-label);
  color: var(--color-ink-2); margin: 0 0 var(--space-xs);
}
.note { color: var(--color-ink-2); font-size: var(--text-xs); margin: 0 0 var(--space-sm); }

.search {
  width: 100%;
  background: var(--color-paper-0); border: var(--rule-soft);
  color: var(--color-ink-0); border-radius: var(--radius-sm);
  padding: 0.5rem 0.65rem; font: inherit; font-size: var(--text-sm);
  margin-bottom: var(--space-sm);
  transition: border-color var(--dur-mid);
}
.search:focus { outline: none; border-color: var(--color-accent); }

.chips { display: flex; flex-wrap: wrap; gap: var(--space-2xs); max-height: 150px; overflow-y: auto; }
.chip {
  background: var(--color-paper-0); border: var(--rule-soft);
  color: var(--color-ink-1); border-radius: var(--radius-sm);
  padding: 0.2rem 0.6rem; font: inherit; font-size: var(--text-xs);
  cursor: pointer;
  transition: border-color var(--dur-mid), background var(--dur-mid), color var(--dur-mid);
}
.chip:hover { border-color: var(--color-accent-soft); }
.chip.on { background: var(--color-accent-tint); border-color: var(--color-accent); color: var(--color-accent-deep); }

.actions { display: flex; gap: var(--space-xs); margin-top: var(--space-sm); }
.ghost {
  font: inherit; font-size: var(--text-sm);
  background: none; border: var(--rule-soft);
  color: var(--color-ink-1); border-radius: var(--radius-sm);
  padding: 0.45rem 0.9rem; cursor: pointer;
  transition: border-color var(--dur-mid), color var(--dur-mid);
}
.ghost:hover { border-color: var(--color-danger); color: var(--color-danger); }
.apply {
  font: inherit; font-size: var(--text-sm); font-weight: 600;
  background: var(--color-accent); border: none; color: oklch(98% 0.01 258);
  border-radius: var(--radius-sm); padding: 0.45rem 1.1rem; cursor: pointer;
  transition: background var(--dur-mid) var(--ease-out);
}
.apply:hover { background: var(--color-accent-deep); }
</style>
