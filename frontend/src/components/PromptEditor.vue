<script setup>
import { ref, watch } from 'vue'

const props = defineProps({
  modelValue: { type: Array, default: () => [] },
  everyN: { type: Number, default: 15 },
})
const emit = defineEmits(['apply'])

const draft = ref([...props.modelValue])
const dirty = ref(false)
const saving = ref(false)

watch(
  () => props.modelValue,
  (v) => {
    if (!dirty.value) draft.value = [...v]
  },
)

const touch = () => (dirty.value = true)
const add = () => { draft.value.push(''); touch() }
const remove = (i) => { draft.value.splice(i, 1); touch() }

async function apply() {
  saving.value = true
  try {
    await emit('apply', draft.value.filter((p) => p.trim()))
    dirty.value = false
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <div class="card">
    <p class="eyebrow">CLIP prompts</p>
    <p class="note">
      Zero-shot: compared against the frame in CLIP's shared image/text space. Editing them
      changes what the model looks for — no retraining, no restart. Scored every
      <b class="num">{{ everyN }}</b>th frame.
    </p>

    <div v-for="(p, i) in draft" :key="i" class="row">
      <input v-model="draft[i]" @input="touch" class="input" placeholder="a photo of …" />
      <button class="del" title="remove" @click="remove(i)" aria-label="remove prompt">×</button>
    </div>

    <div class="actions">
      <button class="ghost" @click="add">+ Add prompt</button>
      <button class="apply" :disabled="!dirty || saving" @click="apply">
        {{ saving ? 'Applying…' : 'Apply' }}
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
.note { color: var(--color-ink-2); font-size: var(--text-xs); line-height: 1.5; margin: 0 0 var(--space-md); }
.note b { color: var(--color-ink-0); }

.row { display: flex; gap: var(--space-xs); margin-bottom: var(--space-xs); }
.input {
  flex: 1; min-width: 0;
  background: var(--color-paper-0);
  border: var(--rule-soft);
  color: var(--color-ink-0);
  border-radius: var(--radius-sm);
  padding: 0.5rem 0.65rem; font: inherit; font-size: var(--text-sm);
  transition: border-color var(--dur-mid);
}
.input:focus { outline: none; border-color: var(--color-accent); }
.del {
  flex: none; width: 34px;
  background: var(--color-paper-0); border: var(--rule-soft);
  color: var(--color-ink-2); border-radius: var(--radius-sm);
  cursor: pointer; font-size: 1.1rem; line-height: 1;
  transition: border-color var(--dur-mid), color var(--dur-mid);
}
.del:hover { border-color: var(--color-danger); color: var(--color-danger); }

.actions { display: flex; gap: var(--space-xs); margin-top: var(--space-sm); }
.ghost {
  font: inherit; font-size: var(--text-sm);
  background: none; border: 1px dashed color-mix(in oklch, var(--color-ink-0) 22%, transparent);
  color: var(--color-ink-1); border-radius: var(--radius-sm);
  padding: 0.45rem 0.9rem; cursor: pointer;
  transition: border-color var(--dur-mid), color var(--dur-mid);
}
.ghost:hover { border-color: var(--color-accent); color: var(--color-accent-deep); }
.apply {
  font: inherit; font-size: var(--text-sm); font-weight: 600;
  background: var(--color-accent); border: none; color: oklch(98% 0.01 258);
  border-radius: var(--radius-sm); padding: 0.45rem 1.1rem; cursor: pointer;
  transition: background var(--dur-mid) var(--ease-out);
}
.apply:hover:not(:disabled) { background: var(--color-accent-deep); }
.apply:disabled { opacity: 0.4; cursor: default; }
</style>
