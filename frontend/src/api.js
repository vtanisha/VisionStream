const j = async (res) => {
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText)
  return res.json()
}

export const api = {
  health: () => fetch('/api/health').then(j),
  stats: () => fetch('/api/stats').then(j),
  videos: () => fetch('/api/videos').then(j),
  classes: () => fetch('/api/classes').then(j),
  prompts: () => fetch('/api/prompts').then(j),

  setPrompts: (prompts) =>
    fetch('/api/prompts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompts }),
    }).then(j),

  setClasses: (classes) =>
    fetch('/api/classes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ classes }),
    }).then(j),

  start: (source) =>
    fetch('/api/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source }),
    }).then(j),

  stop: () => fetch('/api/stop', { method: 'POST' }).then(j),

  upload: (file) => {
    const fd = new FormData()
    fd.append('file', file)
    return fetch('/api/upload', { method: 'POST', body: fd }).then(j)
  },
}
