import { createApp } from 'vue'

// self-hosted so the Docker/offline path keeps its type instead of falling back
import '@fontsource/geist-sans/400.css'
import '@fontsource/geist-sans/500.css'
import '@fontsource/geist-sans/600.css'
import '@fontsource/geist-sans/700.css'
import '@fontsource/geist-mono/400.css'
import '@fontsource/geist-mono/500.css'

// global tokens + base styles now live in App.vue's <style> block (no standalone .css)
import App from './App.vue'

createApp(App).mount('#app')
