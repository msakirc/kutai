const CACHE_NAME = "yazbunu-v2";
const SHELL = ["/", "/static/worker.js", "/static/themes.js", "/static/manifest.json"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))));
  self.clients.claim();
});

self.addEventListener("fetch", e => {
  const u = new URL(e.request.url);
  if (u.pathname.startsWith("/api/") || u.pathname.startsWith("/ws/") || u.pathname === "/health") return;
  e.respondWith(caches.match(e.request).then(c => c || fetch(e.request)));
});
