// v2: always fetch fresh copies from the server (no stale page with new data); fall back to the last copy when offline.
const C = "breakouts-v2";
self.addEventListener("install", e => self.skipWaiting());
self.addEventListener("activate", e => e.waitUntil(
  caches.keys().then(ks => Promise.all(ks.filter(k => k !== C).map(k => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener("fetch", e => {
  if (e.request.method !== "GET") return;
  const same = new URL(e.request.url).origin === self.location.origin;
  const key = e.request.url.split("?")[0];
  e.respondWith(
    (same ? fetch(e.request.url, { cache: "no-store" }) : fetch(e.request)).then(r => {
      if (r.ok) { const copy = r.clone(); caches.open(C).then(c => c.put(key, copy)).catch(() => {}); }
      return r;
    }).catch(() => caches.match(key))
  );
});
