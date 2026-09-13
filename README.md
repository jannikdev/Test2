# VisionID — Schnelle visuelle Objekterkennung

Ein fokussiertes, extrem schnelles System zur optischen Objekterkennung im Web- und Mobilbrowser. Objekte können in wenigen Sekunden mit der Kamera angelernt werden und werden bei erneuter Kameravorhaltung in Echtzeit wiedererkannt.

100% statisch, clientseitig und direkt über **GitHub Pages** lauffähig.

---

## Kernmerkmale

- **Sub-25 MB Vision Backbone:** Nutzt DINOv2 (`Xenova/dinov2-small` quantized) für präzise Form- und Geometrierepräsentation.
- **Räumlicher Lab-Farbdeskriptor:** 45-dimensionales Farb- und Texturhistogramm verhindert Verwechslungen von Objekten gleicher Form mit unterschiedlicher Farbgebung.
- **Duale Umriss-Engine mit Vergleichs-Toggle:**
  - **Schnell (60 fps):** 0 MB Download, ~5 ms Latenz. Adaptive Schwellenwert- und Gradientenberechnung direkt auf dem Canvas.
  - **KI-Maske:** Tiefensensitive neuronale Segmentierung via MobileViT (`deeplabv3-mobilevit-xx-small`, ~2.4 MB).
- **Multi-Shot Registrierung:** 3–5 Blickwinkel pro Objekt für zuverlässige Erkennung aus verschiedenen Perspektiven.
- **Zero-Copy Pipeline:** Bildübertragung an den Web Worker via native `ImageBitmap` (Transferable Objects) – keine Garbage-Collection-Ruckler.
- **Nüchternes, professionelles Design:** Minimalistisches Kamera-HUD im Werkzeug-Stil ohne Sci-Fi- oder Marketing-Buzzwords.
- **PWA & Offline-First:** Ausgestattet mit Service Worker und lokalem IndexedDB-Speicher für uneingeschränkten Offline-Betrieb.

---

## Projektstruktur

```
├── index.html              # App-Shell & Kamera-HUD
├── manifest.json           # PWA Manifest für Home-Screen-Installation
├── sw.js                   # Service Worker für Offline-Caching
├── css/
│   └── app.css             # Minimalistisches Design-System (Utility-First, Dark)
└── js/
    ├── app.js              # Anwendungs-Koordinator & UI-Bindings
    ├── camera.js           # Kamera-Stream & Zero-Copy Frame-Grabbing
    ├── contour-fast.js     # Schneller 60fps Kontur- & Kantendetektor (Modus 1)
    ├── matcher.js          # Hybrid-Vektormatcher & räumlicher Farbdeskriptor
    ├── db.js               # IndexedDB Vault für Objekte, Vektoren & Thumbnails
    └── worker.js           # Background Worker für DINOv2 & MobileViT
```

---

## Bereitstellung auf GitHub Pages

1. Repository auf GitHub pushen:
   ```bash
   git add .
   git commit -m "Implement VisionID instant object recognition engine"
   git push origin main
   ```
2. Im GitHub-Repository auf **Settings** -> **Pages** gehen.
3. Unter **Build and deployment** als Source **Deploy from a branch** und Branch `main` mit Folder `/ (root)` auswählen.
4. Nach wenigen Minuten ist die App unter `https://<username>.github.io/<repo>/` erreichbar.