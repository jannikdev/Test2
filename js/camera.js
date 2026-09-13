/**
 * Camera & Frame Acquisition Manager
 * Handles camera stream, resolution adaptation, reticle coordinate mapping,
 * and zero-copy ArrayBuffer extraction for the Web Worker.
 */

export class CameraManager {
  constructor(videoElement, canvasElement) {
    this.video = videoElement;
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.stream = null;
    this.facingMode = 'environment';
    this.cropCanvas = document.createElement('canvas');
    this.cropCtx = this.cropCanvas.getContext('2d', { willReadFrequently: true });
  }

  async start() {
    if (this.stream) {
      this.stop();
    }

    // Ensure all mobile & desktop browser autoplay prerequisites are set on video element
    this.video.muted = true;
    this.video.defaultMuted = true;
    this.video.playsInline = true;
    this.video.setAttribute('playsinline', 'true');
    this.video.setAttribute('webkit-playsinline', 'true');
    this.video.setAttribute('muted', 'true');
    this.video.setAttribute('autoplay', 'true');

    const constraints = {
      audio: false,
      video: {
        facingMode: this.facingMode === 'environment' ? { ideal: 'environment' } : 'user',
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    };

    let stream = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      console.warn('Primäre Kamera-Constraints fehlgeschlagen, versuche einfachen Fallback...', err);
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      } catch (e) {
        console.error('Kamera-Zugriff verweigert oder nicht verfügbar:', e);
        throw e;
      }
    }

    this.stream = stream;
    this.video.srcObject = stream;

    // Trigger video playback immediately
    try {
      await this.video.play();
    } catch (playErr) {
      console.warn('Autoplay blockiert (User-Interaktion erforderlich):', playErr);
      const onUserAction = () => {
        this.video.play().catch(console.error);
        window.removeEventListener('click', onUserAction);
        window.removeEventListener('touchstart', onUserAction);
        window.removeEventListener('pointerdown', onUserAction);
      };
      window.addEventListener('click', onUserAction, { once: true });
      window.addEventListener('touchstart', onUserAction, { once: true });
      window.addEventListener('pointerdown', onUserAction, { once: true });
    }

    // Wait for video frame decoding with a safety timeout (NEVER deadlock!)
    await new Promise((resolve) => {
      if (this.video.readyState >= 2 && this.video.videoWidth > 0) {
        this.syncCanvasDimensions();
        return resolve();
      }
      let settled = false;
      const onFrame = () => {
        if (settled) return;
        settled = true;
        this.syncCanvasDimensions();
        resolve();
      };
      this.video.addEventListener('loadedmetadata', onFrame, { once: true });
      this.video.addEventListener('loadeddata', onFrame, { once: true });
      this.video.addEventListener('playing', onFrame, { once: true });
      setTimeout(onFrame, 500); // 500ms safety timeout: guarantee progression
    });

    this.syncCanvasDimensions();
    return true;
  }

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
    }
  }

  async flipCamera() {
    this.facingMode = this.facingMode === 'environment' ? 'user' : 'environment';
    return await this.start();
  }

  syncCanvasDimensions() {
    const w = this.video.videoWidth || 640;
    const h = this.video.videoHeight || 480;
    this.canvas.width = w;
    this.canvas.height = h;
  }

  /**
   * Get the reticle bounding box in actual video coordinate space
   * @param {HTMLElement} reticleEl - DOM element of the viewfinder reticle
   * @returns {Object} { x, y, width, height }
   */
  getReticleVideoCoords(reticleEl) {
    const videoRect = this.video.getBoundingClientRect();
    const reticleRect = reticleEl.getBoundingClientRect();

    const vWidth = this.video.videoWidth || 640;
    const vHeight = this.video.videoHeight || 480;

    const scale = Math.max(videoRect.width / vWidth, videoRect.height / vHeight);
    const displayedW = vWidth * scale;
    const displayedH = vHeight * scale;

    const offsetX = (displayedW - videoRect.width) / 2;
    const offsetY = (displayedH - videoRect.height) / 2;

    const relX = reticleRect.left - videoRect.left + offsetX;
    const relY = reticleRect.top - videoRect.top + offsetY;

    const cropX = Math.max(0, Math.floor(relX / scale));
    const cropY = Math.max(0, Math.floor(relY / scale));
    const cropW = Math.min(vWidth - cropX, Math.floor(reticleRect.width / scale));
    const cropH = Math.min(vHeight - cropY, Math.floor(reticleRect.height / scale));

    return { x: cropX, y: cropY, width: cropW, height: cropH };
  }

  /**
   * Extract high-performance cropped frame
   * Returns a 3-channel RGB ArrayBuffer (for transfer to Worker) and ImageData (for spatial color analysis)
   * @param {Object} coords - { x, y, width, height }
   * @param {number} targetSize - Normalization size (224 for DINOv2)
   */
  grabCrop(coords, targetSize = 224) {
    this.cropCanvas.width = targetSize;
    this.cropCanvas.height = targetSize;

    this.cropCtx.drawImage(
      this.video,
      coords.x, coords.y, coords.width, coords.height,
      0, 0, targetSize, targetSize
    );

    const imageData = this.cropCtx.getImageData(0, 0, targetSize, targetSize);
    const rgba = imageData.data;

    // Convert 4-channel RGBA to 3-channel RGB Uint8Array
    const rgb = new Uint8Array(targetSize * targetSize * 3);
    for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
      rgb[j] = rgba[i];
      rgb[j + 1] = rgba[i + 1];
      rgb[j + 2] = rgba[i + 2];
    }

    const thumbDataUrl = this.cropCanvas.toDataURL('image/jpeg', 0.85);

    return {
      buffer: rgb.buffer,
      width: targetSize,
      height: targetSize,
      imageData,
      thumbDataUrl
    };
  }
}
