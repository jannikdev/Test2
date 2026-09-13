/**
 * Camera & Frame Acquisition Manager
 * Handles camera stream, resolution adaptation, reticle coordinate mapping,
 * and zero-copy ImageBitmap extraction for the Web Worker.
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

    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: this.facingMode },
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.video.srcObject = this.stream;
      await new Promise((resolve) => {
        this.video.onloadedmetadata = () => {
          this.syncCanvasDimensions();
          resolve();
        };
      });
      await this.video.play();
      return true;
    } catch (err) {
      console.warn('Primäre Kamera-Auflösung fehlgeschlagen, versuche Fallback...', err);
      // Fallback: minimal constraints
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        this.video.srcObject = this.stream;
        await this.video.play();
        this.syncCanvasDimensions();
        return true;
      } catch (e) {
        console.error('Kamera-Zugriff verweigert oder nicht verfügbar:', e);
        throw e;
      }
    }
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

    // Determine scale and letterboxing of object-fit: cover
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
   * Returns an ImageBitmap (for transfer to Worker) and ImageData (for spatial color analysis)
   * @param {Object} coords - { x, y, width, height }
   * @param {number} targetSize - Normalization size (e.g. 224 for DINOv2)
   */
  async grabCrop(coords, targetSize = 224) {
    this.cropCanvas.width = targetSize;
    this.cropCanvas.height = targetSize;

    this.cropCtx.drawImage(
      this.video,
      coords.x, coords.y, coords.width, coords.height,
      0, 0, targetSize, targetSize
    );

    const imageData = this.cropCtx.getImageData(0, 0, targetSize, targetSize);
    
    // Create zero-copy Transferable ImageBitmap for worker
    const imageBitmap = await createImageBitmap(this.cropCanvas);

    // Also small thumbnail for storage
    const thumbDataUrl = this.cropCanvas.toDataURL('image/jpeg', 0.85);

    return {
      imageBitmap,
      imageData,
      thumbDataUrl
    };
  }
}
