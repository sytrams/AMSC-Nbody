import { parseFrame as parseFrameBuffer } from "./frame_parser.mjs";
import { wheelPanDelta } from "./camera_controls.mjs";

(() => {
  "use strict";

  const BASE_LOOP_FPS = 20;
  const FRAME_CACHE_LIMIT = 24;

  const elements = {
    canvas: document.getElementById("simulation-canvas"),
    runTitle: document.getElementById("run-title"),
    connectionDot: document.getElementById("connection-dot"),
    modeLabel: document.getElementById("mode-label"),
    modeDetail: document.getElementById("mode-detail"),
    step: document.getElementById("step-value"),
    time: document.getElementById("time-value"),
    particles: document.getElementById("particle-value"),
    frameCount: document.getElementById("frame-value"),
    timeline: document.getElementById("timeline"),
    timelineLabel: document.getElementById("timeline-label"),
    sequenceLabel: document.getElementById("sequence-label"),
    playToggle: document.getElementById("play-toggle"),
    playSymbol: document.getElementById("play-symbol"),
    playbackSpeed: document.getElementById("playback-speed"),
    resetCamera: document.getElementById("reset-camera"),
    emptyState: document.getElementById("empty-state"),
    emptyTitle: document.getElementById("empty-title"),
    emptyDetail: document.getElementById("empty-detail"),
    errorBanner: document.getElementById("error-banner"),
  };

  function parseFrame(buffer) {
    return parseFrameBuffer(buffer);
  }

  function vectorSubtract(left, right) {
    return [left[0] - right[0], left[1] - right[1], left[2] - right[2]];
  }

  function vectorCross(left, right) {
    return [
      left[1] * right[2] - left[2] * right[1],
      left[2] * right[0] - left[0] * right[2],
      left[0] * right[1] - left[1] * right[0],
    ];
  }

  function vectorDot(left, right) {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
  }

  function vectorNormalize(vector) {
    const length = Math.hypot(vector[0], vector[1], vector[2]) || 1;
    return [vector[0] / length, vector[1] / length, vector[2] / length];
  }

  function multiplyMatrices(left, right) {
    const output = new Float32Array(16);
    for (let column = 0; column < 4; column += 1) {
      for (let row = 0; row < 4; row += 1) {
        let value = 0;
        for (let inner = 0; inner < 4; inner += 1) {
          value += left[inner * 4 + row] * right[column * 4 + inner];
        }
        output[column * 4 + row] = value;
      }
    }
    return output;
  }

  function lookAtMatrix(eye, target) {
    const backward = vectorNormalize(vectorSubtract(eye, target));
    let right = vectorNormalize(vectorCross([0, 1, 0], backward));
    if (Math.hypot(...right) < 0.001) {
      right = [1, 0, 0];
    }
    const up = vectorCross(backward, right);
    return new Float32Array([
      right[0], up[0], backward[0], 0,
      right[1], up[1], backward[1], 0,
      right[2], up[2], backward[2], 0,
      -vectorDot(right, eye), -vectorDot(up, eye), -vectorDot(backward, eye), 1,
    ]);
  }

  function orthographicMatrix(left, right, bottom, top, near, far) {
    return new Float32Array([
      2 / (right - left), 0, 0, 0,
      0, 2 / (top - bottom), 0, 0,
      0, 0, -2 / (far - near), 0,
      -(right + left) / (right - left),
      -(top + bottom) / (top - bottom),
      -(far + near) / (far - near),
      1,
    ]);
  }

  class Camera {
    constructor() {
      this.target = [0, 0, 0];
      this.radius = 1;
      this.yaw = 0;
      this.pitch = 0;
      this.zoom = 1;
      this.initialized = false;
    }

    fit(bounds) {
      this.target = [0, 1, 2].map(
        (component) => (bounds.minimum[component] + bounds.maximum[component]) / 2,
      );
      const spans = [0, 1, 2].map(
        (component) => bounds.maximum[component] - bounds.minimum[component],
      );
      this.radius = Math.max(...spans) * 0.55;
      if (!Number.isFinite(this.radius) || this.radius < 1e-12) {
        this.radius = 1;
      }
      this.yaw = 0;
      this.pitch = 0;
      this.zoom = 1;
      this.initialized = true;
    }

    orbit(deltaX, deltaY) {
      this.yaw -= deltaX * 0.006;
      this.pitch = Math.max(-1.48, Math.min(1.48, this.pitch - deltaY * 0.006));
    }

    zoomBy(factor) {
      this.zoom = Math.max(0.02, Math.min(500, this.zoom * factor));
    }

    basis() {
      const cosinePitch = Math.cos(this.pitch);
      const eye = [
        this.target[0] + Math.sin(this.yaw) * cosinePitch * this.radius * 3,
        this.target[1] + Math.sin(this.pitch) * this.radius * 3,
        this.target[2] + Math.cos(this.yaw) * cosinePitch * this.radius * 3,
      ];
      const forward = vectorNormalize(vectorSubtract(this.target, eye));
      const right = vectorNormalize(vectorCross(forward, [0, 1, 0]));
      const up = vectorCross(right, forward);
      return { eye, right, up };
    }

    panPixels(deltaX, deltaY, viewportHeight) {
      const { right, up } = this.basis();
      const scale = (2 * this.radius) / (Math.max(viewportHeight, 1) * this.zoom);
      for (let component = 0; component < 3; component += 1) {
        this.target[component] -= right[component] * deltaX * scale;
        this.target[component] += up[component] * deltaY * scale;
      }
    }

    matrix(width, height) {
      const { eye } = this.basis();
      const aspect = width > 0 && height > 0 ? width / height : 1;
      const extent = this.radius / this.zoom;
      const projection = orthographicMatrix(
        -extent * aspect,
        extent * aspect,
        -extent,
        extent,
        -this.radius * 8,
        this.radius * 8,
      );
      return multiplyMatrices(projection, lookAtMatrix(eye, this.target));
    }
  }

  class ParticleRenderer {
    constructor(canvas) {
      this.canvas = canvas;
      this.gl = canvas.getContext("webgl2", {
        alpha: false,
        antialias: true,
        depth: true,
        powerPreference: "high-performance",
      });
      if (!this.gl) {
        throw new Error("This browser does not provide WebGL 2");
      }
      this.camera = new Camera();
      this.currentBounds = null;
      this.particleCount = 0;
      this.program = this.createProgram();
      this.viewProjectionLocation = this.gl.getUniformLocation(this.program, "uViewProjection");
      this.pixelRatioLocation = this.gl.getUniformLocation(this.program, "uPixelRatio");
      this.vertexArray = this.gl.createVertexArray();
      this.positionBuffer = this.gl.createBuffer();
      this.typeBuffer = this.gl.createBuffer();
      this.gl.bindVertexArray(this.vertexArray);
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
      this.gl.enableVertexAttribArray(0);
      this.gl.vertexAttribPointer(0, 3, this.gl.FLOAT, false, 0, 0);
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.typeBuffer);
      this.gl.enableVertexAttribArray(1);
      this.gl.vertexAttribIPointer(1, 1, this.gl.UNSIGNED_BYTE, 0, 0);
      this.gl.bindVertexArray(null);
      this.gl.enable(this.gl.DEPTH_TEST);
      this.gl.depthFunc(this.gl.LEQUAL);
      this.gl.enable(this.gl.BLEND);
      this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
      this.gl.clearColor(0.019, 0.031, 0.043, 1);
      this.resize();
    }

    compileShader(type, source) {
      const shader = this.gl.createShader(type);
      this.gl.shaderSource(shader, source);
      this.gl.compileShader(shader);
      if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
        const log = this.gl.getShaderInfoLog(shader);
        this.gl.deleteShader(shader);
        throw new Error(`Viewer shader compilation failed: ${log}`);
      }
      return shader;
    }

    createProgram() {
      const vertexSource = `#version 300 es
        precision highp float;
        precision highp int;
        layout(location = 0) in vec3 aPosition;
        layout(location = 1) in uint aType;
        uniform mat4 uViewProjection;
        uniform float uPixelRatio;
        out vec4 vColor;
        void main() {
          gl_Position = uViewProjection * vec4(aPosition, 1.0);
          if (aType == 1u) {
            gl_PointSize = 9.0 * uPixelRatio;
            vColor = vec4(1.0, 0.82, 0.25, 1.0);
          } else if (aType == 2u) {
            gl_PointSize = 7.0 * uPixelRatio;
            vColor = vec4(0.25, 0.60, 1.0, 1.0);
          } else if (aType == 3u) {
            gl_PointSize = 5.0 * uPixelRatio;
            vColor = vec4(0.78, 0.82, 0.88, 1.0);
          } else if (aType == 4u) {
            gl_PointSize = 2.0 * uPixelRatio;
            vColor = vec4(0.72, 0.48, 0.28, 1.0);
          } else {
            gl_PointSize = 4.0 * uPixelRatio;
            vColor = vec4(1.0);
          }
        }
      `;
      const fragmentSource = `#version 300 es
        precision highp float;
        in vec4 vColor;
        out vec4 outColor;
        void main() {
          vec2 centered = gl_PointCoord * 2.0 - 1.0;
          float distanceSquared = dot(centered, centered);
          if (distanceSquared > 1.0) discard;
          float edge = 1.0 - smoothstep(0.72, 1.0, distanceSquared);
          outColor = vec4(vColor.rgb, vColor.a * edge);
        }
      `;
      const vertex = this.compileShader(this.gl.VERTEX_SHADER, vertexSource);
      const fragment = this.compileShader(this.gl.FRAGMENT_SHADER, fragmentSource);
      const program = this.gl.createProgram();
      this.gl.attachShader(program, vertex);
      this.gl.attachShader(program, fragment);
      this.gl.linkProgram(program);
      this.gl.deleteShader(vertex);
      this.gl.deleteShader(fragment);
      if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
        const log = this.gl.getProgramInfoLog(program);
        this.gl.deleteProgram(program);
        throw new Error(`Viewer shader link failed: ${log}`);
      }
      return program;
    }

    resize() {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(1, Math.floor(this.canvas.clientWidth * ratio));
      const height = Math.max(1, Math.floor(this.canvas.clientHeight * ratio));
      if (this.canvas.width !== width || this.canvas.height !== height) {
        this.canvas.width = width;
        this.canvas.height = height;
      }
      this.gl.viewport(0, 0, width, height);
    }

    update(frame, resetCamera) {
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, frame.positions, this.gl.DYNAMIC_DRAW);
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.typeBuffer);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, frame.types, this.gl.STATIC_DRAW);
      this.particleCount = frame.particleCount;
      this.currentBounds = frame.bounds;
      if (resetCamera || !this.camera.initialized) {
        this.camera.fit(frame.bounds);
      }
    }

    resetCamera() {
      if (this.currentBounds) {
        this.camera.fit(this.currentBounds);
      }
    }

    draw() {
      this.resize();
      this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
      if (this.particleCount === 0) {
        return;
      }
      this.gl.useProgram(this.program);
      this.gl.uniformMatrix4fv(
        this.viewProjectionLocation,
        false,
        this.camera.matrix(this.canvas.width, this.canvas.height),
      );
      this.gl.uniform1f(this.pixelRatioLocation, Math.min(window.devicePixelRatio || 1, 2));
      this.gl.bindVertexArray(this.vertexArray);
      this.gl.drawArrays(this.gl.POINTS, 0, this.particleCount);
      this.gl.bindVertexArray(null);
    }
  }

  const state = {
    renderer: null,
    catalog: { runId: "", complete: false, frames: [] },
    currentIndex: -1,
    currentRunId: "",
    loadGeneration: 0,
    loading: false,
    cache: new Map(),
    playing: true,
    speed: 1,
    lastAdvance: 0,
    completionSeen: false,
    pollTimer: null,
    pointer: null,
  };

  function setMode(mode, detail) {
    elements.connectionDot.className = `state-dot ${mode}`;
    if (mode === "live") {
      elements.modeLabel.textContent = "Live";
    } else if (mode === "loop") {
      elements.modeLabel.textContent = state.playing ? "Looping" : "Paused";
    } else {
      elements.modeLabel.textContent = "Connecting";
    }
    elements.modeDetail.textContent = detail;
  }

  function showError(message) {
    elements.errorBanner.textContent = message;
    elements.errorBanner.hidden = false;
  }

  function clearError() {
    elements.errorBanner.hidden = true;
    elements.errorBanner.textContent = "";
  }

  function setEmpty(title, detail) {
    elements.emptyTitle.textContent = title;
    elements.emptyDetail.textContent = detail;
    elements.emptyState.hidden = false;
  }

  function formatCount(value) {
    return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 2 }).format(value);
  }

  function formatSimulationTime(value) {
    if (!Number.isFinite(value)) return "—";
    if (value === 0) return "0 s";
    return `${new Intl.NumberFormat(undefined, { maximumSignificantDigits: 6 }).format(value)} s`;
  }

  function touchCache(name, frame) {
    state.cache.delete(name);
    state.cache.set(name, frame);
    while (state.cache.size > FRAME_CACHE_LIMIT) {
      const oldest = state.cache.keys().next().value;
      state.cache.delete(oldest);
    }
  }

  async function fetchFrame(info) {
    if (state.cache.has(info.name)) {
      const cached = state.cache.get(info.name);
      touchCache(info.name, cached);
      return cached;
    }
    const response = await fetch(`/api/frame/${encodeURIComponent(info.name)}`, { cache: "force-cache" });
    if (!response.ok) {
      throw new Error(`Frame download failed (${response.status})`);
    }
    const parsed = parseFrame(await response.arrayBuffer());
    touchCache(info.name, parsed);
    return parsed;
  }

  function updateFrameUi(frame, info) {
    elements.step.textContent = frame.step;
    elements.time.textContent = formatSimulationTime(frame.simulationTime);
    elements.particles.textContent = formatCount(frame.particleCount);
    elements.sequenceLabel.textContent = `Frame ${info.sequence}`;
    elements.timeline.value = String(state.currentIndex);
    if (!frame.hasTypes) {
      elements.modeDetail.textContent = "Legacy frame · body types unavailable";
    }
  }

  async function selectFrame(index, { resetCamera = false } = {}) {
    const frames = state.catalog.frames;
    if (index < 0 || index >= frames.length) return;
    const info = frames[index];
    state.currentIndex = index;
    elements.timeline.value = String(index);
    const generation = ++state.loadGeneration;
    state.loading = true;
    try {
      const frame = await fetchFrame(info);
      if (generation !== state.loadGeneration) return;
      state.renderer.update(frame, resetCamera);
      updateFrameUi(frame, info);
      clearError();

      if (state.catalog.complete && frames.length > 1) {
        const next = frames[(index + 1) % frames.length];
        if (!state.cache.has(next.name)) {
          fetchFrame(next).catch(() => {});
        }
      }
    } catch (error) {
      showError(error instanceof Error ? error.message : String(error));
    } finally {
      if (generation === state.loadGeneration) {
        state.loading = false;
      }
    }
  }

  function updatePlaybackControls() {
    const canLoop = state.catalog.complete && state.catalog.frames.length > 1;
    elements.playToggle.disabled = !canLoop;
    elements.timeline.disabled = !state.catalog.complete || state.catalog.frames.length === 0;
    elements.playSymbol.textContent = state.playing ? "Ⅱ" : "▶";
    elements.playToggle.setAttribute("aria-label", state.playing ? "Pause playback" : "Play frames");
    elements.timelineLabel.textContent = state.catalog.complete
      ? state.playing ? "Looping completed simulation" : "Completed simulation paused"
      : "Following the latest frame";
  }

  async function pollCatalog() {
    try {
      const response = await fetch("/api/frames", { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`Frame catalog request failed (${response.status})`);
      }
      const catalog = await response.json();
      if (!Array.isArray(catalog.frames)) {
        throw new Error("The frame catalog response is invalid");
      }

      const runChanged = catalog.runId !== state.currentRunId;
      const becameComplete = catalog.complete && (!state.completionSeen || runChanged);
      state.catalog = catalog;
      elements.frameCount.textContent = String(catalog.frames.length);
      elements.timeline.max = String(Math.max(0, catalog.frames.length - 1));

      if (runChanged) {
        state.currentRunId = catalog.runId;
        state.cache.clear();
        state.currentIndex = -1;
        state.loadGeneration += 1;
        state.loading = false;
        state.completionSeen = false;
        elements.runTitle.textContent = catalog.runId || "Waiting for a simulation";
      }

      if (catalog.frames.length === 0) {
        setEmpty("Looking for frames", "Completed .nbsnap files will appear here automatically.");
        setMode("waiting", "Inspecting the frame directory");
        updatePlaybackControls();
        clearError();
        return;
      }

      elements.emptyState.hidden = true;
      if (catalog.complete) {
        state.completionSeen = true;
        setMode("loop", `${catalog.frames.length} frames available`);
        if (becameComplete || state.currentIndex < 0) {
          state.playing = true;
          state.lastAdvance = performance.now();
          await selectFrame(0, { resetCamera: runChanged });
        } else if (state.currentIndex >= catalog.frames.length) {
          await selectFrame(0);
        }
      } else {
        setMode("live", "Following new snapshots");
        const newestIndex = catalog.frames.length - 1;
        if (newestIndex !== state.currentIndex || runChanged) {
          await selectFrame(newestIndex, { resetCamera: runChanged });
        }
      }
      updatePlaybackControls();
      clearError();
    } catch (error) {
      setMode("waiting", "Retrying the viewer server");
      showError(error instanceof Error ? error.message : String(error));
    } finally {
      state.pollTimer = window.setTimeout(pollCatalog, 500);
    }
  }

  function animationFrame(timestamp) {
    state.renderer.draw();
    if (
      state.catalog.complete &&
      state.playing &&
      state.catalog.frames.length > 1 &&
      !state.loading &&
      timestamp - state.lastAdvance >= 1000 / (BASE_LOOP_FPS * state.speed)
    ) {
      state.lastAdvance = timestamp;
      selectFrame((state.currentIndex + 1) % state.catalog.frames.length);
    }
    requestAnimationFrame(animationFrame);
  }

  function installInteractions() {
    const canvas = elements.canvas;
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("pointerdown", (event) => {
      canvas.setPointerCapture(event.pointerId);
      state.pointer = {
        id: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        pan: event.shiftKey || event.button === 1 || event.button === 2,
      };
      canvas.classList.add("dragging");
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!state.pointer || state.pointer.id !== event.pointerId) return;
      const deltaX = event.clientX - state.pointer.x;
      const deltaY = event.clientY - state.pointer.y;
      state.pointer.x = event.clientX;
      state.pointer.y = event.clientY;
      if (state.pointer.pan || event.shiftKey) {
        state.renderer.camera.panPixels(deltaX, deltaY, canvas.clientHeight);
      } else {
        state.renderer.camera.orbit(deltaX, deltaY);
      }
    });
    const releasePointer = (event) => {
      if (state.pointer && state.pointer.id === event.pointerId) {
        state.pointer = null;
        canvas.classList.remove("dragging");
      }
    };
    canvas.addEventListener("pointerup", releasePointer);
    canvas.addEventListener("pointercancel", releasePointer);
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      if (event.ctrlKey || event.metaKey) {
        state.renderer.camera.zoomBy(Math.exp(-event.deltaY * 0.002));
      } else {
        const [deltaX, deltaY] = wheelPanDelta(
          event.deltaX,
          event.deltaY,
          event.deltaMode,
        );
        state.renderer.camera.panPixels(deltaX, deltaY, canvas.clientHeight);
      }
    }, { passive: false });

    window.addEventListener("keydown", (event) => {
      const target = event.target;
      if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement) return;
      const move = Math.max(12, canvas.clientHeight * 0.05);
      if (event.key === "ArrowLeft") {
        state.renderer.camera.panPixels(-move, 0, canvas.clientHeight);
      } else if (event.key === "ArrowRight") {
        state.renderer.camera.panPixels(move, 0, canvas.clientHeight);
      } else if (event.key === "ArrowUp") {
        state.renderer.camera.panPixels(0, -move, canvas.clientHeight);
      } else if (event.key === "ArrowDown") {
        state.renderer.camera.panPixels(0, move, canvas.clientHeight);
      } else if ((event.ctrlKey || event.metaKey) && (event.key === "+" || event.key === "=")) {
        state.renderer.camera.zoomBy(1.2);
      } else if ((event.ctrlKey || event.metaKey) && event.key === "-") {
        state.renderer.camera.zoomBy(1 / 1.2);
      } else if (event.key.toLowerCase() === "r") {
        state.renderer.resetCamera();
      } else if (event.code === "Space" && state.catalog.complete) {
        state.playing = !state.playing;
        state.lastAdvance = performance.now();
        setMode("loop", `${state.catalog.frames.length} frames available`);
        updatePlaybackControls();
      } else {
        return;
      }
      event.preventDefault();
    });

    elements.resetCamera.addEventListener("click", () => state.renderer.resetCamera());
    elements.playToggle.addEventListener("click", () => {
      if (!state.catalog.complete) return;
      state.playing = !state.playing;
      state.lastAdvance = performance.now();
      setMode("loop", `${state.catalog.frames.length} frames available`);
      updatePlaybackControls();
    });
    elements.timeline.addEventListener("input", () => {
      if (!state.catalog.complete) return;
      state.playing = false;
      selectFrame(Number(elements.timeline.value));
      setMode("loop", `${state.catalog.frames.length} frames available`);
      updatePlaybackControls();
    });
    elements.playbackSpeed.addEventListener("change", () => {
      state.speed = Number(elements.playbackSpeed.value) || 1;
      state.lastAdvance = performance.now();
    });
  }

  function start() {
    try {
      state.renderer = new ParticleRenderer(elements.canvas);
      installInteractions();
      updatePlaybackControls();
      pollCatalog();
      requestAnimationFrame(animationFrame);
    } catch (error) {
      setEmpty("Viewer unavailable", error instanceof Error ? error.message : String(error));
      showError(error instanceof Error ? error.message : String(error));
    }
  }

  start();
})();
