const FRAME_MAGIC = "NBSNAP01";
const LEGACY_VERSION = 1;
const VERSION_TWO = 2;
const TYPED_VERSION = 3;
const LEGACY_HEADER_BYTES = 72;
const LEGACY_TYPED_HEADER_BYTES = 80;
const QUANTIZED_HEADER_BYTES = 120;
const TYPED_HEADER_BYTES = 128;
const QUANTIZED_MAXIMUM = 65535;

function readAscii(view, offset, length) {
  let text = "";
  for (let index = 0; index < length; index += 1) {
    text += String.fromCharCode(view.getUint8(offset + index));
  }
  return text;
}

function safeNumber(value, label) {
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`${label} is too large for this browser`);
  }
  return Number(value);
}

function computeBounds(positions) {
  const minimum = [Infinity, Infinity, Infinity];
  const maximum = [-Infinity, -Infinity, -Infinity];
  for (let index = 0; index < positions.length; index += 3) {
    for (let component = 0; component < 3; component += 1) {
      const value = positions[index + component];
      minimum[component] = Math.min(minimum[component], value);
      maximum[component] = Math.max(maximum[component], value);
    }
  }
  return { minimum, maximum };
}

export function parseFrame(buffer) {
  if (buffer.byteLength < LEGACY_HEADER_BYTES) {
    throw new Error("The frame is truncated");
  }
  const view = new DataView(buffer);
  if (readAscii(view, 0, 8) !== FRAME_MAGIC) {
    throw new Error("The frame has an invalid N-body signature");
  }

  const version = view.getUint32(8, true);
  const headerBytes = view.getUint32(12, true);
  const legacy = version === LEGACY_VERSION &&
    headerBytes === LEGACY_HEADER_BYTES;
  const legacyTyped = version === VERSION_TWO &&
    headerBytes === LEGACY_TYPED_HEADER_BYTES;
  const quantized = version === VERSION_TWO &&
    headerBytes === QUANTIZED_HEADER_BYTES;
  const typed = version === TYPED_VERSION &&
    headerBytes === TYPED_HEADER_BYTES;
  if ((!legacy && !legacyTyped && !quantized && !typed) ||
      buffer.byteLength < headerBytes) {
    throw new Error(
      `Unsupported frame format version ${version} with ${headerBytes}-byte header`,
    );
  }

  const sequence = view.getBigUint64(16, true);
  const step = view.getBigUint64(24, true);
  const simulationTime = view.getFloat64(32, true);
  const sourceParticleCount = view.getBigUint64(40, true);
  const particleCountBig = view.getBigUint64(48, true);
  const particleCount = safeNumber(particleCountBig, "Particle count");
  const scalarBytes = view.getUint32(56, true);
  const components = view.getUint32(60, true);
  const payloadBytes = safeNumber(
    view.getBigUint64(64, true),
    "Frame payload",
  );
  if (particleCount <= 0 || components !== 3) {
    throw new Error("The frame particle layout is invalid");
  }
  if (buffer.byteLength !== headerBytes + payloadBytes) {
    throw new Error("The frame length does not match its header");
  }

  const totalSteps = legacyTyped ? view.getBigUint64(72, true) : 0n;
  let bounds = null;
  if (quantized || typed) {
    bounds = {
      minimum: [
        view.getFloat64(72, true),
        view.getFloat64(80, true),
        view.getFloat64(88, true),
      ],
      maximum: [
        view.getFloat64(96, true),
        view.getFloat64(104, true),
        view.getFloat64(112, true),
      ],
    };
  }
  const particleTypeBytes = typed
    ? safeNumber(view.getBigUint64(120, true), "Particle type payload")
    : legacyTyped
      ? particleCount
      : 0;

  const valueCount = particleCount * 3;
  const expectedPositionBytes = valueCount * scalarBytes;
  if (expectedPositionBytes + particleTypeBytes !== payloadBytes) {
    throw new Error("The frame payload sections are inconsistent");
  }

  const positions = new Float32Array(valueCount);
  if (legacy || legacyTyped) {
    if (scalarBytes !== 4) {
      throw new Error("The float frame scalar width is invalid");
    }
    for (let index = 0; index < valueCount; index += 1) {
      positions[index] = view.getFloat32(headerBytes + index * 4, true);
    }
    bounds = computeBounds(positions);
  } else {
    if (scalarBytes !== 2 || !bounds) {
      throw new Error("The quantized frame scalar width is invalid");
    }
    for (let index = 0; index < valueCount; index += 1) {
      const component = index % 3;
      const value = view.getUint16(headerBytes + index * 2, true);
      const minimum = bounds.minimum[component];
      const range = bounds.maximum[component] - minimum;
      positions[index] = range === 0
        ? minimum
        : minimum + (range * value) / QUANTIZED_MAXIMUM;
    }
  }

  const types = new Uint8Array(particleCount);
  if (particleTypeBytes > 0) {
    if (particleTypeBytes !== particleCount) {
      throw new Error("The particle type count is invalid");
    }
    const typeOffset = headerBytes + expectedPositionBytes;
    types.set(new Uint8Array(buffer, typeOffset, particleCount));
    for (const type of types) {
      if (type > 4) {
        throw new Error(`Unsupported particle type ${type}`);
      }
    }
  }

  return {
    version,
    headerBytes,
    sequence: sequence.toString(),
    step: step.toString(),
    totalSteps: totalSteps.toString(),
    simulationTime,
    particleCount,
    sourceParticleCount: sourceParticleCount.toString(),
    positions,
    types,
    bounds,
    hasTypes: particleTypeBytes === particleCount,
  };
}
