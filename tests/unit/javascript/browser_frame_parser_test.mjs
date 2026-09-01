import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";

if (process.argv.length !== 3) {
  throw new Error("usage: browser_frame_parser_test.mjs FRAME_PARSER_MODULE");
}

const { parseFrame } = await import(pathToFileURL(process.argv[2]).href);
const encoder = new TextEncoder();

function writeCommonHeader(view, {
  version,
  headerBytes,
  sequence,
  step,
  time,
  sourceParticles,
  particles,
  scalarBytes,
  payloadBytes,
}) {
  new Uint8Array(view.buffer, 0, 8).set(encoder.encode("NBSNAP01"));
  view.setUint32(8, version, true);
  view.setUint32(12, headerBytes, true);
  view.setBigUint64(16, BigInt(sequence), true);
  view.setBigUint64(24, BigInt(step), true);
  view.setFloat64(32, time, true);
  view.setBigUint64(40, BigInt(sourceParticles), true);
  view.setBigUint64(48, BigInt(particles), true);
  view.setUint32(56, scalarBytes, true);
  view.setUint32(60, 3, true);
  view.setBigUint64(64, BigInt(payloadBytes), true);
}

function makeArchivedTypedFrame() {
  const headerBytes = 80;
  const positions = [1, 2, 3, -4, 5, -6];
  const types = [1, 4];
  const payloadBytes = positions.length * 4 + types.length;
  const buffer = new ArrayBuffer(headerBytes + payloadBytes);
  const view = new DataView(buffer);
  writeCommonHeader(view, {
    version: 2,
    headerBytes,
    sequence: 3,
    step: 7,
    time: 1.25,
    sourceParticles: 20,
    particles: 2,
    scalarBytes: 4,
    payloadBytes,
  });
  view.setBigUint64(72, 40n, true);
  positions.forEach((value, index) => {
    view.setFloat32(headerBytes + index * 4, value, true);
  });
  new Uint8Array(buffer, headerBytes + positions.length * 4).set(types);
  return buffer;
}

function makeCurrentTypedFrame() {
  const headerBytes = 128;
  const quantized = [65535, 0, 0, 0, 65535, 65535];
  const types = [1, 4];
  const payloadBytes = quantized.length * 2 + types.length;
  const buffer = new ArrayBuffer(headerBytes + payloadBytes);
  const view = new DataView(buffer);
  writeCommonHeader(view, {
    version: 3,
    headerBytes,
    sequence: 8,
    step: 42,
    time: 0.5,
    sourceParticles: 20,
    particles: 2,
    scalarBytes: 2,
    payloadBytes,
  });
  [-2.5, 3.5, -5].forEach((value, index) => {
    view.setFloat64(72 + index * 8, value, true);
  });
  [1.25, 4.75, 6.125].forEach((value, index) => {
    view.setFloat64(96 + index * 8, value, true);
  });
  view.setBigUint64(120, BigInt(types.length), true);
  quantized.forEach((value, index) => {
    view.setUint16(headerBytes + index * 2, value, true);
  });
  new Uint8Array(buffer, headerBytes + quantized.length * 2).set(types);
  return buffer;
}

const archived = parseFrame(makeArchivedTypedFrame());
assert.equal(archived.version, 2);
assert.equal(archived.headerBytes, 80);
assert.equal(archived.step, "7");
assert.equal(archived.totalSteps, "40");
assert.equal(archived.particleCount, 2);
assert.equal(archived.hasTypes, true);
assert.deepEqual(Array.from(archived.positions), [1, 2, 3, -4, 5, -6]);
assert.deepEqual(Array.from(archived.types), [1, 4]);
assert.deepEqual(archived.bounds, {
  minimum: [-4, 2, -6],
  maximum: [1, 5, 3],
});

const current = parseFrame(makeCurrentTypedFrame());
assert.equal(current.version, 3);
assert.equal(current.headerBytes, 128);
assert.equal(current.hasTypes, true);
assert.deepEqual(Array.from(current.positions), [
  1.25, 3.5, -5, -2.5, 4.75, 6.125,
]);
assert.deepEqual(Array.from(current.types), [1, 4]);
