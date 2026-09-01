import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";

if (process.argv.length !== 3) {
  throw new Error("usage: node test.mjs CAMERA_CONTROLS_MODULE");
}

const { wheelPanDelta } = await import(pathToFileURL(process.argv[2]).href);

assert.deepEqual(
  wheelPanDelta(18, 0, 0),
  [-18, 0],
  "a horizontal-only trackpad gesture must pan horizontally",
);
assert.deepEqual(
  wheelPanDelta(0, -7, 0),
  [0, 7],
  "vertical pixel deltas must preserve the existing pan direction",
);
assert.deepEqual(
  wheelPanDelta(2, -3, 1),
  [-24, 36],
  "line-based wheel deltas must scale both axes equally",
);

console.log("browser camera control tests passed");
