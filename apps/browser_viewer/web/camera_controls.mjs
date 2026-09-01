const DOM_DELTA_PIXEL = 0;
const NON_PIXEL_DELTA_SCALE = 12;

export function wheelPanDelta(deltaX, deltaY, deltaMode) {
  const gestureScale = deltaMode === DOM_DELTA_PIXEL ? 1 : NON_PIXEL_DELTA_SCALE;
  return [
    deltaX === 0 ? 0 : -deltaX * gestureScale,
    deltaY === 0 ? 0 : -deltaY * gestureScale,
  ];
}
