export function applyMelTransform(input) {
  const out = new Float32Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    out[i] = input[i] / 10.0 + 2.0;
  }
  return out;
}

export function makeRingBuffer(capacity) {
  if (!Number.isInteger(capacity) || capacity <= 0) {
    throw new Error('capacity must be a positive integer');
  }

  const queue = [];
  return {
    push(value) {
      queue.push(value);
      if (queue.length > capacity) {
        queue.shift();
      }
    },
    values() {
      return queue.slice();
    },
    size() {
      return queue.length;
    },
    clear() {
      queue.length = 0;
    },
  };
}

export function normalizeScoreMap(rawScores) {
  const out = {
    dexter: 0,
  };

  for (const [key, value] of Object.entries(rawScores || {})) {
    if (String(key).toLowerCase().includes('dexter')) {
      out.dexter = Number(value) || 0;
    }
  }

  return out;
}

export function shouldEmitHit({ score, threshold, nowMs, lastHitMs, cooldownMs }) {
  if (score <= threshold) {
    return false;
  }
  const sinceLast = nowMs - (lastHitMs || 0);
  return sinceLast >= cooldownMs;
}
