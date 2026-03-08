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
    dexter_start: 0,
    dexter_stop: 0,
    dexter_abort: 0,
  };

  for (const [key, value] of Object.entries(rawScores || {})) {
    const lower = key.toLowerCase();
    if (lower.includes('start')) {
      out.dexter_start = Number(value) || 0;
    } else if (lower.includes('stop')) {
      out.dexter_stop = Number(value) || 0;
    } else if (lower.includes('abort')) {
      out.dexter_abort = Number(value) || 0;
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
