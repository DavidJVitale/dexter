import test from 'node:test';
import assert from 'node:assert/strict';

import {
  applyMelTransform,
  makeRingBuffer,
  normalizeScoreMap,
  shouldEmitHit,
} from '../openwakeword_core.js';

test('applyMelTransform applies x/10 + 2', () => {
  const input = new Float32Array([0, 10, -10]);
  const out = applyMelTransform(input);
  assert.deepEqual(Array.from(out), [2, 3, 1]);
});

test('ring buffer keeps latest N embeddings', () => {
  const rb = makeRingBuffer(3);
  rb.push(1);
  rb.push(2);
  rb.push(3);
  rb.push(4);
  assert.deepEqual(rb.values(), [2, 3, 4]);
});

test('normalizeScoreMap maps model names to dexter labels', () => {
  const normalized = normalizeScoreMap({
    dexter_start_v1: 0.5,
    dexter_stop_v1: 0.1,
    dexter_abort_v1: 0.9,
  });
  assert.equal(normalized.dexter_start, 0.5);
  assert.equal(normalized.dexter_stop, 0.1);
  assert.equal(normalized.dexter_abort, 0.9);
});

test('shouldEmitHit respects threshold and cooldown', () => {
  const now = 10_000;
  const hit1 = shouldEmitHit({
    label: 'dexter_start',
    score: 0.7,
    threshold: 0.6,
    nowMs: now,
    lastHitMs: 8_500,
    cooldownMs: 1_000,
  });
  assert.equal(hit1, true);

  const hit2 = shouldEmitHit({
    label: 'dexter_start',
    score: 0.7,
    threshold: 0.6,
    nowMs: now,
    lastHitMs: 9_500,
    cooldownMs: 1_000,
  });
  assert.equal(hit2, false);
});
