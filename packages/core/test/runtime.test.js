import assert from 'node:assert/strict'
import test from 'node:test'

import { decodeOutputs, encodeText, MODEL_OPTIONS } from '../src/index.js'

test('exports the promoted model variants', () => {
  assert.deepEqual(MODEL_OPTIONS.map(({ id }) => id), ['50k', '150k'])
})

test('feature hashing matches the Python checkpoint contract', () => {
  const encoded = encodeText('昨日🙂')
  assert.deepEqual(encoded.chars, ['昨', '日', '🙂'])
  assert.deepEqual(encoded.codepoints, [5065, 5880, 5894])
  assert.deepEqual(encoded.bigrams, [5638, 4895, 6199])
  assert.equal(encoded.categorical.length, 3 * 22)
})

test('decoder closes sentence boundaries and preserves nesting', () => {
  const length = 3
  const outputs = {
    boundaries: { data: new Float32Array(length * 5).fill(-20) },
    atom: { data: new Float32Array(length * 12) },
    function: { data: new Float32Array(length * 12) },
    inflection: { data: new Float32Array(length * 10).fill(-20) },
    role: { data: new Float32Array(length * 9) },
  }
  const result = decodeOutputs('映画。', outputs, length)
  assert.deepEqual(result.boundaries.sentence, [3])
  assert.ok(result.boundaries.clause.every((point) => result.boundaries.bunsetsu.includes(point)))
  assert.equal(result.tree.type, 'SENTENCE')
})
