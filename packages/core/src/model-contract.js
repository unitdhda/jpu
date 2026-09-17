const CAT_DIM = 22
const SCRIPTS = ['HIRAGANA', 'KATAKANA', 'KANJI', 'LATIN', 'DIGIT', 'PUNCT', 'SYMBOL', 'WHITESPACE', 'OTHER']
const ATOMS = ['NOUN_LIKE', 'PROPER_LIKE', 'VERB_STEM', 'ADJECTIVE', 'ADVERB', 'PARTICLE', 'AUXILIARY', 'COPULA', 'PREFIX_SUFFIX', 'NUMBER', 'PUNCT_SYMBOL', 'OTHER']
const FUNCTIONS = ['TOPIC', 'SUBJECT_CASE', 'OBJECT', 'LOCATION', 'DIRECTION', 'TIME', 'COMITATIVE', 'QUOTATIVE', 'GENITIVE', 'CONNECTIVE', 'OTHER', 'UNKNOWN']
const INFLECTIONS = ['PAST', 'NEGATIVE', 'POLITE', 'TE_FORM', 'CONDITIONAL', 'VOLITIONAL', 'PASSIVE', 'CAUSATIVE', 'POTENTIAL', 'ASPECTUAL']
const ROLES = ['TOPIC', 'CASED_NOMINAL', 'PREDICATE', 'MODIFIER', 'CONNECTIVE', 'ADVERBIAL', 'PUNCT', 'OTHER', 'UNKNOWN']
const LEVELS = ['a', 'b', 'bunsetsu', 'clause', 'sentence']
const PUNCT_SENTENCE = new Set(['。', '！', '？', '!', '?', '．'])
const PUNCT_OPEN = new Set(['「', '『', '（', '【', '〔', '〈', '《'])
const PUNCT_CLOSE = new Set(['」', '』', '）', '】', '〕', '〉', '》'])
const PUNCT_COMMA = new Set(['、', '，', ','])

function mix64(value) {
  value = BigInt.asUintN(64, value); value ^= value >> 30n
  value = BigInt.asUintN(64, value * 0xBF58476D1CE4E5B9n); value ^= value >> 27n
  value = BigInt.asUintN(64, value * 0x94D049BB133111EBn)
  return BigInt.asUintN(64, value ^ (value >> 31n))
}
function hashBucket(value, buckets) { return Number(mix64(BigInt(value)) & BigInt(buckets - 1)) }
function script(ch) {
  const cp = ch.codePointAt(0)
  if (cp >= 0x3040 && cp <= 0x309f) return 'HIRAGANA'
  if (cp >= 0x30a0 && cp <= 0x30ff) return 'KATAKANA'
  if ((cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf)) return 'KANJI'
  if (/^[A-Za-z]$/.test(ch)) return 'LATIN'
  if (/^[0-9０-９]$/.test(ch)) return 'DIGIT'
  if (/^\s$/.test(ch)) return 'WHITESPACE'
  if (/^[\p{P}]$/u.test(ch)) return 'PUNCT'
  if (/^[\p{S}]$/u.test(ch)) return 'SYMBOL'
  return 'OTHER'
}
function punctClass(ch) {
  if (PUNCT_SENTENCE.has(ch)) return 1
  if (PUNCT_OPEN.has(ch)) return 2
  if (PUNCT_CLOSE.has(ch)) return 3
  if (PUNCT_COMMA.has(ch)) return 4
  return 0
}
function digitClass(ch) {
  const cp = ch.codePointAt(0)
  if (cp >= 0x30 && cp <= 0x39) return 1
  if (cp >= 0xff10 && cp <= 0xff19) return 2
  return script(ch) === 'DIGIT' ? 3 : 0
}

export function encodeText(text, codepointBuckets = 8192, bigramBuckets = 8192) {
  const chars = Array.from(text)
  const codepoints = chars.map((ch) => hashBucket(ch.codePointAt(0), codepointBuckets))
  const bigrams = chars.map((ch, index) => {
    const right = index + 1 < chars.length ? chars[index + 1].codePointAt(0) : 0
    return hashBucket((BigInt(ch.codePointAt(0)) << 21n) ^ BigInt(right) ^ 0x9E3779B9n, bigramBuckets)
  })
  const categorical = new Float32Array(chars.length * CAT_DIM); let previous = null
  chars.forEach((ch, index) => {
    const offset = index * CAT_DIM; const current = script(ch)
    categorical[offset + SCRIPTS.indexOf(current)] = 1
    if (previous !== null && previous !== current) categorical[offset + 9] = 1
    if ('ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ'.includes(ch)) categorical[offset + 10] = 1
    if ('ーｰ'.includes(ch)) categorical[offset + 11] = 1
    if ('々〆ヽヾゝゞー'.includes(ch)) categorical[offset + 12] = 1
    categorical[offset + 13 + punctClass(ch)] = 1; categorical[offset + 18 + digitClass(ch)] = 1
    previous = current
  })
  return { chars, codepoints, bigrams, categorical }
}

function sigmoid(value) { return 1 / (1 + Math.exp(-value)) }
function argmax(data, offset, width) { let best = 0; for (let i = 1; i < width; i++) if (data[offset + i] > data[offset + best]) best = i; return best }
function softmaxConfidence(data, offset, width) {
  let maximum = -Infinity; for (let i = 0; i < width; i++) maximum = Math.max(maximum, data[offset + i])
  let total = 0; for (let i = 0; i < width; i++) total += Math.exp(data[offset + i] - maximum)
  return Math.exp(data[offset + argmax(data, offset, width)] - maximum) / total
}
function partition(start, end, endpoints) {
  const points = [...new Set(endpoints)].filter((point) => point > start && point <= end).sort((a, b) => a - b)
  const result = []; let cursor = start
  points.forEach((point) => { result.push({ start: cursor, end: point }); cursor = point })
  if (cursor !== end) result.push({ start: cursor, end }); return result
}
function nestedBoundaries(predicted, length) {
  const sets = Object.fromEntries(LEVELS.map((level) => [level, new Set((predicted[level] || []).filter((x) => x >= 1 && x <= length))]))
  sets.sentence.add(length)
  ;[['sentence', ['a', 'b', 'bunsetsu', 'clause']], ['clause', ['a', 'b', 'bunsetsu']], ['bunsetsu', ['a', 'b']], ['b', ['a']]].forEach(([high, lowers]) => lowers.forEach((lower) => sets[high].forEach((point) => sets[lower].add(point))))
  return Object.fromEntries(LEVELS.map((level) => [level, [...sets[level]].sort((a, b) => a - b)]))
}
function node(type, start, end, text, children = [], attrs = {}) { return { type, start, end, text: Array.from(text).slice(start, end).join(''), ...attrs, children } }
function composeTree(text, boundaries, atoms, bs, chunks) {
  const atomNode = (span) => node('A', span.start, span.end, text, [], { atom_type: span.atom_type, function: span.function, inflections: span.inflections })
  const bNode = (span) => node('B', span.start, span.end, text, atoms.filter((x) => x.start >= span.start && x.end <= span.end).map(atomNode))
  const chunkNode = (span) => node('BUNSETSU', span.start, span.end, text, bs.filter((x) => x.start >= span.start && x.end <= span.end).map(bNode), { role: span.role })
  const length = Array.from(text).length
  const clauses = partition(0, length, boundaries.clause).map((span) => node('CLAUSE', span.start, span.end, text, chunks.filter((x) => x.start >= span.start && x.end <= span.end).map(chunkNode)))
  return node('SENTENCE', 0, length, text, clauses)
}

export function decodeOutputs(text, tensors, maxLength, threshold = 0.5) {
  const length = Array.from(text).length; const boundary = tensors.boundaries.data
  const predicted = Object.fromEntries(LEVELS.map((level, column) => [level, Array.from({ length }, (_, i) => sigmoid(boundary[i * 5 + column]) >= threshold ? i + 1 : null).filter(Boolean)]))
  const boundaries = nestedBoundaries(predicted, length); const atom = tensors.atom.data; const fn = tensors.function.data
  const infl = tensors.inflection.data; const role = tensors.role.data
  const atoms = partition(0, length, boundaries.a).map((span) => {
    const i = (span.end - 1) * ATOMS.length; const atomType = ATOMS[argmax(atom, i, ATOMS.length)]
    const functionValue = atomType === 'PARTICLE' ? FUNCTIONS[argmax(fn, (span.end - 1) * FUNCTIONS.length, FUNCTIONS.length)] : null
    const inflections = ['VERB_STEM', 'AUXILIARY', 'COPULA', 'ADJECTIVE'].includes(atomType) ? INFLECTIONS.filter((_, index) => sigmoid(infl[(span.end - 1) * INFLECTIONS.length + index]) >= threshold) : []
    return { ...span, atom_type: atomType, function: functionValue, inflections, confidence: softmaxConfidence(atom, i, ATOMS.length) }
  })
  const bs = partition(0, length, boundaries.b).map((span) => ({ ...span }))
  const chunks = partition(0, length, boundaries.bunsetsu).map((span) => ({ ...span, role: ROLES[argmax(role, (span.end - 1) * ROLES.length, ROLES.length)] }))
  return { text, tokens: atoms, b_spans: bs, bunsetsu: chunks, boundaries, tree: composeTree(text, boundaries, atoms, bs, chunks), preview: false, live: true, maxLength }
}
