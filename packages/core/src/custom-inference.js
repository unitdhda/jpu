import { decodeOutputs, encodeText } from './model-contract.js'

const MAX_LENGTH = 128
const MAX_BATCH = 64
const CAT_DIM = 22
const HEAD_WIDTHS = { boundaries: 5, atom: 12, function: 12, inflection: 10, role: 9 }
const MODEL_SIZES = ['50k', '150k']
const ASSET_VERSION = 'custom-wgsl-fp16-multi-20260917'

export const MODEL_OPTIONS = [
  { id: '50k', label: '50k', description: 'smallest / fastest' },
  { id: '150k', label: '150k', description: 'primary' },
]
const STORAGE = 0x80
const COPY_SRC = 0x04
const COPY_DST = 0x08
const MAP_READ = 0x01
const UNIFORM = 0x40
const COMPUTE = 0x04

function url(name) { return `/models/${name}?v=${ASSET_VERSION}` }
async function loadAsset(name) {
  const compressed = await fetch(url(`${name}.gz`))
  if (compressed.ok && compressed.body) {
    if (compressed.headers.get('content-encoding')?.includes('gzip')) return compressed.arrayBuffer()
    if (typeof DecompressionStream === 'function') {
      return new Response(compressed.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()
    }
  }
  const raw = await fetch(url(name))
  if (!raw.ok) throw new Error(`model fetch failed: ${raw.status}`)
  return raw.arrayBuffer()
}

export function buildCustomShader(meta) {
  const t = meta.tensors
  const n = (name) => t[name].offset
  const hidden = meta.hidden
  const codepointDim = t['codepoint_embedding.weight'].shape[1]
  const bigramDim = t['bigram_embedding.weight'].shape[1]
  const categoricalDim = meta.categorical_dim
  const inputDim = codepointDim + bigramDim + categoricalDim
  // Keep one power-of-two workgroup for both published model variants; extra
  // lanes are idle when the hidden width is smaller than the workgroup.
  const workgroupSize = 2 ** Math.ceil(Math.log2(Math.max(hidden, 48)))
  const maxLength = meta.max_length || MAX_LENGTH
  const blocks = [
    blockSource(meta, 0, 'stateA', 'stateB'), blockSource(meta, 1, 'stateB', 'stateA'),
    blockSource(meta, 2, 'stateA', 'stateB'), blockSource(meta, 3, 'stateB', 'stateA'),
    blockSource(meta, 4, 'stateA', 'stateB'),
  ].join('\n')
  const headBias = [n('boundary_head.bias'), n('atom_head.bias'), n('function_head.bias'), n('inflection_head.bias'), n('role_head.bias')]
  const headWeight = [n('boundary_head.weight'), n('atom_head.weight'), n('function_head.weight'), n('inflection_head.weight'), n('role_head.weight')]
  return `
struct Params { length: u32, batch: u32 };
@group(0) @binding(0) var<storage, read> features: array<u32>;
@group(0) @binding(1) var<storage, read> categorical: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> stateA: array<f32>;
@group(0) @binding(4) var<storage, read_write> stateB: array<f32>;
@group(0) @binding(5) var<storage, read_write> projected: array<f32>;
@group(0) @binding(6) var<storage, read_write> scans: array<f32>;
@group(0) @binding(7) var<storage, read_write> logits: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;
var<workgroup> pre: array<f32, ${hidden}>;

fn weight(index: u32) -> f32 {
  let pair = unpack2x16float(weights[index >> 1u]);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}
fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn erf(x: f32) -> f32 {
  let sign = select(-1.0, 1.0, x >= 0.0); let a = abs(x); let q = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * q - 1.453152027) * q + 1.421413741) * q - 0.284496736) * q + 0.254829592) * q) * exp(-a * a);
  return sign * y;
}
fn gelu(x: f32) -> f32 { return 0.5 * x * (1.0 + erf(x / sqrt(2.0))); }

@compute @workgroup_size(${workgroupSize}) fn input_stage(@builtin(workgroup_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let p = gid.x; let b = gid.y; let c = lid.x; if (p >= params.length || b >= params.batch || c >= ${hidden}u) { return; }
  var value = weight(${n('input_projection.bias')}u + c);
  for (var d = 0u; d < ${codepointDim}u; d++) { value += weight(${n('codepoint_embedding.weight')}u + features[(b * ${maxLength}u + p) * 2u] * ${codepointDim}u + d) * weight(${n('input_projection.weight')}u + c * ${inputDim}u + d); }
  for (var d = 0u; d < ${bigramDim}u; d++) { value += weight(${n('bigram_embedding.weight')}u + features[(b * ${maxLength}u + p) * 2u + 1u] * ${bigramDim}u + d) * weight(${n('input_projection.weight')}u + c * ${inputDim}u + ${codepointDim}u + d); }
  for (var d = 0u; d < ${categoricalDim}u; d++) { value += categorical[(b * ${maxLength}u + p) * ${categoricalDim}u + d] * weight(${n('input_projection.weight')}u + c * ${inputDim}u + ${codepointDim + bigramDim}u + d); }
  stateA[(b * ${maxLength}u + p) * ${hidden}u + c] = value;
}
${blocks}

@compute @workgroup_size(${workgroupSize}) fn project_stage(@builtin(workgroup_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let p = gid.x; let b = gid.y; let c = lid.x; if (p >= params.length || b >= params.batch || c >= ${hidden}u) { return; }
  var forward = weight(${n('scan.forward_project.bias')}u + c);
  var backward = weight(${n('scan.backward_project.bias')}u + c);
  for (var k = 0u; k < ${hidden}u; k++) {
    let value = stateB[(b * ${maxLength}u + p) * ${hidden}u + k];
    forward += value * weight(${n('scan.forward_project.weight')}u + c * ${hidden}u + k);
    backward += value * weight(${n('scan.backward_project.weight')}u + c * ${hidden}u + k);
  }
  projected[((b * ${maxLength}u + p) * ${hidden}u + c) * 2u] = forward;
  projected[((b * ${maxLength}u + p) * ${hidden}u + c) * 2u + 1u] = backward;
}

@compute @workgroup_size(${workgroupSize}) fn scan_stage(@builtin(workgroup_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let p = gid.x; let b = gid.y; let c = lid.x; if (p >= params.length || b >= params.batch || c >= ${hidden}u) { return; }
  let forwardDecay = sigmoid(weight(${n('scan.forward_decay')}u + c));
  let backwardDecay = sigmoid(weight(${n('scan.backward_decay')}u + c));
  var forward = 0.0; for (var j = 0u; j <= p; j++) { forward = forwardDecay * forward + projected[((b * ${maxLength}u + j) * ${hidden}u + c) * 2u]; }
  var backward = 0.0; for (var j = params.length; j > p; j--) { backward = backwardDecay * backward + projected[((b * ${maxLength}u + j - 1u) * ${hidden}u + c) * 2u + 1u]; }
  scans[((b * ${maxLength}u + p) * ${hidden}u + c) * 2u] = forward;
  scans[((b * ${maxLength}u + p) * ${hidden}u + c) * 2u + 1u] = backward;
}

@compute @workgroup_size(${workgroupSize}) fn head_stage(@builtin(workgroup_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let p = gid.x; let b = gid.y; let q = lid.x; if (p >= params.length || b >= params.batch || q >= 48u) { return; }
  var bias = 0u; var row = 0u;
  if (q < 5u) { bias = ${headBias[0]}u; row = q; }
  else if (q < 17u) { bias = ${headBias[1]}u; row = q - 5u; }
  else if (q < 29u) { bias = ${headBias[2]}u; row = q - 17u; }
  else if (q < 39u) { bias = ${headBias[3]}u; row = q - 29u; }
  else { bias = ${headBias[4]}u; row = q - 39u; }
  var value = weight(bias + row);
  for (var c = 0u; c < ${hidden}u; c++) {
    let h = scans[((b * ${maxLength}u + p) * ${hidden}u + c) * 2u] + scans[((b * ${maxLength}u + p) * ${hidden}u + c) * 2u + 1u];
    var base = 0u;
    if (q < 5u) { base = ${headWeight[0]}u; }
    else if (q < 17u) { base = ${headWeight[1]}u; }
    else if (q < 29u) { base = ${headWeight[2]}u; }
    else if (q < 39u) { base = ${headWeight[3]}u; }
    else { base = ${headWeight[4]}u; }
    value += h * weight(base + row * ${hidden}u + c);
  }
  logits[(b * ${maxLength}u + p) * 48u + q] = value;
}`
}

function blockSource(meta, i, read, write) {
  const t = meta.tensors; const n = (name) => t[name].offset
  const hidden = meta.hidden
  const maxLength = meta.max_length || MAX_LENGTH
  const workgroupSize = 2 ** Math.ceil(Math.log2(Math.max(hidden, 48)))
  return `
@compute @workgroup_size(${workgroupSize}) fn block${i}(@builtin(workgroup_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let p = gid.x; let b = gid.y; let c = lid.x; if (p >= params.length || b >= params.batch) { return; }
  var value = 0.0;
  if (c < ${hidden}u) {
    value = weight(${n(`convolution.${i}.pointwise.bias`)}u + c);
    for (var k = 0u; k < ${hidden}u; k++) {
      var depth = weight(${n(`convolution.${i}.depthwise.bias`)}u + k);
      for (var q = 0u; q < 3u; q++) {
        let signed = i32(p) + (i32(q) - 1) * ${meta.dilations[i]};
        if (signed >= 0 && signed < i32(params.length)) {
          depth += weight(${n(`convolution.${i}.depthwise.weight`)}u + k * 3u + q) * ${read}[(b * ${maxLength}u + u32(signed)) * ${hidden}u + k];
        }
      }
      value += weight(${n(`convolution.${i}.pointwise.weight`)}u + c * ${hidden}u + k) * depth;
    }
    pre[c] = gelu(value);
  }
  workgroupBarrier();
  if (c >= ${hidden}u) { return; }
  var mean = 0.0; for (var j = 0u; j < ${hidden}u; j++) { mean += pre[j]; } mean /= ${hidden}.0;
  var variance = 0.0; for (var j = 0u; j < ${hidden}u; j++) { let d = pre[j] - mean; variance += d * d; } variance /= ${hidden}.0;
  let normalized = (pre[c] - mean) / sqrt(variance + 0.00001);
  ${write}[(b * ${maxLength}u + p) * ${hidden}u + c] = ${read}[(b * ${maxLength}u + p) * ${hidden}u + c] + weight(${n(`convolution.${i}.norm.weight`)}u + c) * normalized + weight(${n(`convolution.${i}.norm.bias`)}u + c);
}`
}

export function isWebGpuAvailable() { return typeof navigator !== 'undefined' && 'gpu' in navigator }

export class CustomWebGpuLexer {
  constructor(device, pipelines, bindGroup, buffers, meta) {
    this.device = device; this.pipelines = pipelines; this.bindGroup = bindGroup; this.buffers = buffers
    this.meta = meta; this.hidden = meta.hidden; this.maxLength = meta.max_length || MAX_LENGTH
  }
  static async create({ modelSize = '150k' } = {}) {
    if (!MODEL_SIZES.includes(modelSize)) throw new Error(`unsupported model size: ${modelSize}`)
    if (!isWebGpuAvailable()) throw new Error('WebGPU is not available in this browser')
    const prefix = `jpu-${modelSize}-custom`
    const [meta, binary] = await Promise.all([
      fetch(url(`${prefix}.json`)).then((r) => { if (!r.ok) throw new Error(`metadata fetch failed: ${r.status}`); return r.json() }),
      loadAsset(`${prefix}.bin`),
    ])
    if (meta.model_size && meta.model_size !== modelSize) throw new Error(`model metadata mismatch: expected ${modelSize}, got ${meta.model_size}`)
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) throw new Error('WebGPU adapter unavailable')
    const device = await adapter.requestDevice()
    const module = device.createShaderModule({ code: buildCustomShader(meta) })
    // WebGPU does not expose portable serialized pipeline binaries. Eagerly
    // request compilation for every entry point and inspect diagnostics before
    // making the model available to the UI; this moves all shader compilation
    // out of the first inference request.
    if (module.getCompilationInfo) {
      const info = await module.getCompilationInfo()
      const errors = info.messages.filter((message) => message.type === 'error')
      if (errors.length) throw new Error(`WGSL compilation failed: ${errors[0].message}`)
    }
    const layout = device.createBindGroupLayout({ entries: [
      ...[0, 1, 2].map((binding) => ({ binding, visibility: COMPUTE, buffer: { type: 'read-only-storage' } })),
      ...[3, 4, 5, 6, 7].map((binding) => ({ binding, visibility: COMPUTE, buffer: { type: 'storage' } })),
      { binding: 8, visibility: COMPUTE, buffer: { type: 'uniform' } },
    ] })
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] })
    const entries = ['input_stage', 'block0', 'block1', 'block2', 'block3', 'block4', 'project_stage', 'scan_stage', 'head_stage']
    const pipelines = await Promise.all(entries.map((entry) => device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint: entry } })))
    const max = meta.max_length || MAX_LENGTH; const maxBatch = MAX_BATCH; const hidden = meta.hidden
    const make = (size, usage) => device.createBuffer({ size: Math.max(4, size), usage })
    const slots = max * maxBatch
    const buffers = {
      features: make(slots * 2 * 4, STORAGE | COPY_DST),
      categorical: make(slots * meta.categorical_dim * 4, STORAGE | COPY_DST),
      weights: make(binary.byteLength, STORAGE | COPY_DST),
      stateA: make(slots * hidden * 4, STORAGE), stateB: make(slots * hidden * 4, STORAGE),
      projected: make(slots * hidden * 2 * 4, STORAGE), scans: make(slots * hidden * 2 * 4, STORAGE),
      logits: make(slots * 48 * 4, STORAGE | COPY_SRC), readback: make(maxBatch * max * 48 * 4, MAP_READ | COPY_DST),
      params: make(8, UNIFORM | COPY_DST),
    }
    device.queue.writeBuffer(buffers.weights, 0, binary)
    const boundBuffers = ['features', 'categorical', 'weights', 'stateA', 'stateB', 'projected', 'scans', 'logits', 'params']
    const bindGroup = device.createBindGroup({
      layout,
      entries: boundBuffers.map((name, binding) => ({ binding, resource: { buffer: buffers[name] } })),
    })
    const lexer = new CustomWebGpuLexer(device, pipelines, bindGroup, buffers, meta)
    // Submit one real pass so validation and lazy driver work happen during
    // model selection, not during the user's first visible inference.
    await lexer.analyze('日')
    return lexer
  }
  async analyze(text) {
    const started = performance.now(); const encoded = encodeText(text, this.meta.codepoint_buckets, this.meta.bigram_buckets); const featuresMs = performance.now() - started
    const length = encoded.chars.length
    if (length === 0) throw new Error('text must be non-empty')
    if (length > this.maxLength) throw new Error(`Text exceeds ${this.maxLength} codepoints`)
    const { logits, inferenceMs } = await this._runGpu([encoded], length)
    const decodeStart = performance.now()
    const outputs = this._splitLogits(logits, length, 1)
    const result = decodeOutputs(text, outputs, length)
    const decodeMs = performance.now() - decodeStart
    result.timings = { featuresMs, inferenceMs, decodeMs, decodeSamples: 1, totalMs: performance.now() - started }
    return result
  }
  _splitLogits(logits, length, batch) {
    const outputs = {}; let offset = 0
    for (const [name, width] of Object.entries(HEAD_WIDTHS)) {
      const data = new Float32Array(batch * length * width)
      for (let b = 0; b < batch; b++) for (let i = 0; i < length; i++) {
        const source = ((b * length + i) * 48) + offset
        data.set(logits.subarray(source, source + width), (b * length + i) * width)
      }
      outputs[name] = { data, dims: [batch, length, width] }; offset += width
    }
    return outputs
  }
  async _runGpu(encodedBatch, length) {
    const { buffers, device } = this; const batch = encodedBatch.length
    if (batch > MAX_BATCH) throw new Error(`batch must contain 1-${MAX_BATCH} texts`)
    const features = new Uint32Array(batch * this.maxLength * 2)
    const categorical = new Float32Array(batch * this.maxLength * CAT_DIM)
    encodedBatch.forEach((encoded, b) => {
      encoded.codepoints.forEach((value, index) => { features[(b * this.maxLength + index) * 2] = value; features[(b * this.maxLength + index) * 2 + 1] = encoded.bigrams[index] })
      categorical.set(encoded.categorical, b * this.maxLength * CAT_DIM)
    })
    device.queue.writeBuffer(buffers.features, 0, features)
    device.queue.writeBuffer(buffers.categorical, 0, categorical)
    device.queue.writeBuffer(buffers.params, 0, new Uint32Array([length, batch]))
    const inferenceStart = performance.now(); const encoder = device.createCommandEncoder()
    const dispatch = (index) => { const pass = encoder.beginComputePass(); pass.setPipeline(this.pipelines[index]); pass.setBindGroup(0, this.bindGroup); pass.dispatchWorkgroups(length, batch); pass.end() }
    for (let i = 0; i < 9; i++) dispatch(i)
    const rowBytes = length * 48 * 4
    for (let b = 0; b < batch; b++) encoder.copyBufferToBuffer(buffers.logits, b * this.maxLength * 48 * 4, buffers.readback, b * rowBytes, rowBytes)
    device.queue.submit([encoder.finish()])
    const readBytes = batch * rowBytes
    await buffers.readback.mapAsync(MAP_READ, 0, readBytes)
    const logits = new Float32Array(buffers.readback.getMappedRange(0, readBytes).slice(0)); buffers.readback.unmap()
    return { logits, inferenceMs: performance.now() - inferenceStart }
  }
  async analyzeBatch(texts) {
    if (!Array.isArray(texts) || texts.length === 0 || texts.length > MAX_BATCH) throw new Error(`batch must contain 1-${MAX_BATCH} texts`)
    const encodedBatch = texts.map((text) => encodeText(text, this.meta.codepoint_buckets, this.meta.bigram_buckets)); const length = encodedBatch[0].chars.length
    if (length === 0 || length > this.maxLength) throw new Error(`batch length must be 1-${this.maxLength}`)
    if (encodedBatch.some((encoded) => encoded.chars.length !== length)) throw new Error('analyzeBatch requires equal-length texts')
    const { logits, inferenceMs } = await this._runGpu(encodedBatch, length)
    const outputs = this._splitLogits(logits, length, texts.length)
    return texts.map((text, b) => {
      const single = {}; for (const [name, tensor] of Object.entries(outputs)) { const width = tensor.dims[2]; single[name] = { data: tensor.data.slice(b * length * width, (b + 1) * length * width), dims: [1, length, width] } }
      const result = decodeOutputs(text, single, length); result.timings = { inferenceMs, batchSize: texts.length, decodeMs: 0, totalMs: inferenceMs }; return result
    })
  }
}
