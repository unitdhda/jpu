import { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { CustomWebGpuLexer, isWebGpuAvailable, MODEL_OPTIONS } from 'jpu'

const SAMPLE = '昨日は友達と映画を見に行かなかった。'

const METRIC_INFO = {
  a: 'A F1: how accurately the analyzer places small cuts between characters, such as the cuts between a stem and an ending. F1 balances missed cuts and extra cuts.',
  b: 'B F1: how accurately the analyzer groups those small pieces into larger word-like groups. These are closer to dictionary-sized units.',
  bunsetsu: 'Bunsetsu F1: how accurately the analyzer finds phrase-sized chunks, such as a noun together with its following particle. This is chunking, not a full sentence diagram.',
  atom: 'Atom macro F1: the average accuracy of the broad labels attached to small pieces, such as noun-like, verb stem, particle, or ending. Every label category counts equally.',
  inflection: 'Inflection macro F1: the average accuracy of form flags such as past, negative, polite, or conditional. A piece can have more than one flag.',
  particle: 'Particle F1: how accurately particles are assigned a broad surface function, such as topic, object, location, or companion. This is not a deep semantic role.',
  role: 'Role macro F1: the average accuracy of the job assigned to each phrase-sized chunk, such as topic, case phrase, predicate, or modifier. Every role counts equally.',
}

const METRIC_DETAILS = [
  ['A F1', METRIC_INFO.a],
  ['B F1', METRIC_INFO.b],
  ['Bunsetsu F1', METRIC_INFO.bunsetsu],
  ['Atom macro F1', METRIC_INFO.atom],
  ['Inflection macro F1', METRIC_INFO.inflection],
  ['Particle F1', METRIC_INFO.particle],
  ['Role macro F1', METRIC_INFO.role],
]
const CHART_METRICS = [
  ['a', 'A boundary'], ['b', 'B boundary'], ['bunsetsu', 'Bunsetsu boundary'],
  ['atom', 'Atom labels'], ['inflection', 'Inflection flags'],
  ['particle', 'Particle function'], ['role', 'Bunsetsu role'],
]

function BenchmarkBars({ value }) {
  const [hover, setHover] = useState(null)
  const systems = Object.entries(value.systems)
  return <div className="benchmark-visuals">
    <div className="chart-heading"><h3>Quality by decision category</h3><span className="muted">hover or focus a bar for the exact score</span></div>
    <div className="score-charts">
      {CHART_METRICS.map(([key, label]) => <div className="score-chart" key={key}>
        <strong>{label}</strong>
        {systems.map(([system, scores]) => {
          const score = scores[key]
          if (score == null) return null
          const item = { system, label, score }
          return <button
            type="button"
            className="chart-bar-row"
            key={system}
            aria-label={`${system}, ${label}: ${score.toFixed(1)} percent`}
            title={`${system} · ${label}: ${score.toFixed(1)}%`}
            onMouseEnter={() => setHover(item)}
            onFocus={() => setHover(item)}
            onMouseLeave={() => setHover(null)}
            onBlur={() => setHover(null)}
          >
            <span className="chart-system">{system.replace('jpu · ', '')}</span>
            <span className="chart-track"><span className="chart-bar" style={{ width: `${score}%` }} /></span>
            <span className="chart-value">{score.toFixed(1)}%</span>
          </button>
        })}
      </div>)}
    </div>
    <div className="chart-tooltip" role="status">{hover ? `${hover.system}: ${hover.label} = ${hover.score.toFixed(1)}%. This is the aggregate F1 score for that concrete category; higher is better.` : 'Select a bar to inspect one concrete category and value.'}</div>
  </div>
}

function RuntimeBars({ value }) {
  const [hover, setHover] = useState(null)
  const rows = Object.entries(value.systems).filter(([, scores]) => scores.runtime_ms != null)
  const maximum = Math.max(...rows.map(([, scores]) => scores.runtime_ms), 1)
  return <div className="runtime-chart">
    <div className="chart-heading"><h3>Offline runtime reference</h3><span className="muted">lower is faster; not a cross-backend speed claim</span></div>
    {rows.map(([system, scores]) => <button
      type="button"
      className="chart-bar-row runtime-bar-row"
      key={system}
      aria-label={`${system}, ${scores.runtime_ms.toFixed(1)} milliseconds per sentence, batch ${scores.runtime_batch_size || 'unknown'}`}
      title={`${system} · ${scores.runtime_ms.toFixed(1)} ms/sentence · batch ${scores.runtime_batch_size || 'unknown'}`}
      onMouseEnter={() => setHover({ system, score: scores.runtime_ms, batch: scores.runtime_batch_size })}
      onFocus={() => setHover({ system, score: scores.runtime_ms, batch: scores.runtime_batch_size })}
      onMouseLeave={() => setHover(null)}
      onBlur={() => setHover(null)}
    >
      <span className="chart-system">{system.replace('jpu · ', '')}</span>
      <span className="chart-track"><span className="chart-bar" style={{ width: `${scores.runtime_ms / maximum * 100}%` }} /></span>
      <span className="chart-value">{scores.runtime_ms.toFixed(1)} ms</span>
    </button>)}
    <div className="chart-tooltip" role="status">{hover ? `${hover.system}: ${hover.score.toFixed(1)} ms/sentence at batch ${hover.batch || 'unknown'} (total wall time ÷ all sentences).` : 'Select a runtime bar to inspect its batch size and denominator.'}</div>
  </div>
}

function LiveRuntimeBars({ result }) {
  const [hover, setHover] = useState(null)
  if (!result) return null
  const rows = [
    { label: 'warm batch 1', batch: 1, total: result.single.gpuMs, perSentence: result.single.gpuPerSentenceMs },
    ...result.batches.filter((row) => row.batchSize !== 1).map((row) => ({ label: `warm batch ${row.batchSize}`, batch: row.batchSize, total: row.gpuMs, perSentence: row.gpuPerSentenceMs })),
  ]
  const maximum = Math.max(...rows.map((row) => row.perSentence), 1)
  return <div className="runtime-chart live-runtime-chart">
    <div className="chart-heading"><h3>WebGPU throughput by batch</h3><span className="muted">bars show GPU ms / sentence; hover for total time</span></div>
    {rows.map((row) => <button
      type="button"
      className="chart-bar-row"
      key={row.label}
      aria-label={`${row.label}, ${row.perSentence.toFixed(2)} GPU milliseconds per sentence`}
      title={`${row.label} · ${row.perSentence.toFixed(2)} GPU ms/sentence`}
      onMouseEnter={() => setHover(row)}
      onFocus={() => setHover(row)}
      onMouseLeave={() => setHover(null)}
      onBlur={() => setHover(null)}
    >
      <span className="chart-system">{row.label}</span>
      <span className="chart-track"><span className="chart-bar" style={{ width: `${row.perSentence / maximum * 100}%` }} /></span>
      <span className="chart-value">{row.perSentence.toFixed(2)} ms/s</span>
    </button>)}
    <div className="chart-tooltip" role="status">{hover ? `${hover.label}: ${hover.perSentence.toFixed(2)} GPU ms/sentence from ${hover.total.toFixed(2)} ms total for ${hover.batch} sentences.` : 'Select a WebGPU batch bar to inspect total and per-sentence time.'}</div>
  </div>
}

function markerFor(token) {
  const atom = token.atom_type.replace('_LIKE', '')
  return token.function ? `${atom} · ${token.function}` : atom
}

function Token({ token, index, active, onActivate }) {
  return <button
    className={`inline-token${active ? ' selected' : ''}`}
    onMouseEnter={() => onActivate(index)}
    onFocus={() => onActivate(index)}
    onClick={() => onActivate(index)}
    aria-label={`${token.surface}: ${markerFor(token)}`}
  >
    <span className="token-marker">{markerFor(token)}</span>
    <span className="token-frame"><span className="token-surface">{token.surface}</span></span>
  </button>
}

function TokenDetails({ token }) {
  if (!token) return <p className="muted">Hover over or select a marked span.</p>
  return <div className="token-details">
    <div><strong>{token.surface}</strong> <span className="muted">({token.start}–{token.end})</span></div>
    <div><span className="muted">atom</span> {token.atom_type}</div>
    <div><span className="muted">function</span> {token.function || '—'}</div>
    <div><span className="muted">inflections</span> {token.inflections?.length ? token.inflections.join(', ') : '—'}</div>
    <div><span className="muted">confidence</span> {typeof token.confidence === 'number' ? `${(token.confidence * 100).toFixed(1)}% (atom softmax)` : 'not available'}</div>
  </div>
}

function TreeNode({ node, prefix = '', branch = '', isLast = true }) {
  const children = node.children || []
  const continuation = branch ? `${prefix}${isLast ? '   ' : '│  '}` : prefix
  return <div className="tree-node">
    <div className="tree-line">
      <span className="tree-connector" aria-hidden="true">{prefix}{branch}</span>
      <span className="tree-type">{node.type}</span>
      {node.atom_type && <span className="tree-label"> {node.atom_type}{node.function ? `:${node.function}` : ''}</span>}
      {node.text && <span className="tree-text"> {node.text}</span>}
      {node.role && <span className="tree-role"> [{node.role}]</span>}
      {node.inflections?.length > 0 && <span className="tree-inflections"> ({node.inflections.join(', ')})</span>}
    </div>
    {children.map((child, index) => {
      const childIsLast = index === children.length - 1
      return <TreeNode
        key={`${child.type}-${child.start}-${index}`}
        node={child}
        prefix={continuation}
        branch={childIsLast ? '└─ ' : '├─ '}
        isLast={childIsLast}
      />
    })}
  </div>
}

function Timing({ timing }) {
  if (!timing) return <p className="muted">No live inference measured yet.</p>
  return <dl className="timings">
    <div><dt>model initialization (cold)</dt><dd>{timing.loadMs.toFixed(1)} ms</dd></div>
    <div><dt>feature + model + decode</dt><dd>{timing.totalMs.toFixed(1)} ms</dd></div>
    {typeof timing.featuresMs === 'number' && <div><dt>feature encoding</dt><dd>{timing.featuresMs.toFixed(1)} ms</dd></div>}
    {typeof timing.inferenceMs === 'number' && <div><dt>WebGPU passes + readback</dt><dd>{timing.inferenceMs.toFixed(1)} ms</dd></div>}
    {typeof timing.decodeMs === 'number' && <div><dt>deterministic decode{timing.decodeSamples > 1 ? ` (avg/${timing.decodeSamples})` : ''}</dt><dd>{timing.decodeMs.toFixed(2)} ms</dd></div>}
    <div><dt>total request</dt><dd>{timing.requestMs.toFixed(1)} ms</dd></div>
  </dl>
}

function Benchmark({ benchmark, timing }) {
  if (!benchmark) return <section className="benchmark-section"><div className="section-heading"><h2>Benchmark against teachers</h2></div><p className="muted">Loading benchmark results…</p></section>
  return <section className="benchmark-section">
    <div className="section-heading"><h2>Benchmark against teachers</h2><span className="muted">independent gold test sets</span></div>
    <p className="muted">These charts compare independent-gold quality scores. The offline Python runtime is <strong>total CPU wall time divided by all sentences</strong>; model rows were measured with batch size 64, so it is a throughput reference, not single-sentence latency. The WebGPU chart is a live warm direct-WGSL measurement for the selected model, shown as total GPU time divided by its batch size. CPU and GPU times are not directly comparable across machines or backends.</p>
    <details className="benchmark-explainer" open>
      <summary>What do these labels and runtime numbers mean?</summary>
      <p>F1 is a single 0–100 score that rewards correct answers while penalizing both missed answers and made-up answers. “Macro” means we calculate a separate score for each label category and then average them, so common labels cannot hide poor performance on rare labels.</p>
      <div className="metric-guide">
        {METRIC_DETAILS.map(([label, description]) => <div key={label}><strong>{label}</strong><p>{description}</p></div>)}
      </div>
      <p className="muted">Runtime denominator: offline values are corpus wall time ÷ sentence count, with the batch size shown in each cell. WebGPU values are one warm batch request ÷ batch size; batch 1 measures interactive latency, while batches 16/64 measure throughput. If total batch time stays almost flat as batch size grows, ms/sentence should fall because more sentences were processed.</p>
      <p className="muted">Capacity caveat: 50k and 150k were trained with the same production-corpus recipe, but these results are not yet a controlled size-scaling ablation.</p>
      <div className="benchmark-notes">
        <div className="benchmark-note-card"><strong>GiNZA teacher</strong><p>ja-ginza 5.2.0 contains approximately <strong>7,847,413</strong> learned Thinc parameters. The benchmark disables NER, leaving approximately <strong>7,756,607</strong> active learned parameters: Tok2Vec 7,316,736, parser 435,502, and morphologizer 4,369. Compound splitting and bunsetsu recognition are rule-based.</p></div>
        <div className="benchmark-note-card"><strong>Sudachi teacher</strong><p>Sudachi is dictionary-based Viterbi decoding, not a neural model. It has <strong>0 neural parameters</strong>; dictionary entries and costs are data, not trainable neural weights.</p></div>
      </div>
      <p className="muted">The analyzer works with visible form and phrase structure only. For example, “particle = object” describes how <span lang="ja">を</span> is used in this sentence; it does not claim to recover the speaker’s deeper meaning. A dash means that a reference analyzer does not provide that particular kind of answer.</p>
    </details>
    {Object.entries(benchmark.datasets).map(([dataset, value]) => <div className="benchmark-chart-wrap" key={dataset}>
      <h3>{dataset} <span className="muted">({value.sentences} sentences)</span></h3>
      <BenchmarkBars value={value} />
      <RuntimeBars value={value} />
    </div>)}
    <p className="muted benchmark-note">Model quality charts are independent-gold scores, not browser measurements. The chart bars expose the concrete category and value on hover or keyboard focus.</p>
    <h3>Current direct WGSL/WebGPU run</h3>
    <Timing timing={timing} />
  </section>
}

function BrowserBenchmark({ result, running, status, onRun, available, modelSize }) {
  return <section className="benchmark-section browser-benchmark">
    <div className="section-heading"><h2>jpu WebGPU benchmark · {modelSize}</h2><span className="muted">direct WGSL · same-length sample batch</span></div>
    <p className="muted">This measures the selected {modelSize} model on the browser GPU. Initialization and shader compilation are excluded. “GPU passes + readback” is the total WebGPU work for the batch; the per-sentence number divides that total by the number of sentences. Teacher tools are not run here because they are Python programs with no WebGPU implementation.</p>
    <button className="run-button" onClick={() => void onRun()} disabled={!available || running}>{running ? 'measuring…' : 'measure WebGPU runs'}</button>
    {!available && <p className="muted">WebGPU is unavailable in this browser, so the live jpu speed benchmark cannot run here.</p>}
    {status && <p className={status.startsWith('failed') ? 'error' : 'muted'}>{status}</p>}
    <LiveRuntimeBars result={result} />
  </section>
}

function App() {
  const [text, setText] = useState(SAMPLE)
  const [modelSize, setModelSize] = useState('150k')
  const [modelStatus, setModelStatus] = useState('')
  const [snapshot, setSnapshot] = useState(null)
  const [liveResult, setLiveResult] = useState(null)
  const [active, setActive] = useState(0)
  const [webgpu, setWebgpu] = useState(false)
  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState('')
  const [timing, setTiming] = useState(null)
  const [benchmark, setBenchmark] = useState(null)
  const [browserBenchmark, setBrowserBenchmark] = useState(null)
  const [browserBenchmarkRunning, setBrowserBenchmarkRunning] = useState(false)
  const [browserBenchmarkStatus, setBrowserBenchmarkStatus] = useState('')
  const engines = useRef(new Map())
  const enginePromises = useRef(new Map())

  function getEngine(key) {
    const cached = engines.current.get(key)
    if (cached) return Promise.resolve(cached)
    let promise = enginePromises.current.get(key)
    if (!promise) {
      promise = CustomWebGpuLexer.create({ modelSize: key }).then((engine) => {
        engines.current.set(key, engine)
        return engine
      })
      enginePromises.current.set(key, promise)
      promise.catch(() => { if (enginePromises.current.get(key) === promise) enginePromises.current.delete(key) })
    }
    return promise
  }

  useEffect(() => {
    const available = isWebGpuAvailable()
    setWebgpu(available)
    fetch('/demo-output.json').then((response) => response.ok ? response.json() : null).then(setSnapshot).catch(() => {})
    fetch('/teacher-benchmark.json').then((response) => response.ok ? response.json() : null).then(setBenchmark).catch(() => {})
    // Compile and warm the default model during idle page time. Other model
    // variants are prepared when selected, and remain cached for reuse.
    if (available) {
      setModelStatus('preparing 150k model…')
      void getEngine('150k').then(() => setModelStatus('150k ready')).catch(() => setModelStatus('150k unavailable'))
    }
  }, [])

  function selectModel(nextSize) {
    setModelSize(nextSize)
    setLiveResult(null)
    setTiming(null)
    setBrowserBenchmark(null)
    setBrowserBenchmarkStatus('')
    setRunError('')
    if (!webgpu) return
    if (engines.current.has(nextSize)) {
      setModelStatus(`${nextSize} ready`)
      return
    }
    setModelStatus(`preparing ${nextSize} model…`)
    void getEngine(nextSize).then(() => setModelStatus(`${nextSize} ready`)).catch((error) => {
      setModelStatus(`${nextSize} unavailable`)
      setRunError(error instanceof Error ? error.message : String(error))
    })
  }

  // Editing the input is intentionally non-reactive for the result panel. A
  // result is replaced only after the selected model returns a new analysis.
  const analysis = liveResult || (modelSize === '150k' && text === SAMPLE ? snapshot : null)
  // The compact checkpoint JSON stores offsets, not duplicated surface text.
  // Resolve them here so the visual token frames always contain the source word.
  const analysisChars = Array.from(analysis?.text || '')
  const displayTokens = (analysis?.tokens || []).map((token) => ({
    ...token,
    surface: token.surface ?? analysisChars.slice(token.start, token.end).join(''),
  }))
  const activeToken = displayTokens[active] || displayTokens[0]

  async function runBrowserBenchmark() {
    if (!webgpu || browserBenchmarkRunning) return
    setBrowserBenchmarkRunning(true)
    setBrowserBenchmarkStatus('warming the direct WGSL runtime…')
    try {
      const engine = await getEngine(modelSize)
      setBrowserBenchmarkStatus('measuring warm single requests…')
      const firstStart = performance.now(); const first = await engine.analyze(SAMPLE)
      const firstTotal = performance.now() - firstStart
      const repeats = 3; let singleWall = 0; let singleGpu = 0
      for (let i = 0; i < repeats; i++) {
        const started = performance.now(); const result = await engine.analyze(SAMPLE)
        singleWall += performance.now() - started; singleGpu += result.timings.inferenceMs
      }
      const batches = []
      for (const batchSize of [1, 16, 64]) {
        setBrowserBenchmarkStatus(`measuring warm batch ${batchSize}…`)
        const texts = Array.from({ length: batchSize }, () => SAMPLE)
        await engine.analyzeBatch(texts)
        const started = performance.now(); const results = await engine.analyzeBatch(texts)
        const totalMs = performance.now() - started
        const gpuMs = results[0].timings.inferenceMs
        batches.push({ batchSize, totalMs, perSentenceMs: totalMs / batchSize, gpuMs, gpuPerSentenceMs: gpuMs / batchSize })
      }
      setBrowserBenchmark({ firstTotalMs: firstTotal, firstGpuMs: first.timings.inferenceMs, single: { totalMs: singleWall / repeats, perSentenceMs: singleWall / repeats, gpuMs: singleGpu / repeats, gpuPerSentenceMs: singleGpu / repeats }, batches })
      setBrowserBenchmarkStatus('complete')
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setBrowserBenchmarkStatus(`failed: ${message}`)
      setRunError(`benchmark: ${message}`)
    } finally {
      setBrowserBenchmarkRunning(false)
    }
  }

  async function analyze() {
    if (!text || !webgpu || running) return
    setActive(0)
    setRunError('')
    setRunning(true)
    const requestStart = performance.now()
    const key = modelSize
    let loadMs = 0
    try {
      let engine = engines.current.get(key)
      if (!engine) {
        const loadStart = performance.now()
        setModelStatus(`preparing ${key} model…`)
        engine = await getEngine(key)
        setModelStatus(`${key} ready`)
        loadMs = performance.now() - loadStart
      }
      const result = await engine.analyze(text)
      setLiveResult(result)
      setTiming({ loadMs, ...result.timings, requestMs: performance.now() - requestStart })
    } catch (error) {
      setRunError(error instanceof Error ? error.message : String(error))
      // Keep the last model result visible; an error must not manufacture a
      // new row from heuristics.
      setTiming({ loadMs, totalMs: performance.now() - requestStart, requestMs: performance.now() - requestStart })
    } finally {
      setRunning(false)
    }
  }

  return <main className="page">
    <header className="site-header"><strong>jpu</strong> <span className="muted">experimental Japanese surface analyzer</span></header>

    <section className="intro">
      <h1>How to use this model</h1>
      <p>Enter one Japanese sentence and run the small model in your browser. The labels above each span show its coarse surface category; hover or click a span for its function, inflections, and confidence. The model predicts character-gap boundaries, then a deterministic composer turns those boundaries into the nested section below.</p>
      <p className="muted">This is surface grammar, not translation or semantic-role analysis. Confidence is the normalized probability of the selected atom type. The result panel updates only after a WebGPU inference completes.</p>
    </section>

    <section className="input-section">
      <div className="model-picker">
        <label htmlFor="model-size">WebGPU model</label>
        <select id="model-size" value={modelSize} onChange={(event) => selectModel(event.target.value)} disabled={!webgpu || running}>
          {MODEL_OPTIONS.map((option) => <option key={option.id} value={option.id}>{option.label} — {option.description}</option>)}
        </select>
        <span className="muted">each choice loads its own packed-FP16 weights and precompiled WGSL pipelines</span>
      </div>
      <label htmlFor="sentence">Japanese sentence</label>
      <textarea id="sentence" value={text} onChange={(event) => { setText(event.target.value); setTiming(null) }} rows={2} placeholder="日本語の文を入力" />
      <div className="input-actions">
        <button className="run-button" onClick={analyze} disabled={!webgpu || running || !text}>{running ? 'running…' : 'run inference'}</button>
        <button className="link-button" onClick={() => { setText(SAMPLE); setActive(0); setTiming(null) }}>use sample</button>
        <span className="muted">{webgpu ? `WebGPU available · ${modelStatus || `${modelSize} selected`}` : 'WebGPU unavailable; inference disabled'}</span>
      </div>
      {runError && <p className="error">Live inference failed: {runError}</p>}
    </section>

    <section className="result-section" aria-label="inline model output">
      <div className="section-heading"><h2>Inline model output</h2><span className="muted">{analysis ? (analysis.text === text ? 'current model output' : 'edit text, then run inference') : 'waiting for model output'}</span></div>
      <div className="inline-output" role="list" aria-label="deconstructed tokens">
        {analysis ? displayTokens.map((token, index) => <Token key={`${token.start}-${token.end}`} token={token} index={index} active={active === index} onActivate={setActive} />) : <span className="muted output-placeholder">Run inference to see model output.</span>}
      </div>
      <TokenDetails token={activeToken} />
      {analysis && <div className="counts muted">A boundaries: {analysis.boundaries?.a?.length || 0} · B boundaries: {analysis.boundaries?.b?.length || 0} · bunsetsu: {analysis.bunsetsu?.length || 0}</div>}
    </section>

    <section className="deterministic-section">
      <div className="section-heading"><h2>Deterministic composition</h2><span className="muted">no learned parser</span></div>
      <p className="muted">Predicted boundaries are closed upward so higher-level boundaries always contain lower-level ones. The tree is composed from A spans to B spans, bunsetsu, clauses, and the sentence.</p>
      <div className="tree">{analysis ? <TreeNode node={analysis.tree} /> : <span className="muted">Run inference to compose the tree.</span>}</div>
    </section>

    <Benchmark benchmark={benchmark} timing={timing} />
    <BrowserBenchmark result={browserBenchmark} running={browserBenchmarkRunning} status={browserBenchmarkStatus} onRun={runBrowserBenchmark} available={webgpu} modelSize={modelSize} />

    <footer className="site-footer muted">{modelSize} parameters · Unicode codepoints · packed FP16 weights · no runtime dictionary · experimental project inspired by <a href="https://github.com/vercel-labs/gpu-lexer" target="_blank" rel="noreferrer">Shu Ding's gpu-lexer</a> · <a href="https://github.com/unitdhda/gpu-jpu/blob/main/THIRD_PARTY_NOTICES.md" target="_blank" rel="noreferrer">notices</a></footer>
  </main>
}

createRoot(document.getElementById('root')).render(<App />)
