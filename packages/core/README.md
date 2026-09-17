# jpu

Browser runtime for JPU's direct WebGPU Japanese surface analyzer.

```js
import { CustomWebGpuLexer } from "jpu";

const analyzer = await CustomWebGpuLexer.create({ modelSize: "150k" });
const tree = await analyzer.analyze("昨日は映画を見た。");
```

The host must provide WebGPU and expose this package's `models/` directory at
`/models/`. The demo app does this with a workspace symlink. The runtime has no Sudachi, GiNZA, ONNX Runtime, or dictionary dependency.
