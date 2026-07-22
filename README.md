# schism-dsp

Swift reference implementations of the host-side DSP frontends **and
full-track pipeline orchestration** for the
[schism-audio](https://huggingface.co/schism-audio) Core ML models, built on
Accelerate. Companion to [schism-mlx](https://github.com/schism-audio/schism-mlx).

The Core ML `.mlpackage`s in the org contain the networks only; the host is
responsible for feature extraction, resynthesis, and full-track chunking.
Every routine here is a transliteration of the verified references in
schism-mlx and is tested against the `test_vectors_*.npz` shipped in each
model repo — an implementation passing those vectors is interchangeable with
the pipeline the Core ML models were verified against.

Two modules:

- **`SchismDSP`** — the transforms: fbank, logmel, STFT/iSTFT, model-specific
  framing and spectrogram layouts.
- **`SchismPipeline`** — the orchestration around the model call: demucs
  segmented separation, RoFormer chunked separation, bag combination, AST
  long-audio windowing and score aggregation. The model itself is a closure
  you supply (Core ML predict + `SchismDSP` transforms).

| Module.routine | For | Vector source |
|---|---|---|
| `KaldiFbank.fbank` / `.astFeatures` | [ast-audioset-10-10-coreml](https://huggingface.co/schism-audio/ast-audioset-10-10-coreml) | `test_vectors_fbank.npz` |
| `LogMel.compute` | [cnn14-audioset-coreml](https://huggingface.co/schism-audio/cnn14-audioset-coreml) | `test_vectors_logmel.npz` |
| `STFT.forward/inverse` (normalized) + `Demucs.spec/ispec` framing | [htdemucs-coreml](https://huggingface.co/schism-audio/htdemucs-coreml) (+ ft, 6s) | `test_vectors_stft.npz` |
| `STFT.forward/inverse` (unnormalized) + `Roformer.merge/unmerge/applyMask` | [mini-bs-roformer-18m-coreml](https://huggingface.co/schism-audio/mini-bs-roformer-18m-coreml), [v2](https://huggingface.co/schism-audio/mini-bs-roformer-v2-coreml) | `test_vectors_stft.npz` |
| `Separation.demucs` / `.bagCombine` | htdemucs-coreml (+ ft, 6s) | `test_vectors_pipeline.npz` |
| `Separation.roformer` | mini-bs-roformer-{18m,v2}-coreml | `test_vectors_pipeline.npz` |
| `Classification.astWindows` / `.aggregateMax/Mean` | ast-audioset-10-10-coreml | `test_vectors_pipeline.npz` |

```swift
.package(url: "https://github.com/schism-audio/schism-dsp", branch: "main")
```

## Example: CNN14 input

```swift
import SchismDSP

let (frames, logmel) = LogMel.compute(samples32k)  // (frames, 64) row-major
// -> feed a (1, 1001, 64) window into Cnn14_fp16.mlpackage
```

## Example: one Mini-BS-RoFormer chunk

```swift
let zs = channels.map { STFT.forward($0, nFFT: 2048, hopLength: 512, normalized: false) }
let spec = Roformer.merge(zs)                       // (690, 4100) model input
// mask = model.predict(spec)                        // (1, 4, 690, 4100)
let stem = Roformer.unmerge(maskStem, freqs: 1025, channels: 2, frames: 690)
let audio = STFT.inverse(
    Roformer.applyMask(zs[0], mask: stem[0]),
    hopLength: 512, length: chunkLength, normalized: false
)
```

(The V2 model analyzes at n_fft 4096 but masks a second n_fft-2048 STFT —
see its model card for the exact contract.)

## Example: full-track separation

`SchismPipeline` handles the chunk grid, blending windows, and fold
normalization; the closure runs one model chunk (the example above):

```swift
import SchismPipeline

let stems = try Separation.roformer(
    mix: channels,                       // [2][length] at 44.1 kHz
    sources: 4,
    chunkSize: Separation.roformerChunkSize    // 352800 (8 s)
) { chunk in
    try separateOneChunk(chunk)          // [2][352800] -> [4][2][352800]
}
// stems: [4][2][length], bass/drums/other/vocals (RoFormer stem order)
```

`Separation.demucs` is the HTDemucs equivalent (`apply_model(shifts=0,
split=True)` semantics: centered padding pulled from the surrounding track,
triangular blending). For htdemucs_ft, run `Separation.demucs` once per
core over the full track and merge the four estimates with
`Separation.bagCombine(estimates, weights:)` (one-hot weights — stem *k*
from core *k*).

## Example: AST on long audio

```swift
let windows = Classification.astWindowedFeatures(samples16k)  // [(1024*128)]
let scores = try windows.map { window in
    Classification.sigmoid(try predictLogits(window))   // (527,) each
}
let perClass = Classification.aggregateMax(scores)  // "did it occur anywhere?"
```

## Testing

```sh
swift test                       # downloads vectors from Hugging Face (cached)
SCHISM_DSP_VECTORS_DIR=path/to/build swift test   # offline, local vectors

# end-to-end: real fp32 Core ML cores + DSP + pipeline on a whole track,
# vs the verified MLX references (~600 MB of model downloads, cached)
SCHISM_DSP_INTEGRATION=1 swift test --filter IntegrationTests
```

The integration tests cover one representative per family (RoFormer 18M,
HTDemucs, AST — the sibling models share every host code path) against
`test_vectors_integration.npz`, running fp32 on CPU. Their per-chunk
closures double as the reference host glue for each model card's I/O
contract.

Comparisons are `numpy.allclose`-style (`|a−b| ≤ atol + rtol·|b|`) with
per-frontend tolerances around 1e-4 relative — the float32-vs-float64-FFT
floor. The pipeline vectors use a deterministic mock model (a 2-tap FIR with
dyadic constants, bit-reproducible in any IEEE implementation) and pass at
1e-6; they are generated by the *shipped* schism-mlx orchestration code, and
are scaled down in size (the algorithms are size-independent — production
chunk sizes are exposed as `Separation.htdemucsSegmentLength` /
`.roformerChunkSize`). Clarity is prioritized over throughput (per-frame
scalar loops, full complex DFT); all routines are still far faster than
realtime.

## License

MIT. The models these frontends feed carry their own licenses — see each
model repo.
