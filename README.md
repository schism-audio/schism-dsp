# schism-dsp

Swift reference implementations of the host-side DSP frontends for the
[schism-audio](https://huggingface.co/schism-audio) Core ML models, built on
Accelerate. Companion to [schism-mlx](https://github.com/schism-audio/schism-mlx).

The Core ML `.mlpackage`s in the org contain the networks only; the host is
responsible for feature extraction and resynthesis. Every routine here is a
transliteration of the verified numpy references in `schism_mlx.audio` and is
tested against the `test_vectors_*.npz` shipped in each model repo — an
implementation passing those vectors is interchangeable with the pipeline the
Core ML models were verified against.

| Frontend | For | Vector source |
|---|---|---|
| `KaldiFbank.fbank` / `.astFeatures` | [ast-audioset-10-10-coreml](https://huggingface.co/schism-audio/ast-audioset-10-10-coreml) | `test_vectors_fbank.npz` |
| `LogMel.compute` | [cnn14-audioset-coreml](https://huggingface.co/schism-audio/cnn14-audioset-coreml) | `test_vectors_logmel.npz` |
| `STFT.forward/inverse` (normalized) + `Demucs.spec/ispec` framing | [htdemucs-coreml](https://huggingface.co/schism-audio/htdemucs-coreml) (+ ft, 6s) | `test_vectors_stft.npz` |
| `STFT.forward/inverse` (unnormalized) + `Roformer.merge/unmerge/applyMask` | [mini-bs-roformer-18m-coreml](https://huggingface.co/schism-audio/mini-bs-roformer-18m-coreml), [v2](https://huggingface.co/schism-audio/mini-bs-roformer-v2-coreml) | `test_vectors_stft.npz` |

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

## Testing

```sh
swift test                       # downloads vectors from Hugging Face (cached)
SCHISM_DSP_VECTORS_DIR=path/to/build swift test   # offline, local vectors
```

Comparisons are `numpy.allclose`-style (`|a−b| ≤ atol + rtol·|b|`) with
per-frontend tolerances around 1e-4 relative — the float32-vs-float64-FFT
floor. Clarity is prioritized over throughput (per-frame scalar loops, full
complex DFT); all routines are still far faster than realtime.

## License

MIT. The models these frontends feed carry their own licenses — see each
model repo.
