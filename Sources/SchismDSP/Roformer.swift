import Foundation

/// Layout helpers for the Mini-BS-RoFormer Core ML cores
/// (`schism-audio/mini-bs-roformer-{18m,v2}-coreml`).
///
/// The model input "spec" and output "mask" use one merged layout:
/// `spec[t, k]` with `k = (f * C + c) * 2 + reim` — frequency outer, then
/// channel, re/im innermost. This is `rearrange("b c f t T -> b t (f c T)")`
/// from the reference implementation.
public enum Roformer {
    /// Per-channel spectrograms (each `freqs x frames`) -> flat
    /// `(frames, freqs * channels * 2)` model input, frame-major.
    public static func merge(_ channels: [ComplexMatrix]) -> [Float] {
        let c = channels.count
        let freqs = channels[0].freqs
        let frames = channels[0].frames
        var out = [Float](repeating: 0, count: frames * freqs * c * 2)
        for t in 0..<frames {
            let row = t * freqs * c * 2
            for f in 0..<freqs {
                for ch in 0..<c {
                    let k = row + (f * c + ch) * 2
                    out[k] = channels[ch].real[channels[ch].index(f, t)]
                    out[k + 1] = channels[ch].imag[channels[ch].index(f, t)]
                }
            }
        }
        return out
    }

    /// Inverse of `merge` — used to unpack one stem of the model's "mask"
    /// output (pass that stem's `(frames, freqs*channels*2)` slice).
    public static func unmerge(
        _ merged: [Float], freqs: Int, channels: Int, frames: Int
    ) -> [ComplexMatrix] {
        precondition(merged.count == frames * freqs * channels * 2)
        var out = (0..<channels).map { _ in ComplexMatrix(freqs: freqs, frames: frames) }
        for t in 0..<frames {
            let row = t * freqs * channels * 2
            for f in 0..<freqs {
                for ch in 0..<channels {
                    let k = row + (f * channels + ch) * 2
                    out[ch].real[out[ch].index(f, t)] = merged[k]
                    out[ch].imag[out[ch].index(f, t)] = merged[k + 1]
                }
            }
        }
        return out
    }

    /// Apply one stem's complex mask to a synthesis spectrogram:
    /// `masked = z * mask` (complex multiply), elementwise.
    public static func applyMask(
        _ z: ComplexMatrix, mask: ComplexMatrix
    ) -> ComplexMatrix {
        precondition(z.freqs == mask.freqs && z.frames == mask.frames)
        var out = ComplexMatrix(freqs: z.freqs, frames: z.frames)
        for i in 0..<z.real.count {
            out.real[i] = z.real[i] * mask.real[i] - z.imag[i] * mask.imag[i]
            out.imag[i] = z.real[i] * mask.imag[i] + z.imag[i] * mask.real[i]
        }
        return out
    }
}
