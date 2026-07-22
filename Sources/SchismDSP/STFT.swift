import Foundation

/// `torch.stft` / `torch.istft`-compatible STFT, transliterated from
/// `schism_mlx.audio.stft_torchlike` / `istft_torchlike` (the verified numpy
/// references the Core ML models ship with).
///
/// Conventions: periodic Hann window (`win_length == n_fft`), `center=True`
/// reflect padding, `normalized` multiplies the analysis by `1/sqrt(n_fft)`
/// (HTDemucs: `true`; Mini-BS-RoFormer: `false`).
public enum STFT {
    /// One channel of audio -> `(n_fft/2 + 1, num_frames)` complex.
    public static func forward(
        _ x: [Float], nFFT: Int, hopLength: Int, normalized: Bool
    ) -> ComplexMatrix {
        let padded = DSP.reflectPad(x, left: nFFT / 2, right: nFFT / 2)
        let numFrames = 1 + (padded.count - nFFT) / hopLength
        let window = DSP.hannPeriodic(nFFT)
        let fft = RealFFT(length: nFFT)
        let bins = nFFT / 2 + 1
        var out = ComplexMatrix(freqs: bins, frames: numFrames)
        let scale: Float = normalized ? 1 / sqrt(Float(nFFT)) : 1

        var frame = [Float](repeating: 0, count: nFFT)
        for t in 0..<numFrames {
            let start = t * hopLength
            for i in 0..<nFFT { frame[i] = padded[start + i] * window[i] }
            let (re, im) = fft.forward(frame)
            for f in 0..<bins {
                out.real[out.index(f, t)] = re[f] * scale
                out.imag[out.index(f, t)] = im[f] * scale
            }
        }
        return out
    }

    /// Inverse of `forward`, matching `torch.istft(..., center=True,
    /// length: length)`: windowed overlap-add normalized by the summed
    /// squared window (accumulated in Double, as the numpy reference does),
    /// then trimmed by `n_fft/2` on the left.
    public static func inverse(
        _ z: ComplexMatrix, hopLength: Int, length: Int, normalized: Bool
    ) -> [Float] {
        let nFFT = 2 * (z.freqs - 1)
        let window = DSP.hannPeriodic(nFFT)
        let fft = RealFFT(length: nFFT)
        let total = nFFT + hopLength * (z.frames - 1)
        var acc = [Double](repeating: 0, count: total)
        var wsum = [Double](repeating: 0, count: total)
        let scale: Float = normalized ? sqrt(Float(nFFT)) : 1

        var re = [Float](repeating: 0, count: z.freqs)
        var im = [Float](repeating: 0, count: z.freqs)
        for t in 0..<z.frames {
            for f in 0..<z.freqs {
                re[f] = z.real[z.index(f, t)]
                im[f] = z.imag[z.index(f, t)]
            }
            let frame = fft.inverse(real: re, imag: im)
            let start = t * hopLength
            for i in 0..<nFFT {
                acc[start + i] += Double(frame[i] * scale * window[i])
                wsum[start + i] += Double(window[i] * window[i])
            }
        }
        let tiny = Double(Float.leastNormalMagnitude)
        for i in 0..<total where wsum[i] > tiny { acc[i] /= wsum[i] }

        var out = [Float](repeating: 0, count: length)
        for i in 0..<length {
            let j = nFFT / 2 + i
            out[i] = j < total ? Float(acc[j]) : 0
        }
        return out
    }
}
