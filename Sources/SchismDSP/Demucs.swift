import Foundation

/// The HTDemucs segment framing around the normalized STFT, transliterated
/// from `HTDemucs._spec` / `_ispec` in schism-mlx (which mirror the demucs
/// reference). The Core ML cores consume/produce this framing:
/// `spec` of a 343980-sample segment is `(2048, 336)` per channel.
public enum Demucs {
    /// `_spec`: reflect-pad by `3*hop/2` (plus right-pad to a hop multiple),
    /// normalized STFT, keep frames `[2, 2+ceil(L/hop))`, drop the Nyquist
    /// bin. Returns `(nFFT/2, ceil(L/hop))`.
    public static func spec(
        _ x: [Float], nFFT: Int = 4096, hopLength: Int = 1024
    ) -> ComplexMatrix {
        let le = (x.count + hopLength - 1) / hopLength
        let pad = hopLength / 2 * 3
        let padded = DSP.reflectPad(x, left: pad, right: pad + le * hopLength - x.count)
        let z = STFT.forward(padded, nFFT: nFFT, hopLength: hopLength, normalized: true)
        var out = ComplexMatrix(freqs: nFFT / 2, frames: le) // Nyquist dropped
        for f in 0..<out.freqs {
            for t in 0..<le {
                out.real[out.index(f, t)] = z.real[z.index(f, t + 2)]
                out.imag[out.index(f, t)] = z.imag[z.index(f, t + 2)]
            }
        }
        return out
    }

    /// `_ispec`: zero-pad the Nyquist bin back, pad two frames of zeros on
    /// each side, normalized iSTFT over the padded length, trim `3*hop/2`.
    public static func ispec(
        _ z: ComplexMatrix, length: Int, hopLength: Int = 1024
    ) -> [Float] {
        let pad = hopLength / 2 * 3
        var full = ComplexMatrix(freqs: z.freqs + 1, frames: z.frames + 4)
        for f in 0..<z.freqs {
            for t in 0..<z.frames {
                full.real[full.index(f, t + 2)] = z.real[z.index(f, t)]
                full.imag[full.index(f, t + 2)] = z.imag[z.index(f, t)]
            }
        }
        let le = hopLength * ((length + hopLength - 1) / hopLength) + 2 * pad
        let y = STFT.inverse(full, hopLength: hopLength, length: le, normalized: true)
        return Array(y[pad..<(pad + length)])
    }
}
