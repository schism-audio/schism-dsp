import Foundation

/// Kaldi-compatible log-mel filterbank, transliterated from
/// `schism_mlx.audio.kaldi_fbank` (verified against
/// `torchaudio.compliance.kaldi.fbank`) — the AST frontend:
/// `fbank(w, htk_compat=True, use_energy=False, window_type="hanning",
/// dither=0.0, num_mel_bins=128)` at 16 kHz.
public enum KaldiFbank {
    static let epsilon = Float.ulpOfOne // torchaudio's EPSILON (f32 machine eps)

    static func melScale(_ freq: Float) -> Float {
        1127 * log1p(freq / 700)
    }

    /// Kaldi triangular mel banks, `(numBins, paddedWindowSize/2)` row-major —
    /// the Nyquist bin is excluded, exactly as in Kaldi/torchaudio.
    public static func melBanks(
        numBins: Int, paddedWindowSize: Int, sampleRate: Float,
        lowFreq: Float = 20, highFreq: Float = 0
    ) -> [Float] {
        let numFFTBins = paddedWindowSize / 2
        let high = highFreq <= 0 ? sampleRate / 2 + highFreq : highFreq
        let binWidth = sampleRate / Float(paddedWindowSize)
        let melLow = melScale(lowFreq)
        let melDelta = (melScale(high) - melLow) / Float(numBins + 1)

        var banks = [Float](repeating: 0, count: numBins * numFFTBins)
        for m in 0..<numBins {
            let left = melLow + Float(m) * melDelta
            let center = melLow + Float(m + 1) * melDelta
            let right = melLow + Float(m + 2) * melDelta
            for f in 0..<numFFTBins {
                let mel = melScale(binWidth * Float(f))
                let up = (mel - left) / (center - left)
                let down = (right - mel) / (right - center)
                banks[m * numFFTBins + f] = max(0, min(up, down))
            }
        }
        return banks
    }

    /// `(numFrames, numMelBins)` log-mel energies, row-major. Frames use
    /// snip_edges=true (no padding: frames start at `i * shift` and the tail
    /// that doesn't fill a window is dropped).
    public static func fbank(
        _ waveform: [Float],
        sampleRate: Float = 16000,
        numMelBins: Int = 128,
        frameLengthMs: Float = 25,
        frameShiftMs: Float = 10,
        preemphasis: Float = 0.97,
        removeDCOffset: Bool = true,
        lowFreq: Float = 20,
        highFreq: Float = 0
    ) -> (frames: Int, data: [Float]) {
        let windowSize = Int(sampleRate * frameLengthMs * 0.001)
        let windowShift = Int(sampleRate * frameShiftMs * 0.001)
        var padded = 1
        while padded < windowSize { padded <<= 1 } // Kaldi pads FFT to pow2

        guard waveform.count >= windowSize else { return (0, []) }
        let numFrames = 1 + (waveform.count - windowSize) / windowShift
        let window = DSP.hannSymmetric(windowSize) // Kaldi "hanning"
        let fft = RealFFT(length: padded)
        let bins = padded / 2 + 1

        // (numMelBins, padded/2) — torchaudio's zero Nyquist column is
        // realized below by summing only the first bins-1 power bins
        let banks = melBanks(
            numBins: numMelBins, paddedWindowSize: padded,
            sampleRate: sampleRate, lowFreq: lowFreq, highFreq: highFreq
        )

        var out = [Float](repeating: 0, count: numFrames * numMelBins)
        var frame = [Float](repeating: 0, count: padded)
        for t in 0..<numFrames {
            let start = t * windowShift
            for i in 0..<windowSize { frame[i] = waveform[start + i] }
            for i in windowSize..<padded { frame[i] = 0 }

            if removeDCOffset {
                var mean: Float = 0
                for i in 0..<windowSize { mean += frame[i] }
                mean /= Float(windowSize)
                for i in 0..<windowSize { frame[i] -= mean }
            }
            if preemphasis != 0 {
                // first sample pre-emphasized against itself (replicate pad)
                for i in stride(from: windowSize - 1, through: 1, by: -1) {
                    frame[i] -= preemphasis * frame[i - 1]
                }
                frame[0] -= preemphasis * frame[0]
            }
            for i in 0..<windowSize { frame[i] *= window[i] }

            let (re, im) = fft.forward(frame)
            var power = [Float](repeating: 0, count: bins)
            for f in 0..<bins { power[f] = re[f] * re[f] + im[f] * im[f] }

            // mel = power @ banks.T; the Nyquist power bin (f = bins-1) is
            // multiplied by the implicit zero column and dropped
            for m in 0..<numMelBins {
                var acc: Float = 0
                for f in 0..<(bins - 1) { acc += power[f] * banks[m * (bins - 1) + f] }
                out[t * numMelBins + m] = log(max(acc, epsilon))
            }
        }
        return (numFrames, out)
    }

    /// The full AST model input: fbank, normalized
    /// `(x + 4.2677393) / (2 * 4.5689974)`, zero-padded or truncated to
    /// `maxFrames` rows. Returns `(maxFrames, 128)` row-major.
    public static func astFeatures(
        _ waveform: [Float], maxFrames: Int = 1024
    ) -> [Float] {
        let (frames, fb) = fbank(waveform)
        let bins = 128
        var out = [Float](repeating: 0, count: maxFrames * bins)
        for t in 0..<min(frames, maxFrames) {
            for m in 0..<bins {
                out[t * bins + m] = (fb[t * bins + m] + 4.2677393) / (2 * 4.5689974)
            }
        }
        return out
    }
}
