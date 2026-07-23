import Foundation

/// Beat This!'s log-mel frontend, transliterated from
/// `schism_mlx.analyze.beat_this.model.logmel_beat_this`: 22.05 kHz mono,
/// `torchaudio.transforms.MelSpectrogram(n_fft=1024, hop_length=441,
/// power=1, normalized="frame_length", mel_scale="slaney", norm=None,
/// f_min=30, f_max=11000, n_mels=128)` followed by `log1p(1000 * x)`.
public enum BeatThisMel {
    /// Bit-exact replica of float32 `torch.linspace`: two-sided fill from
    /// float32 start/end/step with the accumulation done in double (one
    /// rounding per element), lower half from the start, upper half from
    /// the end.
    static func linspaceTorchlike(_ start: Float, _ end: Float, _ steps: Int) -> [Float] {
        let step = (end - start) / Float(steps - 1)
        return (0..<steps).map { i in
            i < steps / 2
                ? Float(Double(start) + Double(step) * Double(i))
                : Float(Double(end) - Double(step) * Double(steps - 1 - i))
        }
    }

    /// Slaney-scale triangular mel filterbank *without* area normalization —
    /// `torchaudio.functional.melscale_fbanks(..., norm=None,
    /// mel_scale="slaney")`, the torchaudio `MelSpectrogram` default. This
    /// differs from `LogMel.melBanks` (librosa `norm="slaney"`), which
    /// multiplies each filter by `2 / (melF[m+2] - melF[m])`.
    ///
    /// Float32 tensor arithmetic throughout, as torchaudio computes it.
    /// Returns `(numMelBins, nFFT/2 + 1)` row-major.
    public static func melBanks(
        sampleRate: Int = 22050, nFFT: Int = 1024, numMelBins: Int = 128,
        fmin: Double = 30, fmax: Double = 11000
    ) -> [Float] {
        let fSp = 200.0 / 3
        let minLogHz = 1000.0
        let minLogMel = minLogHz / fSp
        let logstep = log(6.4) / 27

        func hzToMel(_ freq: Double) -> Double {
            freq >= minLogHz ? minLogMel + log(freq / minLogHz) / logstep : freq / fSp
        }

        let bins = nFFT / 2 + 1
        let allFreqs = linspaceTorchlike(0, Float(sampleRate / 2), bins)
        let mPts = linspaceTorchlike(
            Float(hzToMel(fmin)), Float(hzToMel(fmax)), numMelBins + 2
        )
        // mel -> Hz in float32, as torchaudio's _mel_to_hz
        let fPts: [Float] = mPts.map { m in
            m >= Float(minLogMel)
                ? Float(minLogHz) * exp(Float(logstep) * (m - Float(minLogMel)))
                : Float(fSp) * m
        }

        var banks = [Float](repeating: 0, count: numMelBins * bins)
        for m in 0..<numMelBins {
            let fDiffLower = fPts[m + 1] - fPts[m]
            let fDiffUpper = fPts[m + 2] - fPts[m + 1]
            for f in 0..<bins {
                let down = (allFreqs[f] - fPts[m]) / fDiffLower
                let up = (fPts[m + 2] - allFreqs[f]) / fDiffUpper
                banks[m * bins + f] = max(0, min(down, up))
            }
        }
        return banks
    }

    /// `(numFrames, numMelBins)` log-mel, row-major, at
    /// `sampleRate / hopLength` = 50 fps: `log1p(1000 * mel)` over the
    /// *magnitude* (power=1) STFT scaled by `1/sqrt(n_fft)` (torchaudio's
    /// `normalized="frame_length"` with `win_length == n_fft`).
    public static func compute(
        _ waveform: [Float],
        sampleRate: Int = 22050, nFFT: Int = 1024, hopLength: Int = 441,
        numMelBins: Int = 128, fmin: Double = 30, fmax: Double = 11000,
        logMultiplier: Float = 1000
    ) -> (frames: Int, data: [Float]) {
        let z = STFT.forward(
            waveform, nFFT: nFFT, hopLength: hopLength, normalized: true
        )
        let banks = melBanks(
            sampleRate: sampleRate, nFFT: nFFT, numMelBins: numMelBins,
            fmin: fmin, fmax: fmax
        )
        let bins = z.freqs
        var out = [Float](repeating: 0, count: z.frames * numMelBins)
        var magnitude = [Float](repeating: 0, count: bins)
        for t in 0..<z.frames {
            for f in 0..<bins {
                let re = z.real[z.index(f, t)]
                let im = z.imag[z.index(f, t)]
                magnitude[f] = sqrt(re * re + im * im)
            }
            for m in 0..<numMelBins {
                var acc: Float = 0
                for f in 0..<bins { acc += magnitude[f] * banks[m * bins + f] }
                out[t * numMelBins + m] = log1p(logMultiplier * acc)
            }
        }
        return (z.frames, out)
    }
}
