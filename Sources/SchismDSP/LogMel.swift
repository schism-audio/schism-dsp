import Foundation

/// librosa/torchlibrosa-style log-mel spectrogram, transliterated from
/// `schism_mlx.audio.logmel` — the PANNs CNN14 frontend: 32 kHz, n_fft 1024,
/// hop 320, 64 mels, 50–14000 Hz, Slaney mel scale + Slaney norm,
/// `10*log10(clamp(mel, 1e-10))`.
public enum LogMel {
    static func hzToMelSlaney(_ freq: Double) -> Double {
        let fSp = 200.0 / 3
        if freq >= 1000 {
            return 1000 / fSp + log(freq / 1000) / (log(6.4) / 27)
        }
        return freq / fSp
    }

    static func melToHzSlaney(_ mel: Double) -> Double {
        let fSp = 200.0 / 3
        let minLogMel = 1000 / fSp
        if mel >= minLogMel {
            return 1000 * exp(log(6.4) / 27 * (mel - minLogMel))
        }
        return mel * fSp
    }

    /// Slaney-normalized triangular mel banks as `librosa.filters.mel`
    /// (htk=False, norm="slaney"): `(numMelBins, nFFT/2 + 1)` row-major.
    public static func melBanks(
        sampleRate: Double = 32000, nFFT: Int = 1024, numMelBins: Int = 64,
        fmin: Double = 50, fmax: Double = 14000
    ) -> [Float] {
        let bins = nFFT / 2 + 1
        let fftFreqs = (0..<bins).map { sampleRate / 2 * Double($0) / Double(bins - 1) }
        let melLow = hzToMelSlaney(fmin)
        let melHigh = hzToMelSlaney(fmax)
        let melF = (0..<(numMelBins + 2)).map {
            melToHzSlaney(melLow + (melHigh - melLow) * Double($0) / Double(numMelBins + 1))
        }
        var banks = [Float](repeating: 0, count: numMelBins * bins)
        for m in 0..<numMelBins {
            let enorm = 2.0 / (melF[m + 2] - melF[m])
            for f in 0..<bins {
                let lower = (fftFreqs[f] - melF[m]) / (melF[m + 1] - melF[m])
                let upper = (melF[m + 2] - fftFreqs[f]) / (melF[m + 2] - melF[m + 1])
                banks[m * bins + f] = Float(max(0, min(lower, upper)) * enorm)
            }
        }
        return banks
    }

    /// `(numFrames, numMelBins)` log-mel, row-major — the CNN14 "logmel"
    /// model input is 10 s of this (1001 frames).
    public static func compute(
        _ waveform: [Float],
        sampleRate: Double = 32000, nFFT: Int = 1024, hopLength: Int = 320,
        numMelBins: Int = 64, fmin: Double = 50, fmax: Double = 14000,
        amin: Float = 1e-10
    ) -> (frames: Int, data: [Float]) {
        let padded = DSP.reflectPad(waveform, left: nFFT / 2, right: nFFT / 2)
        let numFrames = 1 + (padded.count - nFFT) / hopLength
        let window = DSP.hannPeriodic(nFFT)
        let fft = RealFFT(length: nFFT)
        let bins = nFFT / 2 + 1
        let banks = melBanks(
            sampleRate: sampleRate, nFFT: nFFT, numMelBins: numMelBins,
            fmin: fmin, fmax: fmax
        )

        var out = [Float](repeating: 0, count: numFrames * numMelBins)
        var frame = [Float](repeating: 0, count: nFFT)
        for t in 0..<numFrames {
            let start = t * hopLength
            for i in 0..<nFFT { frame[i] = padded[start + i] * window[i] }
            let (re, im) = fft.forward(frame)
            var power = [Float](repeating: 0, count: bins)
            for f in 0..<bins { power[f] = re[f] * re[f] + im[f] * im[f] }
            for m in 0..<numMelBins {
                var acc: Float = 0
                for f in 0..<bins { acc += power[f] * banks[m * bins + f] }
                out[t * numMelBins + m] = 10 * log10(max(acc, amin))
            }
        }
        return (numFrames, out)
    }
}
