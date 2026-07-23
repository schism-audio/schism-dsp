import Foundation

/// EfficientAT's `AugmentMelSTFT` frontend in eval mode, transliterated from
/// `schism_mlx.audio.efficient_at_logmel` (fschmid56/EfficientAT
/// `models/preprocess.py`): 32 kHz, conv-form pre-emphasis, torch.stft with
/// n_fft 1024 / hop 320 / win_length 800, Kaldi mel banks 128 bins 0–15 kHz,
/// `ln(mel + 1e-5)`, then the "fast normalization" `(x + 4.5) / 5`.
public enum EfficientATMel {
    /// `(numFrames, numMelBins)` normalized log-mel, row-major — the MN
    /// models take 10 s of this (exactly 1000 frames).
    ///
    /// Stages, exactly as the reference:
    /// 1. pre-emphasis `y[t] = x[t+1] - 0.97 * x[t]` — a conv1d with kernel
    ///    `[-0.97, 1]`, so the output is one sample *shorter* (unlike Kaldi's
    ///    per-frame replicate-padded form in `KaldiFbank`),
    /// 2. `torch.stft(center=True)` reflect padding; the window is a
    ///    *symmetric* Hann of `winLength` zero-padded on both sides to
    ///    `nFFT` (torch.stft centers a short window inside the frame),
    /// 3. power spectrum through Kaldi mel banks
    ///    (`torchaudio.compliance.kaldi.get_mel_banks`) with a zeroed
    ///    Nyquist column,
    /// 4. `log(mel + 1e-5)`, then `(log + 4.5) / 5`.
    public static func compute(
        _ waveform: [Float],
        sampleRate: Float = 32000,
        nFFT: Int = 1024,
        winLength: Int = 800,
        hopLength: Int = 320,
        numMelBins: Int = 128,
        fmin: Float = 0,
        fmax: Float = 15000,
        preemphasis: Float = 0.97
    ) -> (frames: Int, data: [Float]) {
        // 1. conv-form pre-emphasis: length N-1
        var x = [Float](repeating: 0, count: waveform.count - 1)
        for i in 0..<x.count { x[i] = waveform[i + 1] - preemphasis * waveform[i] }

        // 2. torch.stft framing: reflect pad nFFT/2, symmetric Hann of
        // winLength centered into nFFT by zero padding
        let padded = DSP.reflectPad(x, left: nFFT / 2, right: nFFT / 2)
        let numFrames = 1 + (padded.count - nFFT) / hopLength
        let short = DSP.hannSymmetric(winLength)
        let padLeft = (nFFT - winLength) / 2
        var window = [Float](repeating: 0, count: nFFT)
        for i in 0..<winLength { window[padLeft + i] = short[i] }

        let fft = RealFFT(length: nFFT)
        let bins = nFFT / 2 + 1

        // 3. Kaldi mel banks (numMelBins, nFFT/2); the reference appends a
        // zero Nyquist column, realized here by summing only bins-1 bins
        let banks = KaldiFbank.melBanks(
            numBins: numMelBins, paddedWindowSize: nFFT,
            sampleRate: sampleRate, lowFreq: fmin, highFreq: fmax
        )

        var out = [Float](repeating: 0, count: numFrames * numMelBins)
        var frame = [Float](repeating: 0, count: nFFT)
        for t in 0..<numFrames {
            let start = t * hopLength
            for i in 0..<nFFT { frame[i] = padded[start + i] * window[i] }
            let (re, im) = fft.forward(frame)
            var power = [Float](repeating: 0, count: bins)
            for f in 0..<bins { power[f] = re[f] * re[f] + im[f] * im[f] }

            // 4. mel = power @ banks.T, ln(mel + 1e-5), (x + 4.5) / 5
            for m in 0..<numMelBins {
                var acc: Float = 0
                for f in 0..<(bins - 1) { acc += power[f] * banks[m * (bins - 1) + f] }
                out[t * numMelBins + m] = (log(acc + 1e-5) + 4.5) / 5
            }
        }
        return (numFrames, out)
    }
}
