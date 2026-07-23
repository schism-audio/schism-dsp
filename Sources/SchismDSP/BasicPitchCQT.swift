import Foundation

/// Basic Pitch's CQT frontend for one 43844-sample window, transliterated
/// from `schism_mlx.transcribe.basic_pitch.model.CQTFrontend` (itself a
/// verified numpy transliteration of the official ONNX graph:
/// `basic_pitch.layers.nnaudio.CQT2010v2` with sr=22050, hop=256,
/// fmin=27.5, n_bins=309, bins_per_octave=36, followed by
/// `basic_pitch.layers.signal.NormalizedLog`), plus the model's harmonic
/// stacking layer.
///
/// The 2010 multirate scheme: one bank of 36 top-octave complex kernels
/// (length 256) is convolved against the signal, which is then repeatedly
/// lowpassed and decimated by 2 for each of the 9 octaves (the hop halving
/// alongside). All constants are fixed and non-learned; the ONNX graph ships
/// them baked in, and the conversion pipeline verified they match this exact
/// recomputation (`convert.compute_cqt_kernels` / `compute_lowpass_filter`),
/// which is what `init` performs — in Double, cast once to Float.
public struct BasicPitchCQT {
    public static let sampleRate = 22050
    /// One model window: 2 s minus one hop (2*22050 - 256).
    public static let windowSamples = 43844
    public static let framesPerWindow = 172
    public static let cqtBins = 309
    public static let contourBins = 264
    public static let numHarmonics = 8

    static let binsPerOctave = 36
    static let numOctaves = 9
    static let kernelLength = 256
    static let fmin = 27.5
    static let topHop = 256

    /// Harmonic-stacking shifts in contour bins: `round(36 * log2(h))` for
    /// harmonics `[0.5, 1, 2, ..., 7]` -> `[-36, 0, 36, 57, 72, 84, 93, 101]`.
    static let harmonicShifts: [Int] = ([0.5] + (1...7).map(Double.init)).map {
        Int((Double(binsPerOctave) * log2($0)).rounded())
    }

    /// Top-octave kernel bank, `(36, 256)` row-major.
    let kernelsReal: [Float]
    let kernelsImag: [Float]
    /// Anti-aliasing FIR for the decimate-by-2 chain, `(256,)`.
    let lowpass: [Float]
    /// librosa-style per-bin normalization `sqrt(ceil(Q * sr / f_bin))`, `(309,)`.
    let sqrtLengths: [Float]

    public init() {
        (kernelsReal, kernelsImag) = Self.computeKernels()
        lowpass = Self.computeLowpass()
        let q = 1.0 / (pow(2.0, 1.0 / Double(Self.binsPerOctave)) - 1.0)
        sqrtLengths = (0..<Self.cqtBins).map {
            let freq = Self.fmin * pow(2.0, Double($0) / Double(Self.binsPerOctave))
            return Float((q * Double(Self.sampleRate) / freq).rounded(.up).squareRoot())
        }
    }

    // ------------------------------------------------------------------
    // Constants (nnaudio.create_cqt_kernels / create_lowpass_filter)
    // ------------------------------------------------------------------

    /// The 36 top-octave kernels: periodic-Hann-windowed complex exponentials
    /// at `f_k = fmin_t * 2^(k/36)`, support `ceil(Q*sr/f_k)` samples centered
    /// in 256, divided by their length then L1-normalized. `fmin_t` is NOT
    /// simply `fmin * 2^8` — nnaudio derives it back from `fmax_t` using the
    /// bin remainder (309 % 36 = 21), landing the bank on
    /// `7040 * 2^(-15/36) ≈ 5274 Hz`.
    static func computeKernels() -> (real: [Float], imag: [Float]) {
        let bpo = Double(binsPerOctave)
        let sr = Double(sampleRate)
        let q = 1.0 / (pow(2.0, 1.0 / bpo) - 1.0)
        var fminT = fmin * pow(2.0, Double(numOctaves - 1))
        let remainder = cqtBins % binsPerOctave
        let fmaxT = fminT * pow(2.0, Double(remainder - 1) / bpo)
        fminT = fmaxT / pow(2.0, 1.0 - 1.0 / bpo)

        var real = [Float](repeating: 0, count: binsPerOctave * kernelLength)
        var imag = [Float](repeating: 0, count: binsPerOctave * kernelLength)
        for k in 0..<binsPerOctave {
            let freq = fminT * pow(2.0, Double(k) / bpo)
            let length = (q * sr / freq).rounded(.up) // integer-valued Double
            let l = Int(length)
            let start = Int((Double(kernelLength) / 2 - length / 2).rounded(.up)) - (l % 2)
            var re = [Double](repeating: 0, count: l)
            var im = [Double](repeating: 0, count: l)
            var norm = 0.0
            for n in 0..<l {
                let window = 0.5 - 0.5 * cos(2.0 * .pi * Double(n) / length)
                // np.r_[-length//2 : length//2]: floor(-l/2) + n
                let r = Double(-(l + 1) / 2 + n) // floor(-l/2) = -(l+1)/2 for Int
                let angle = r * 2.0 * .pi * freq / sr
                re[n] = window * cos(angle) / length
                im[n] = window * sin(angle) / length
                norm += (re[n] * re[n] + im[n] * im[n]).squareRoot()
            }
            for n in 0..<l {
                real[k * kernelLength + start + n] = Float(re[n] / norm)
                imag[k * kernelLength + start + n] = Float(im[n] / norm)
            }
        }
        return (real, imag)
    }

    /// `scipy.signal.firwin2(256, [0, 0.5/(1+1e-3), 0.5*(1+1e-3), 1],
    /// [1, 1, 0, 0])`, transliterated: linearly interpolate the desired gain
    /// on a 257-point mesh, apply the linear-phase shift `e^{-i·127.5·π·x}`,
    /// inverse-rfft (n=512, direct O(n²) sum — one-time constant), keep the
    /// first 256 taps under a symmetric Hamming window.
    static func computeLowpass() -> [Float] {
        let numtaps = kernelLength
        let nfreqs = 257 // 1 + 2^ceil(log2(numtaps))
        let transition = 1e-3
        let freq = [0.0, 0.5 / (1 + transition), 0.5 * (1 + transition), 1.0]
        let gain = [1.0, 1.0, 0.0, 0.0]

        var fx2Re = [Double](repeating: 0, count: nfreqs)
        var fx2Im = [Double](repeating: 0, count: nfreqs)
        for i in 0..<nfreqs {
            let x = Double(i) / Double(nfreqs - 1)
            var g = gain[gain.count - 1]
            for s in 0..<(freq.count - 1) where x <= freq[s + 1] {
                g = gain[s] + (gain[s + 1] - gain[s]) * (x - freq[s]) / (freq[s + 1] - freq[s])
                break
            }
            let phase = -Double(numtaps - 1) / 2 * .pi * x
            fx2Re[i] = g * cos(phase)
            fx2Im[i] = g * sin(phase)
        }

        let n = 2 * (nfreqs - 1) // 512
        var taps = [Float](repeating: 0, count: numtaps)
        for t in 0..<numtaps {
            var acc = fx2Re[0] + fx2Re[nfreqs - 1] * cos(.pi * Double(t))
            for k in 1..<(nfreqs - 1) {
                let angle = 2.0 * .pi * Double(k) * Double(t) / Double(n)
                acc += 2 * (fx2Re[k] * cos(angle) - fx2Im[k] * sin(angle))
            }
            // symmetric Hamming (scipy general_cosine: 0.54 + 0.46*cos(fac),
            // fac = linspace(-pi, pi, numtaps))
            let fac = -Double.pi + 2.0 * .pi * Double(t) / Double(numtaps - 1)
            taps[t] = Float(acc / Double(n) * (0.54 + 0.46 * cos(fac)))
        }
        return taps
    }

    // ------------------------------------------------------------------
    // Transform
    // ------------------------------------------------------------------

    /// One octave of complex CQT: reflect pad 128, frames of 256 every `hop`
    /// against the kernel bank. `(numFrames, 36)` row-major each. The sign
    /// follows the reference: `CQT_imag = -conv(x, imag_kernels)`.
    func cqtOctave(_ x: [Float], hop: Int) -> (real: [Float], imag: [Float], frames: Int) {
        let klen = Self.kernelLength
        let padded = DSP.reflectPad(x, left: klen / 2, right: klen / 2)
        let numFrames = (padded.count - klen) / hop + 1
        let bpo = Self.binsPerOctave
        var real = [Float](repeating: 0, count: numFrames * bpo)
        var imag = [Float](repeating: 0, count: numFrames * bpo)
        for t in 0..<numFrames {
            let start = t * hop
            for k in 0..<bpo {
                var accRe: Float = 0
                var accIm: Float = 0
                for i in 0..<klen {
                    accRe += padded[start + i] * kernelsReal[k * klen + i]
                    accIm += padded[start + i] * kernelsImag[k * klen + i]
                }
                real[t * bpo + k] = accRe
                imag[t * bpo + k] = -accIm
            }
        }
        return (real, imag, numFrames)
    }

    /// Anti-aliased 2x decimation (torch-style symmetric ZERO padding of
    /// 127 — unlike the reflect padding of the octave conv).
    func downsampleBy2(_ x: [Float]) -> [Float] {
        let klen = Self.kernelLength
        let pad = (klen - 1) / 2
        var padded = [Float](repeating: 0, count: x.count + 2 * pad)
        for i in 0..<x.count { padded[pad + i] = x[i] }
        let n = (padded.count - klen) / 2 + 1
        var out = [Float](repeating: 0, count: n)
        for t in 0..<n {
            var acc: Float = 0
            for i in 0..<klen { acc += padded[2 * t + i] * lowpass[i] }
            out[t] = acc
        }
        return out
    }

    /// CQT magnitude, `(172, 309)` row-major: 9 octaves assembled lowest
    /// octave first, the top 309 of the 324 bins kept (the bottom kernel
    /// bank overshoots below fmin), each bin scaled by `sqrtLengths`.
    public func magnitude(_ window: [Float]) -> [Float] {
        precondition(
            window.count == Self.windowSamples,
            "expected a \(Self.windowSamples)-sample window"
        )
        var x = window
        var hop = Self.topHop
        var octaves: [(real: [Float], imag: [Float])] = []
        var frames = 0
        for octave in 0..<Self.numOctaves {
            let (re, im, n) = cqtOctave(x, hop: hop)
            octaves.append((re, im))
            precondition(octave == 0 || n == frames, "octave frame counts diverge")
            frames = n // 172 for every octave under the input contract
            if octave < Self.numOctaves - 1 {
                x = downsampleBy2(x)
                hop /= 2
            }
        }

        let bpo = Self.binsPerOctave
        let total = Self.numOctaves * bpo // 324
        let drop = total - Self.cqtBins // 15 sub-fmin bins
        var out = [Float](repeating: 0, count: frames * Self.cqtBins)
        for t in 0..<frames {
            for b in 0..<Self.cqtBins {
                let j = b + drop // index into the lowest-octave-first concat
                let octave = Self.numOctaves - 1 - j / bpo
                let k = j % bpo
                let re = octaves[octave].real[t * bpo + k] * sqrtLengths[b]
                let im = octaves[octave].imag[t * bpo + k] * sqrtLengths[b]
                out[t * Self.cqtBins + b] = (re * re + im * im).squareRoot()
            }
        }
        return out
    }

    /// `NormalizedLog`: log-power in dB (the reference computes
    /// `10 * (ln(power + 1e-10) * 0.43429446)` — that float32 constant, not
    /// an exact log10), min-subtracted then max-divided *per window*, so the
    /// output lands in [0, 1]. `(172, 309)` row-major — the body-only model
    /// input.
    public func features(_ window: [Float]) -> [Float] {
        var logPower = magnitude(window)
        for i in 0..<logPower.count {
            let power = logPower[i] * logPower[i]
            logPower[i] = 10 * (log(power + 1e-10) * 0.43429446)
        }
        var lo = logPower[0]
        for v in logPower where v < lo { lo = v }
        for i in 0..<logPower.count { logPower[i] -= lo }
        var peak = logPower[0]
        for v in logPower where v > peak { peak = v }
        guard peak != 0 else { return [Float](repeating: 0, count: logPower.count) }
        for i in 0..<logPower.count { logPower[i] /= peak }
        return logPower
    }

    /// Harmonic stacking, `(172, 309)` features -> `(172, 264, 8)` row-major:
    /// channel `c` at bin `k` is `features[t, k + shift_c]` (zero where out
    /// of range), cropped to the first 264 contour bins.
    public static func harmonicStack(_ features: [Float]) -> [Float] {
        precondition(features.count % cqtBins == 0)
        let frames = features.count / cqtBins
        let shifts = harmonicShifts
        var out = [Float](repeating: 0, count: frames * contourBins * shifts.count)
        for t in 0..<frames {
            for k in 0..<contourBins {
                for (c, shift) in shifts.enumerated() {
                    let src = k + shift
                    guard src >= 0 && src < cqtBins else { continue }
                    out[(t * contourBins + k) * shifts.count + c] =
                        features[t * cqtBins + src]
                }
            }
        }
        return out
    }
}
