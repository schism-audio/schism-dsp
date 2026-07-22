import Foundation

/// Full-track orchestration for the schism-audio separation models:
/// chunking, blending and fold normalization, transliterated from the
/// verified `separate()` implementations in schism-mlx (which mirror the
/// upstream references). The per-chunk model call — Core ML plus the
/// STFT/iSTFT from SchismDSP — is supplied as a closure.
///
/// Waveforms are `[channels][samples]`; stems are `[source][channel][samples]`.
/// Mono input should be duplicated to the model's channel count by the
/// caller. Accumulation is in Double, matching the references' float64.
public enum Separation {
    /// Production chunk sizes (samples at 44.1 kHz) — the values the Core ML
    /// exports were traced and verified with.
    public static let htdemucsSegmentLength = 343_980 // 7.8 s
    public static let roformerChunkSize = 352_800 // 8.0 s

    /// HTDemucs-style segmented separation, mirroring the reference
    /// `apply_model(shifts=0, split=True)`: segments every
    /// `(1 - overlap) * segmentLength` samples, each **centered-padded by
    /// pulling real samples from the surrounding track** (zero only past the
    /// track edges), model output center-trimmed back and blended with a
    /// triangular weight.
    ///
    /// - Parameters:
    ///   - mix: `[channels][length]` waveform at the model rate.
    ///   - sources: number of output stems.
    ///   - segmentLength: model segment size (`htdemucsSegmentLength` in
    ///     production; test vectors use a scaled-down value).
    ///   - overlap: overlap fraction (reference default 0.25).
    ///   - process: `[channels][segmentLength] -> [sources][channels][segmentLength]`.
    /// - Returns: `[sources][channels][length]`.
    public static func demucs(
        mix: [[Float]], sources: Int, segmentLength: Int, overlap: Double = 0.25,
        process: ([[Float]]) throws -> [[[Float]]]
    ) rethrows -> [[[Float]]] {
        let channels = mix.count
        let length = mix[0].count
        let strideLength = Int((1 - overlap) * Double(segmentLength))

        // triangular weight: concat(1...seg/2, seg-seg/2...1) / max
        let half = segmentLength / 2
        var weight = [Float](repeating: 0, count: segmentLength)
        for i in 0..<half { weight[i] = Float(i + 1) }
        for i in 0..<(segmentLength - half) {
            weight[half + i] = Float(segmentLength - half - i)
        }
        let wmax = weight.max() ?? 1
        for i in 0..<segmentLength { weight[i] /= wmax }

        var out = Array(
            repeating: Array(
                repeating: [Double](repeating: 0, count: length), count: channels
            ),
            count: sources
        )
        var sumWeight = [Double](repeating: 0, count: length)

        var offset = 0
        while offset < length {
            let chunkLength = min(segmentLength, length - offset)
            let delta = segmentLength - chunkLength
            let start = offset - delta / 2
            let end = start + segmentLength
            let cStart = max(0, start)
            let cEnd = min(length, end)
            var padded = Array(
                repeating: [Float](repeating: 0, count: segmentLength),
                count: channels
            )
            for c in 0..<channels {
                for i in cStart..<cEnd { padded[c][i - start] = mix[c][i] }
            }
            let res = try process(padded)
            let trim = delta / 2
            for s in 0..<sources {
                for c in 0..<channels {
                    for i in 0..<chunkLength {
                        out[s][c][offset + i] += Double(weight[i] * res[s][c][trim + i])
                    }
                }
            }
            for i in 0..<chunkLength { sumWeight[offset + i] += Double(weight[i]) }
            offset += strideLength
        }
        return out.map { stem in
            stem.map { ch in (0..<length).map { Float(ch[$0] / sumWeight[$0]) } }
        }
    }

    /// Reference bag combination (e.g. `htdemucs_ft`): each model's
    /// full-track estimate, weighted per stem and averaged.
    ///
    /// - Parameters:
    ///   - estimates: `[model][sources][channels][length]`, one per bag member.
    ///   - weights: `[model][sources]` bag weights (one-hot for htdemucs_ft).
    /// - Returns: `[sources][channels][length]`.
    public static func bagCombine(
        _ estimates: [[[[Float]]]], weights: [[Double]]
    ) -> [[[Float]]] {
        let sources = estimates[0].count
        let channels = estimates[0][0].count
        let length = estimates[0][0][0].count
        var totals = [Double](repeating: 0, count: sources)
        var acc = Array(
            repeating: Array(
                repeating: [Double](repeating: 0, count: length), count: channels
            ),
            count: sources
        )
        for (m, estimate) in estimates.enumerated() {
            for s in 0..<sources {
                let w = weights[m][s]
                totals[s] += w
                for c in 0..<channels {
                    for i in 0..<length { acc[s][c][i] += w * Double(estimate[s][c][i]) }
                }
            }
        }
        return (0..<sources).map { s in
            acc[s].map { ch in ch.map { Float($0 / totals[s]) } }
        }
    }

    /// Mini-BS-RoFormer chunked separation, mirroring `separate()` in
    /// schism-mlx `bs_roformer/model.py`: 50%-overlapped chunks on a fixed
    /// grid (track zero-padded to fill it), blended with a linear fade
    /// window (fade = chunk/10) and fold-normalized `acc / max(wsum, 1e-8)`.
    ///
    /// `gapSize` zeroes that many samples at each window edge before the
    /// fade. The v1 upstream default (1 s) silences at least the first
    /// second of every track — the model cards and schism-mlx default to 0;
    /// so does this.
    ///
    /// - Parameters:
    ///   - mix: `[channels][length]` waveform at the model rate.
    ///   - sources: number of output stems.
    ///   - chunkSize: model chunk size (`roformerChunkSize` in production).
    ///   - gapSize: zeroed samples at each window edge (default 0).
    ///   - process: `[channels][chunkSize] -> [sources][channels][chunkSize]`.
    /// - Returns: `[sources][channels][length]`.
    public static func roformer(
        mix: [[Float]], sources: Int, chunkSize: Int, gapSize: Int = 0,
        process: ([[Float]]) throws -> [[[Float]]]
    ) rethrows -> [[[Float]]] {
        let channels = mix.count
        let length = mix[0].count
        let overlap = chunkSize / 2
        let fade = chunkSize / 10

        // ones with linear fades over the core, zero-padded by gapSize;
        // fades mirror np.linspace(0, 1, fade): i * (1/(fade-1)) in float64
        // with forced endpoints, cast to float32
        let core = chunkSize - 2 * gapSize
        var window = [Float](repeating: 0, count: chunkSize)
        for i in 0..<core { window[gapSize + i] = 1 }
        let step = 1.0 / Double(fade - 1)
        for i in 0..<fade {
            window[gapSize + i] = Float(Double(i) * step)
            window[gapSize + core - fade + i] = Float(1.0 - Double(i) * step)
        }
        window[gapSize + fade - 1] = 1
        window[gapSize + core - 1] = 0

        let excess = max(length - chunkSize, 0)
        let n = excess == 0 ? 1 : (excess + overlap - 1) / overlap + 1
        let required = (n - 1) * overlap + chunkSize

        var acc = Array(
            repeating: Array(
                repeating: [Double](repeating: 0, count: required), count: channels
            ),
            count: sources
        )
        var wsum = [Double](repeating: 0, count: required)

        for i in 0..<n {
            let offset = i * overlap
            var chunk = Array(
                repeating: [Float](repeating: 0, count: chunkSize), count: channels
            )
            let upper = min(length, offset + chunkSize)
            for c in 0..<channels {
                for j in offset..<upper { chunk[c][j - offset] = mix[c][j] }
            }
            let res = try process(chunk)
            for s in 0..<sources {
                for c in 0..<channels {
                    for j in 0..<chunkSize {
                        acc[s][c][offset + j] += Double(res[s][c][j] * window[j])
                    }
                }
            }
            for j in 0..<chunkSize { wsum[offset + j] += Double(window[j]) }
        }
        return acc.map { stem in
            stem.map { ch in (0..<length).map { Float(ch[$0] / max(wsum[$0], 1e-8)) } }
        }
    }
}
