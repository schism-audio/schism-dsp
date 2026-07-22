import Foundation
import SchismDSP

/// Long-audio windowed classification for AST, mirroring
/// `ASTForAudioClassification.classify` in schism-mlx: the full-file fbank
/// is sliced into overlapping fixed-size windows, each window is classified,
/// and per-class sigmoid scores are aggregated across windows.
///
/// (CNN14 needs none of this — it accepts any input length natively.)
public enum Classification {
    /// AST model input length in fbank frames (~10.24 s at 16 kHz).
    public static let astMaxFrames = 1024
    /// Default window hop in fbank frames (50% overlap).
    public static let astHopFrames = 512

    static let astMean: Float = -4.2677393
    static let astStd: Float = 4.5689974

    /// Slice a raw (unnormalized) fbank into overlapping normalized model
    /// inputs. Windows start every `hopFrames`; a final end-aligned window
    /// at `frames - maxFrames` is added if the stride didn't land there.
    /// Audio of `maxFrames` frames or fewer yields one window, zero-padded
    /// in the **raw** domain before normalization — padding rows become
    /// `(0 + 4.2677393) / (2 * 4.5689974)` ≈ 0.467, exactly like the
    /// reference feature extractor.
    ///
    /// - Parameters:
    ///   - fbank: `(frames, bins)` row-major raw fbank (`KaldiFbank.fbank`).
    ///   - frames: number of fbank rows.
    /// - Returns: windows of `maxFrames * bins` floats, row-major.
    public static func astWindows(
        fbank: [Float], frames: Int, bins: Int = 128,
        maxFrames: Int = astMaxFrames, hopFrames: Int = astHopFrames
    ) -> [[Float]] {
        func normalize(_ x: Float) -> Float { (x - astMean) / (2 * astStd) }
        if frames <= maxFrames {
            var window = [Float](repeating: normalize(0), count: maxFrames * bins)
            for i in 0..<(frames * bins) { window[i] = normalize(fbank[i]) }
            return [window]
        }
        var starts = Array(stride(from: 0, through: frames - maxFrames, by: hopFrames))
        if starts.last != frames - maxFrames { starts.append(frames - maxFrames) }
        return starts.map { start in
            (0..<(maxFrames * bins)).map { normalize(fbank[start * bins + $0]) }
        }
    }

    /// Full path from a mono waveform at 16 kHz: `KaldiFbank.fbank` →
    /// `astWindows`. Feed each window to the model as `(1, 1024, 128)`.
    public static func astWindowedFeatures(
        _ waveform: [Float], hopFrames: Int = astHopFrames
    ) -> [[Float]] {
        let (frames, fbank) = KaldiFbank.fbank(waveform)
        return astWindows(fbank: fbank, frames: frames, hopFrames: hopFrames)
    }

    /// Elementwise logistic sigmoid — AST logits → multi-label scores.
    public static func sigmoid(_ logits: [Float]) -> [Float] {
        logits.map { Float(1.0 / (1.0 + exp(-Double($0)))) }
    }

    /// Per-class max across windows: "did this sound occur anywhere?"
    public static func aggregateMax(_ scores: [[Float]]) -> [Float] {
        var out = scores[0]
        for window in scores.dropFirst() {
            for i in 0..<out.count { out[i] = max(out[i], window[i]) }
        }
        return out
    }

    /// Per-class mean across windows: "how present is it overall?"
    public static func aggregateMean(_ scores: [[Float]]) -> [Float] {
        var acc = [Double](repeating: 0, count: scores[0].count)
        for window in scores {
            for i in 0..<acc.count { acc[i] += Double(window[i]) }
        }
        return acc.map { Float($0 / Double(scores.count)) }
    }
}
