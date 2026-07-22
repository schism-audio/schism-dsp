import Foundation

/// Shared framing primitives, transliterated from `schism_mlx.audio`.
public enum DSP {
    /// Periodic Hann: `0.5 - 0.5*cos(2*pi*n / N)` — used by every STFT and
    /// the librosa-style logmel frontend.
    public static func hannPeriodic(_ n: Int) -> [Float] {
        (0..<n).map { 0.5 - 0.5 * cos(2 * Float.pi * Float($0) / Float(n)) }
    }

    /// Symmetric Hann: `0.5 - 0.5*cos(2*pi*n / (N-1))` — Kaldi's "hanning"
    /// window (fbank only). NOT interchangeable with the periodic variant.
    public static func hannSymmetric(_ n: Int) -> [Float] {
        (0..<n).map { 0.5 - 0.5 * cos(2 * Float.pi * Float($0) / Float(n - 1)) }
    }

    /// `numpy.pad(mode="reflect")`: mirrors *excluding* the edge sample —
    /// `[a,b,c,d]` with pad 2 becomes `[c,b,a,b,c,d,c,b]`.
    public static func reflectPad(_ x: [Float], left: Int, right: Int) -> [Float] {
        precondition(left < x.count && right < x.count, "reflect pad exceeds signal")
        var out = [Float]()
        out.reserveCapacity(left + x.count + right)
        for i in stride(from: left, through: 1, by: -1) { out.append(x[i]) }
        out.append(contentsOf: x)
        for i in 1...max(right, 1) where right > 0 { out.append(x[x.count - 1 - i]) }
        return out
    }

    /// `numpy.pad` with zeros on the right.
    public static func zeroPadRight(_ x: [Float], to length: Int) -> [Float] {
        x.count >= length ? x : x + [Float](repeating: 0, count: length - x.count)
    }
}

/// A complex spectrogram, `freqs` rows by `frames` columns, row-major
/// (frequency-major) — matching the `(F, T)` layout of the numpy references.
public struct ComplexMatrix {
    public var real: [Float]
    public var imag: [Float]
    public let freqs: Int
    public let frames: Int

    public init(freqs: Int, frames: Int) {
        self.freqs = freqs
        self.frames = frames
        self.real = [Float](repeating: 0, count: freqs * frames)
        self.imag = [Float](repeating: 0, count: freqs * frames)
    }

    public init(real: [Float], imag: [Float], freqs: Int, frames: Int) {
        precondition(real.count == freqs * frames && imag.count == freqs * frames)
        self.real = real
        self.imag = imag
        self.freqs = freqs
        self.frames = frames
    }

    @inlinable public func index(_ f: Int, _ t: Int) -> Int { f * frames + t }
}
