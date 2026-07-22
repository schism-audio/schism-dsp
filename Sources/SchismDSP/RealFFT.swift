import Accelerate

/// Real forward/inverse DFT matching `numpy.fft.rfft` / `numpy.fft.irfft`.
///
/// Built on `vDSP_DFT_zop` (full complex DFT, unscaled forward) for clarity:
/// no split-complex packing and no vDSP real-FFT 2x scaling to compensate.
/// Supported lengths are f * 2^n with f in {1, 3, 5, 15} — every length used
/// by the schism frontends (512, 1024, 2048, 4096) qualifies.
public final class RealFFT {
    public let length: Int
    private let forwardSetup: vDSP_DFT_Setup
    private let inverseSetup: vDSP_DFT_Setup
    private var zeros: [Float]

    public init(length: Int) {
        guard
            let f = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(length), .FORWARD),
            let i = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(length), .INVERSE)
        else {
            fatalError("vDSP_DFT does not support length \(length)")
        }
        self.length = length
        self.forwardSetup = f
        self.inverseSetup = i
        self.zeros = [Float](repeating: 0, count: length)
    }

    deinit {
        vDSP_DFT_DestroySetup(forwardSetup)
        vDSP_DFT_DestroySetup(inverseSetup)
    }

    /// `numpy.fft.rfft(x)`: real signal of `length` samples -> `length/2 + 1`
    /// complex bins (unscaled).
    public func forward(_ x: [Float]) -> (real: [Float], imag: [Float]) {
        precondition(x.count == length)
        let bins = length / 2 + 1
        var outR = [Float](repeating: 0, count: length)
        var outI = [Float](repeating: 0, count: length)
        vDSP_DFT_Execute(forwardSetup, x, zeros, &outR, &outI)
        return (Array(outR[0..<bins]), Array(outI[0..<bins]))
    }

    /// `numpy.fft.irfft(z, n: length)`: `length/2 + 1` complex bins -> real
    /// signal. Mirrors numpy exactly: the imaginary parts of the DC and
    /// Nyquist bins are ignored, the upper half is conjugate-reconstructed,
    /// and the result carries the 1/N inverse scaling.
    public func inverse(real: [Float], imag: [Float]) -> [Float] {
        let bins = length / 2 + 1
        precondition(real.count == bins && imag.count == bins)
        var fullR = [Float](repeating: 0, count: length)
        var fullI = [Float](repeating: 0, count: length)
        fullR[0] = real[0] // Im(DC) discarded, per numpy irfft
        for k in 1..<(length / 2) {
            fullR[k] = real[k]
            fullI[k] = imag[k]
            fullR[length - k] = real[k]
            fullI[length - k] = -imag[k]
        }
        fullR[length / 2] = real[length / 2] // Im(Nyquist) discarded
        var outR = [Float](repeating: 0, count: length)
        var outI = [Float](repeating: 0, count: length)
        vDSP_DFT_Execute(inverseSetup, fullR, fullI, &outR, &outI)
        var scale = Float(1) / Float(length)
        vDSP_vsmul(outR, 1, &scale, &outR, 1, vDSP_Length(length))
        return outR
    }
}
