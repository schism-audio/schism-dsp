import XCTest

@testable import SchismDSP

final class RealFFTTests: XCTestCase {
    func testRoundtrip() {
        for n in [512, 1024, 2048, 4096] {
            var rng = SystemRandomNumberGenerator()
            let x = (0..<n).map { _ in Float.random(in: -1...1, using: &rng) }
            let fft = RealFFT(length: n)
            let (re, im) = fft.forward(x)
            let back = fft.inverse(real: re, imag: im)
            assertClose(back, x, atol: 1e-5, rtol: 1e-5, "roundtrip n=\(n)")
        }
    }
}

final class RectWindowStftTests: XCTestCase {
    // torch.stft called WITHOUT a window (SCNet) is rectangular. No public
    // vectors exist for SCNet, so this is a pure-Swift consistency check:
    // rect-window forward/inverse must round-trip (overlap-add with a
    // constant window is exact wherever the window sum is nonzero), and the
    // rect path must actually differ from the default Hann path.
    func testRectangularRoundtrip() {
        let length = 5000
        var rng = SystemRandomNumberGenerator()
        let x = (0..<length).map { _ in Float.random(in: -1...1, using: &rng) }
        for (nFFT, hop) in [(1024, 256), (512, 160)] {
            let rect = [Float](repeating: 1, count: nFFT)
            for normalized in [false, true] {
                let z = STFT.forward(
                    x, nFFT: nFFT, hopLength: hop, normalized: normalized,
                    window: rect
                )
                let back = STFT.inverse(
                    z, hopLength: hop, length: length, normalized: normalized,
                    window: rect
                )
                assertClose(
                    back, x, atol: 1e-5, rtol: 1e-5,
                    "rect roundtrip n=\(nFFT) hop=\(hop) norm=\(normalized)"
                )
            }
            let hann = STFT.forward(x, nFFT: nFFT, hopLength: hop, normalized: false)
            let rectZ = STFT.forward(
                x, nFFT: nFFT, hopLength: hop, normalized: false, window: rect
            )
            var maxDiff: Float = 0
            for i in 0..<hann.real.count {
                maxDiff = max(maxDiff, abs(hann.real[i] - rectZ.real[i]))
            }
            XCTAssertGreaterThan(maxDiff, 1, "rect window must differ from Hann")
        }
    }
}

final class RoformerStftTests: XCTestCase {
    // unnormalized STFT (hop 512) at both resolutions + the merged layout,
    // vs schism-audio/mini-bs-roformer-v2-coreml test vectors
    func testUnnormalizedStftAndMergedLayout() throws {
        let npz = try Vectors.load(
            repo: "mini-bs-roformer-v2-coreml", file: "test_vectors_stft.npz"
        )
        let wav = try XCTUnwrap(npz["waveform"]) // (2, 44100)
        let samples = wav.shape[1]
        let channels = (0..<wav.shape[0]).map {
            Array(wav.data[$0 * samples ..< ($0 + 1) * samples])
        }

        for nFFT in [4096, 2048] {
            let re = try XCTUnwrap(npz["stft\(nFFT)_real"]) // (2, F, T)
            let im = try XCTUnwrap(npz["stft\(nFFT)_imag"])
            let (bins, frames) = (re.shape[1], re.shape[2])
            for (c, x) in channels.enumerated() {
                let z = STFT.forward(x, nFFT: nFFT, hopLength: 512, normalized: false)
                XCTAssertEqual(z.freqs, bins)
                XCTAssertEqual(z.frames, frames)
                let plane = bins * frames
                // raw magnitudes reach ~1e2 -> atol dominated by rtol scale
                assertClose(
                    z.real, Array(re.data[c * plane ..< (c + 1) * plane]),
                    atol: 2e-3, rtol: 1e-4, "stft\(nFFT) real ch\(c)"
                )
                assertClose(
                    z.imag, Array(im.data[c * plane ..< (c + 1) * plane]),
                    atol: 2e-3, rtol: 1e-4, "stft\(nFFT) imag ch\(c)"
                )
            }
        }

        // merged model-input layout: f outer, channel, re/im innermost
        let merged = try XCTUnwrap(npz["merged_input"]) // (T, F*C*2)
        let zs = channels.map {
            STFT.forward($0, nFFT: 4096, hopLength: 512, normalized: false)
        }
        let got = Roformer.merge(zs)
        XCTAssertEqual(got.count, merged.data.count)
        assertClose(got, merged.data, atol: 2e-3, rtol: 1e-4, "merged layout")

        // unmerge is the exact inverse
        let back = Roformer.unmerge(
            got, freqs: zs[0].freqs, channels: 2, frames: zs[0].frames
        )
        for c in 0..<2 {
            assertClose(back[c].real, zs[c].real, atol: 0, rtol: 0, "unmerge re ch\(c)")
            assertClose(back[c].imag, zs[c].imag, atol: 0, rtol: 0, "unmerge im ch\(c)")
        }
    }

    func testUnnormalizedIstft() throws {
        let npz = try Vectors.load(
            repo: "mini-bs-roformer-v2-coreml", file: "test_vectors_stft.npz"
        )
        let re = try XCTUnwrap(npz["ispec_input_real"]) // (2, 1025, 87)
        let im = try XCTUnwrap(npz["ispec_input_imag"])
        let want = try XCTUnwrap(npz["ispec_output"]) // (2, 44100)
        let (bins, frames) = (re.shape[1], re.shape[2])
        let plane = bins * frames
        let length = want.shape[1]
        for c in 0..<re.shape[0] {
            let z = ComplexMatrix(
                real: Array(re.data[c * plane ..< (c + 1) * plane]),
                imag: Array(im.data[c * plane ..< (c + 1) * plane]),
                freqs: bins, frames: frames
            )
            let y = STFT.inverse(z, hopLength: 512, length: length, normalized: false)
            assertClose(
                y, Array(want.data[c * length ..< (c + 1) * length]),
                atol: 1e-5, rtol: 1e-4, "istft ch\(c)"
            )
        }
    }
}
