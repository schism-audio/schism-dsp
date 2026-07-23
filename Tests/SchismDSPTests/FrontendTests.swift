import XCTest

@testable import SchismDSP

final class KaldiFbankTests: XCTestCase {
    func testFbankAndAstNormalization() throws {
        let npz = try Vectors.load(
            repo: "ast-audioset-10-10-coreml", file: "test_vectors_fbank.npz"
        )
        let wav = try XCTUnwrap(npz["waveform"]) // (32000,)
        let want = try XCTUnwrap(npz["fbank"]) // (frames, 128)
        let (frames, got) = KaldiFbank.fbank(wav.data)
        XCTAssertEqual(frames, want.shape[0])
        // log-domain values (~-16..+5); vector computed with float64 FFT
        assertClose(got, want.data, atol: 2e-3, rtol: 1e-4, "fbank")

        let wantNorm = try XCTUnwrap(npz["normalized"])
        let norm = got.map { ($0 + 4.2677393) / (2 * 4.5689974) }
        assertClose(norm, wantNorm.data, atol: 2e-4, rtol: 1e-4, "ast normalization")

        // astFeatures pads to the model's fixed 1024 frames — in the RAW
        // domain, so padding rows normalize to ~0.467 (see the pipeline
        // vectors' short_windows for the reference-pinned value)
        let features = KaldiFbank.astFeatures(wav.data)
        XCTAssertEqual(features.count, 1024 * 128)
        assertClose(
            Array(features[0..<(frames * 128)]), wantNorm.data,
            atol: 2e-4, rtol: 1e-4, "astFeatures body"
        )
        let pad: Float = (0 + 4.2677393) / (2 * 4.5689974)
        XCTAssertTrue(features[(frames * 128)...].allSatisfy { $0 == pad })
    }

    func testShortInputIsEmpty() {
        let (frames, data) = KaldiFbank.fbank([Float](repeating: 0, count: 100))
        XCTAssertEqual(frames, 0)
        XCTAssertTrue(data.isEmpty)
    }
}

final class EfficientATMelTests: XCTestCase {
    func testEfficientATMel() throws {
        let npz = try Vectors.load(
            repo: "efficient-at-mn10-coreml", file: "test_vectors_logmel.npz"
        )
        let wav = try XCTUnwrap(npz["waveform"]) // (64000,)
        let want = try XCTUnwrap(npz["logmel"]) // (200, 128)
        let (frames, got) = EfficientATMel.compute(wav.data)
        XCTAssertEqual(frames, want.shape[0])
        // log-domain but divided by 5, and the values cross zero after the
        // (x + 4.5) / 5 shift, so atol dominates; vector computed with a
        // float64 FFT — observed worst |diff| 5.8e-5
        assertClose(got, want.data, atol: 2e-4, rtol: 1e-4, "efficient-at logmel")
    }

    func testTenSecondsIsExactly1000Frames() {
        let (frames, data) = EfficientATMel.compute(
            [Float](repeating: 0, count: 320000)
        )
        XCTAssertEqual(frames, 1000)
        XCTAssertEqual(data.count, 1000 * 128)
        // silence: mel = 0 -> (ln(1e-5) + 4.5) / 5 everywhere
        let silent = (log(Float(1e-5)) + 4.5) / 5
        XCTAssertTrue(data.allSatisfy { abs($0 - silent) < 1e-6 })
    }
}

final class LogMelTests: XCTestCase {
    func testLogMel() throws {
        let npz = try Vectors.load(
            repo: "cnn14-audioset-coreml", file: "test_vectors_logmel.npz"
        )
        let wav = try XCTUnwrap(npz["waveform"]) // (64000,)
        let want = try XCTUnwrap(npz["logmel"]) // (201, 64)
        let (frames, got) = LogMel.compute(wav.data)
        XCTAssertEqual(frames, want.shape[0])
        assertClose(got, want.data, atol: 2e-3, rtol: 1e-4, "logmel")
    }
}

final class DemucsTests: XCTestCase {
    func testNormalizedStftIstft() throws {
        let npz = try Vectors.load(repo: "htdemucs-coreml", file: "test_vectors_stft.npz")
        let wav = try XCTUnwrap(npz["waveform"]) // (2, 44100)
        let re = try XCTUnwrap(npz["stft_real"]) // (2, 2049, T)
        let im = try XCTUnwrap(npz["stft_imag"])
        let samples = wav.shape[1]
        let (bins, frames) = (re.shape[1], re.shape[2])
        let plane = bins * frames
        for c in 0..<2 {
            let x = Array(wav.data[c * samples ..< (c + 1) * samples])
            let z = STFT.forward(x, nFFT: 4096, hopLength: 1024, normalized: true)
            XCTAssertEqual(z.freqs, bins)
            XCTAssertEqual(z.frames, frames)
            // normalized scale ~1e-2 -> atol can be tight
            assertClose(
                z.real, Array(re.data[c * plane ..< (c + 1) * plane]),
                atol: 1e-5, rtol: 1e-4, "stft real ch\(c)"
            )
            assertClose(
                z.imag, Array(im.data[c * plane ..< (c + 1) * plane]),
                atol: 1e-5, rtol: 1e-4, "stft imag ch\(c)"
            )
        }

        let ire = try XCTUnwrap(npz["ispec_input_real"]) // (2, 2049, 40)
        let iim = try XCTUnwrap(npz["ispec_input_imag"])
        let want = try XCTUnwrap(npz["ispec_output"]) // (2, 44100)
        let iplane = ire.shape[1] * ire.shape[2]
        for c in 0..<2 {
            let z = ComplexMatrix(
                real: Array(ire.data[c * iplane ..< (c + 1) * iplane]),
                imag: Array(iim.data[c * iplane ..< (c + 1) * iplane]),
                freqs: ire.shape[1], frames: ire.shape[2]
            )
            let y = STFT.inverse(z, hopLength: 1024, length: samples, normalized: true)
            assertClose(
                y, Array(want.data[c * samples ..< (c + 1) * samples]),
                atol: 1e-5, rtol: 1e-4, "istft ch\(c)"
            )
        }
    }

    func testSegmentFraming() throws {
        let npz = try Vectors.load(repo: "htdemucs-coreml", file: "test_vectors_stft.npz")
        let wav = try XCTUnwrap(npz["waveform"]) // (2, 44100)
        let re = try XCTUnwrap(npz["spec_framed_real"]) // (2, 2048, 44)
        let im = try XCTUnwrap(npz["spec_framed_imag"])
        let samples = wav.shape[1]
        let (bins, frames) = (re.shape[1], re.shape[2])
        let plane = bins * frames
        for c in 0..<2 {
            let x = Array(wav.data[c * samples ..< (c + 1) * samples])
            let z = Demucs.spec(x)
            XCTAssertEqual(z.freqs, bins) // Nyquist dropped: 2048
            XCTAssertEqual(z.frames, frames) // ceil(L/hop): 44
            assertClose(
                z.real, Array(re.data[c * plane ..< (c + 1) * plane]),
                atol: 1e-5, rtol: 1e-4, "spec real ch\(c)"
            )
            assertClose(
                z.imag, Array(im.data[c * plane ..< (c + 1) * plane]),
                atol: 1e-5, rtol: 1e-4, "spec imag ch\(c)"
            )
        }

        let ire = try XCTUnwrap(npz["ispec_framed_input_real"]) // (2, 2048, 44)
        let iim = try XCTUnwrap(npz["ispec_framed_input_imag"])
        let want = try XCTUnwrap(npz["ispec_framed_output"]) // (2, 44100)
        for c in 0..<2 {
            let z = ComplexMatrix(
                real: Array(ire.data[c * plane ..< (c + 1) * plane]),
                imag: Array(iim.data[c * plane ..< (c + 1) * plane]),
                freqs: bins, frames: frames
            )
            let y = Demucs.ispec(z, length: samples)
            assertClose(
                y, Array(want.data[c * samples ..< (c + 1) * samples]),
                atol: 1e-5, rtol: 1e-4, "ispec ch\(c)"
            )
        }
    }
}
