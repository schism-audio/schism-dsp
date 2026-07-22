import XCTest

@testable import SchismDSP
@testable import SchismPipeline

/// The deterministic mock model from `make_pipeline_vectors.py`: a 2-tap FIR
/// plus a position ramp, dyadic float32 constants —
/// `out[s,c,i] = A[s]*x[i] + D[s]*x[i-1] + B[s]*(i/(L-1))`, `x[-1] = 0`.
/// The FIR tap makes the demucs centered-padding *pull* observable: a
/// pipeline that zero-pads segment edges instead of pulling real samples
/// from the surrounding track fails the vectors.
private let stemA: [Float] = [0.5, 0.75, 1.0, 1.25]
private let stemD: [Float] = [0.25, 0.1875, 0.125, 0.0625]
private let stemB: [Float] = [0.125, 0.25, 0.375, 0.5]

private func mockSeparator(_ chunk: [[Float]]) -> [[[Float]]] {
    let length = chunk[0].count
    let ramp = (0..<length).map { Float($0) / Float(length - 1) }
    return (0..<4).map { s in
        chunk.map { x in
            (0..<length).map { i in
                let prev: Float = i > 0 ? x[i - 1] : 0
                return stemA[s] * x[i] + stemD[s] * prev + stemB[s] * ramp[i]
            }
        }
    }
}

private func channels(_ entry: NpyArray) -> [[Float]] {
    let samples = entry.shape[1]
    return (0..<entry.shape[0]).map {
        Array(entry.data[$0 * samples ..< ($0 + 1) * samples])
    }
}

final class SeparationPipelineTests: XCTestCase {
    func testDemucsSegmented() throws {
        let npz = try Vectors.load(repo: "htdemucs-coreml", file: "test_vectors_pipeline.npz")
        for (mixKey, stemsKey) in [("mix", "stems"), ("short_mix", "short_stems")] {
            let mix = channels(try XCTUnwrap(npz[mixKey]))
            let want = try XCTUnwrap(npz[stemsKey]) // (4, 2, L)
            let got = Separation.demucs(
                mix: mix, sources: 4, segmentLength: 3510, process: mockSeparator
            )
            assertClose(
                got.flatMap { $0.flatMap { $0 } }, want.data,
                atol: 1e-6, rtol: 1e-5, "demucs \(stemsKey)"
            )
        }
    }

    func testRoformerChunked() throws {
        let npz = try Vectors.load(
            repo: "mini-bs-roformer-18m-coreml", file: "test_vectors_pipeline.npz"
        )
        let mix = channels(try XCTUnwrap(npz["mix"]))
        for (gap, key) in [(0, "stems_gap0"), (300, "stems_gap300")] {
            let want = try XCTUnwrap(npz[key]) // (4, 2, 8000)
            let got = Separation.roformer(
                mix: mix, sources: 4, chunkSize: 3000, gapSize: gap,
                process: mockSeparator
            )
            assertClose(
                got.flatMap { $0.flatMap { $0 } }, want.data,
                atol: 1e-6, rtol: 1e-5, "roformer \(key)"
            )
        }
        // gap zeroes the window edges -> the track head has no fold weight
        let gapped = Separation.roformer(
            mix: mix, sources: 4, chunkSize: 3000, gapSize: 300,
            process: mockSeparator
        )
        XCTAssertTrue(gapped[0][0][..<300].allSatisfy { $0 == 0 })

        // single-chunk track (shorter than chunkSize)
        let singleMix = channels(try XCTUnwrap(npz["single_mix"]))
        let wantSingle = try XCTUnwrap(npz["single_stems"])
        let gotSingle = Separation.roformer(
            mix: singleMix, sources: 4, chunkSize: 3000, process: mockSeparator
        )
        assertClose(
            gotSingle.flatMap { $0.flatMap { $0 } }, wantSingle.data,
            atol: 1e-6, rtol: 1e-5, "roformer single chunk"
        )
    }

    func testBagCombine() {
        // two models, S=2, C=1, L=3; hand-computed weighted per-stem average
        let e0: [[[Float]]] = [[[1, 2, 3]], [[4, 5, 6]]]
        let e1: [[[Float]]] = [[[3, 2, 1]], [[0, 0, 0]]]
        let got = Separation.bagCombine([e0, e1], weights: [[1, 3], [1, 1]])
        assertClose(got[0][0], [2, 2, 2], atol: 1e-7, rtol: 0, "bag stem0")
        assertClose(got[1][0], [3, 3.75, 4.5], atol: 1e-7, rtol: 0, "bag stem1")

        // one-hot weights (htdemucs_ft): stem k comes from model k
        let oneHot = Separation.bagCombine([e0, e1], weights: [[1, 0], [0, 1]])
        assertClose(oneHot[0][0], [1, 2, 3], atol: 0, rtol: 0, "one-hot stem0")
        assertClose(oneHot[1][0], [0, 0, 0], atol: 0, rtol: 0, "one-hot stem1")
    }
}

final class ClassificationPipelineTests: XCTestCase {
    func testAstWindows() throws {
        let npz = try Vectors.load(
            repo: "ast-audioset-10-10-coreml", file: "test_vectors_pipeline.npz"
        )
        // exact windowing over the reference-computed fbank
        for (fbKey, winKey) in [("fbank", "windows"), ("short_fbank", "short_windows")] {
            let fbank = try XCTUnwrap(npz[fbKey]) // (frames, 128)
            let want = try XCTUnwrap(npz[winKey]) // (W, 1024, 128)
            let got = Classification.astWindows(
                fbank: fbank.data, frames: fbank.shape[0]
            )
            XCTAssertEqual(got.count, want.shape[0], "\(winKey) count")
            assertClose(
                got.flatMap { $0 }, want.data, atol: 1e-5, rtol: 1e-5, winKey
            )
        }

        // composed path: waveform -> own fbank -> windows (fbank tolerance)
        let audio = try XCTUnwrap(npz["audio"])
        let want = try XCTUnwrap(npz["windows"])
        let got = Classification.astWindowedFeatures(audio.data)
        XCTAssertEqual(got.count, want.shape[0])
        assertClose(
            got.flatMap { $0 }, want.data, atol: 5e-4, rtol: 1e-4,
            "windows from waveform"
        )
    }

    func testAstAggregation() throws {
        let npz = try Vectors.load(
            repo: "ast-audioset-10-10-coreml", file: "test_vectors_pipeline.npz"
        )
        let scoresEntry = try XCTUnwrap(npz["scores"]) // (W, 8)
        let classCount = scoresEntry.shape[1]
        let scores = (0..<scoresEntry.shape[0]).map {
            Array(scoresEntry.data[$0 * classCount ..< ($0 + 1) * classCount])
        }
        let wantMax = try XCTUnwrap(npz["agg_max"])
        let wantMean = try XCTUnwrap(npz["agg_mean"])
        assertClose(
            Classification.aggregateMax(scores), wantMax.data,
            atol: 1e-7, rtol: 0, "aggregate max"
        )
        assertClose(
            Classification.aggregateMean(scores), wantMean.data,
            atol: 1e-6, rtol: 0, "aggregate mean"
        )
        // sigmoid sanity: monotone, symmetric around 0.5
        let sig = Classification.sigmoid([-1, 0, 1])
        XCTAssertEqual(sig[1], 0.5)
        XCTAssertEqual(sig[0] + sig[2], 1.0, accuracy: 1e-6)
    }
}
