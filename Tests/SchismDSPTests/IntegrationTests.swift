import CoreML
import XCTest

@testable import SchismDSP
@testable import SchismPipeline

/// End-to-end integration: published fp32 Core ML core + SchismDSP
/// transforms + SchismPipeline orchestration on a whole track, compared to
/// the verified MLX references (`test_vectors_integration.npz`). Covers
/// every distinct host path: RoFormer single-res (18M), RoFormer dual-res
/// (V2), HTDemucs, AST windowing — htdemucs-ft/6s and CNN14 reuse these
/// exact paths.
///
/// Gated behind `SCHISM_DSP_INTEGRATION=1` — downloads ~600 MB of models on
/// first run (cached under Caches/schism-dsp/models). Runs the fp32 cores
/// under CPU+GPU (the verified configuration); expected stem residual is
/// the float32 host-DSP floor (~−70…−90 dB vs the references' float64
/// FFTs), gate at −60 dB — wiring bugs land at ≥ −30 dB.
///
/// The per-chunk closures below are also the reference host glue for each
/// model family's I/O contract.

// MARK: - model fetching (HF tree listing -> cached .mlpackage -> compiled)

private enum Models {
    static func fetch(repo: String, package: String) async throws -> MLModel {
        let cacheDir = FileManager.default.urls(
            for: .cachesDirectory, in: .userDomainMask
        )[0].appendingPathComponent("schism-dsp/models/\(repo)")
        try FileManager.default.createDirectory(
            at: cacheDir, withIntermediateDirectories: true
        )
        let compiled = cacheDir.appendingPathComponent(package + "c") // .mlmodelc
        if !FileManager.default.fileExists(atPath: compiled.path) {
            let pkg = cacheDir.appendingPathComponent(package)
            if !FileManager.default.fileExists(
                atPath: pkg.appendingPathComponent("Manifest.json").path
            ) {
                try download(repo: repo, package: package, to: pkg)
            }
            let tmp = try await MLModel.compileModel(at: pkg)
            _ = try? FileManager.default.replaceItemAt(compiled, withItemAt: tmp)
            if !FileManager.default.fileExists(atPath: compiled.path) {
                try FileManager.default.moveItem(at: tmp, to: compiled)
            }
        }
        let config = MLModelConfiguration()
        // the configuration the fp32 parity numbers were verified under
        // (coremltools CPU_AND_GPU); .cpuOnly runs some of these graphs on a
        // pathologically slow single-threaded path. Core ML falls back to
        // CPU where no usable GPU exists (e.g. virtualized CI runners).
        config.computeUnits = .cpuAndGPU
        return try MLModel(contentsOf: compiled, configuration: config)
    }

    /// List the .mlpackage's files via the HF tree API and download each.
    private static func download(repo: String, package: String, to dest: URL) throws {
        let api = URL(
            string:
                "https://huggingface.co/api/models/schism-audio/\(repo)/tree/main/\(package)?recursive=true"
        )!
        let entries =
            try JSONSerialization.jsonObject(with: Data(contentsOf: api))
            as? [[String: Any]] ?? []
        let files = entries.compactMap { e -> String? in
            (e["type"] as? String) == "file" ? e["path"] as? String : nil
        }
        guard !files.isEmpty else {
            throw NSError(
                domain: "schism-dsp", code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey: "no files listed for \(repo)/\(package)"
                ]
            )
        }
        for path in files {
            let escaped = path.split(separator: "/")
                .map {
                    $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed)!
                }
                .joined(separator: "/")
            let url = URL(
                string: "https://huggingface.co/schism-audio/\(repo)/resolve/main/\(escaped)"
            )!
            let local = dest.appendingPathComponent(
                String(path.dropFirst(package.count + 1))
            )
            try FileManager.default.createDirectory(
                at: local.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try Data(contentsOf: url).write(to: local)
        }
    }
}

private func predict(
    _ model: MLModel, _ inputs: [String: MLShapedArray<Float>]
) throws -> [String: MLShapedArray<Float>] {
    let provider = try MLDictionaryFeatureProvider(
        dictionary: inputs.mapValues { MLFeatureValue(multiArray: MLMultiArray($0)) }
    )
    let out = try model.prediction(from: provider)
    var result = [String: MLShapedArray<Float>]()
    for name in out.featureNames {
        if let arr = out.featureValue(for: name)?.multiArrayValue {
            // MLShapedArray yields logical C-order regardless of strides
            result[name] = MLShapedArray<Float>(arr)
        }
    }
    return result
}

// MARK: - comparison helpers

private func residualDB(_ got: [Float], _ want: [Float]) -> (db: Double, refRMS: Double) {
    var errSum = 0.0
    var refSum = 0.0
    for i in 0..<want.count {
        let e = Double(got[i]) - Double(want[i])
        errSum += e * e
        refSum += Double(want[i]) * Double(want[i])
    }
    let db = 10 * log10((errSum + .leastNormalMagnitude) / (refSum + .leastNormalMagnitude))
    return (db, (refSum / Double(want.count)).squareRoot())
}

/// Per-stem residual vs the reference; stems whose reference RMS clears the
/// energy floor gate at `maxDB`, near-silent stems are reported only (their
/// tiny denominator makes the ratio meaningless).
private func assertStemResiduals(
    _ got: [[[Float]]], want: NpyArray, order: [String], maxDB: Double,
    file: StaticString = #filePath, line: UInt = #line
) {
    let channels = want.shape[1]
    let samples = want.shape[2]
    let plane = channels * samples
    for s in 0..<want.shape[0] {
        let ref = Array(want.data[s * plane ..< (s + 1) * plane])
        let (db, rms) = residualDB(got[s].flatMap { $0 }, ref)
        print(
            String(
                format: "  %@: residual %.1f dB (ref RMS %.4f)%@",
                order[s], db, rms, rms < 1e-3 ? " [near-silent, not gated]" : ""
            )
        )
        if rms >= 1e-3 {
            XCTAssertLessThan(
                db, maxDB, "\(order[s]) residual", file: file, line: line
            )
        }
    }
}

private func integrationChannels(_ entry: NpyArray) -> [[Float]] {
    let samples = entry.shape[1]
    return (0..<entry.shape[0]).map {
        Array(entry.data[$0 * samples ..< ($0 + 1) * samples])
    }
}

private func skipUnlessIntegration() throws {
    try XCTSkipUnless(
        ProcessInfo.processInfo.environment["SCHISM_DSP_INTEGRATION"] == "1",
        "set SCHISM_DSP_INTEGRATION=1 to run the Core ML integration tests"
    )
}

// MARK: - tests

final class IntegrationTests: XCTestCase {
    func testRoformer18MFullTrack() async throws {
        try skipUnlessIntegration()
        let npz = try Vectors.load(
            repo: "mini-bs-roformer-18m-coreml", file: "test_vectors_integration.npz"
        )
        let mix = integrationChannels(try XCTUnwrap(npz["mix"])) // [2][441000]
        let want = try XCTUnwrap(npz["stems"]) // (4, 2, 441000)
        let model = try await Models.fetch(
            repo: "mini-bs-roformer-18m-coreml",
            package: "BSRoformer18M_Core_fp32.mlpackage"
        )

        let stems = try Separation.roformer(
            mix: mix, sources: 4, chunkSize: Separation.roformerChunkSize
        ) { chunk in
            // host glue per the model card: unnormalized STFT -> merged
            // layout -> mask -> multiply with the SAME STFT -> iSTFT
            let zs = chunk.map {
                STFT.forward($0, nFFT: 2048, hopLength: 512, normalized: false)
            }
            let frames = zs[0].frames // 690
            let merged = Roformer.merge(zs) // (690, 4100)
            let out = try predict(
                model,
                ["spec": MLShapedArray(scalars: merged, shape: [1, frames, 4100])]
            )
            let maskData = try XCTUnwrap(out["mask"]).scalars // (1, 4, 690, 4100)
            let per = frames * 4100
            return (0..<4).map { s in
                let stemMask = Array(maskData[s * per ..< (s + 1) * per])
                let masks = Roformer.unmerge(
                    stemMask, freqs: 1025, channels: 2, frames: frames
                )
                return (0..<2).map { c in
                    STFT.inverse(
                        Roformer.applyMask(zs[c], mask: masks[c]),
                        hopLength: 512, length: chunk[c].count, normalized: false
                    )
                }
            }
        }
        print("RoFormer 18M full track (fp32 core, CPU+GPU):")
        assertStemResiduals(
            stems, want: want, order: ["bass", "drums", "other", "vocals"],
            maxDB: -60
        )
    }

    /// V2 is the one sibling with a distinct host path: merged **analysis**
    /// STFT (n_fft 4096, 2049 freqs -> 8196-wide) in, mask at **synthesis**
    /// resolution (1025 freqs, 4100-wide) out, applied to a second n_fft-2048
    /// STFT of the same chunk — two STFTs per chunk.
    func testRoformerV2FullTrack() async throws {
        try skipUnlessIntegration()
        let npz = try Vectors.load(
            repo: "mini-bs-roformer-v2-coreml", file: "test_vectors_integration.npz"
        )
        let mix = integrationChannels(try XCTUnwrap(npz["mix"]))
        let want = try XCTUnwrap(npz["stems"])
        let model = try await Models.fetch(
            repo: "mini-bs-roformer-v2-coreml",
            package: "BSRoformerV2_Core_fp32.mlpackage"
        )

        let stems = try Separation.roformer(
            mix: mix, sources: 4, chunkSize: Separation.roformerChunkSize
        ) { chunk in
            let zAnalysis = chunk.map {
                STFT.forward($0, nFFT: 4096, hopLength: 512, normalized: false)
            }
            let zSynthesis = chunk.map {
                STFT.forward($0, nFFT: 2048, hopLength: 512, normalized: false)
            }
            let frames = zAnalysis[0].frames // 690
            let merged = Roformer.merge(zAnalysis) // (690, 2049*2*2 = 8196)
            let out = try predict(
                model,
                ["spec": MLShapedArray(scalars: merged, shape: [1, frames, 8196])]
            )
            let maskData = try XCTUnwrap(out["mask"]).scalars // (1, 4, 690, 4100)
            let per = frames * 4100
            return (0..<4).map { s in
                let stemMask = Array(maskData[s * per ..< (s + 1) * per])
                let masks = Roformer.unmerge(
                    stemMask, freqs: 1025, channels: 2, frames: frames
                )
                return (0..<2).map { c in
                    STFT.inverse(
                        Roformer.applyMask(zSynthesis[c], mask: masks[c]),
                        hopLength: 512, length: chunk[c].count, normalized: false
                    )
                }
            }
        }
        print("RoFormer V2 full track (fp32 core, CPU+GPU):")
        // V2 amplifies fp32 backend noise input-dependently (its torch-vs-
        // Core-ML gap alone reaches -66 dB on this audio, vs -132.7 dB on
        // the conversion-verification audio) — hence the looser gate; wiring
        // bugs still land at >= -30 dB
        assertStemResiduals(
            stems, want: want, order: ["bass", "drums", "other", "vocals"],
            maxDB: -50
        )
    }

    func testHTDemucsFullTrack() async throws {
        try skipUnlessIntegration()
        let npz = try Vectors.load(
            repo: "htdemucs-coreml", file: "test_vectors_integration.npz"
        )
        let mix = integrationChannels(try XCTUnwrap(npz["mix"]))
        let want = try XCTUnwrap(npz["stems"])
        let model = try await Models.fetch(
            repo: "htdemucs-coreml", package: "HTDemucs_Core_fp32.mlpackage"
        )

        let stems = try Separation.demucs(
            mix: mix, sources: 4, segmentLength: Separation.htdemucsSegmentLength
        ) { segment in
            // host glue per the model card: demucs-framed STFT as
            // complex-as-channels [c0re, c0im, c1re, c1im] + raw waveform in;
            // iSTFT(spec_out) + time_out = stems
            let zs = segment.map { Demucs.spec($0) } // 2 x (2048, 336)
            let freqs = zs[0].freqs
            let frames = zs[0].frames
            let plane = freqs * frames
            let samples = segment[0].count
            var mag = [Float]()
            mag.reserveCapacity(4 * plane)
            for c in 0..<2 {
                mag += zs[c].real
                mag += zs[c].imag
            }
            let out = try predict(
                model,
                [
                    "mag": MLShapedArray(scalars: mag, shape: [1, 4, freqs, frames]),
                    "mix": MLShapedArray(
                        scalars: segment[0] + segment[1], shape: [1, 2, samples]
                    ),
                ]
            )
            // .scalars materializes the full array — hoist out of the loops
            let specData = try XCTUnwrap(out["spec_out"]).scalars // (1, 4, 4, 2048, 336)
            let timeData = try XCTUnwrap(out["time_out"]).scalars // (1, 4, 2, 343980)
            return (0..<4).map { s in
                (0..<2).map { c in
                    let base = (s * 2 + c) * 2 * plane
                    let z = ComplexMatrix(
                        real: Array(specData[base ..< base + plane]),
                        imag: Array(specData[base + plane ..< base + 2 * plane]),
                        freqs: freqs, frames: frames
                    )
                    var wave = Demucs.ispec(z, length: samples)
                    let tBase = (s * 2 + c) * samples
                    for i in 0..<samples { wave[i] += timeData[tBase + i] }
                    return wave
                }
            }
        }
        print("HTDemucs full track (fp32 core, CPU+GPU):")
        assertStemResiduals(
            stems, want: want, order: ["drums", "bass", "other", "vocals"],
            maxDB: -60
        )
    }

    func testASTLongAudio() async throws {
        try skipUnlessIntegration()
        let npz = try Vectors.load(
            repo: "ast-audioset-10-10-coreml", file: "test_vectors_integration.npz"
        )
        let audio = try XCTUnwrap(npz["audio"]).data // (336240,)
        let wantLogits = try XCTUnwrap(npz["logits"]) // (4, 527)
        let wantMax = try XCTUnwrap(npz["agg_max"]).data
        let wantMean = try XCTUnwrap(npz["agg_mean"]).data
        let model = try await Models.fetch(
            repo: "ast-audioset-10-10-coreml", package: "AST_fp32.mlpackage"
        )

        let windows = Classification.astWindowedFeatures(audio)
        XCTAssertEqual(windows.count, wantLogits.shape[0])
        var scores = [[Float]]()
        var maxLogitDiff: Float = 0
        for (w, window) in windows.enumerated() {
            let out = try predict(
                model,
                ["features": MLShapedArray(scalars: window, shape: [1, 1024, 128])]
            )
            let logits = Array(try XCTUnwrap(out["logits"]).scalars) // (527,)
            let ref = Array(wantLogits.data[w * 527 ..< (w + 1) * 527])
            for i in 0..<527 { maxLogitDiff = max(maxLogitDiff, abs(logits[i] - ref[i])) }
            scores.append(Classification.sigmoid(logits))
        }
        let aggMax = Classification.aggregateMax(scores)
        let aggMean = Classification.aggregateMean(scores)
        let maxDiff = zip(aggMax, wantMax).map { abs($0 - $1) }.max() ?? 1
        let meanDiff = zip(aggMean, wantMean).map { abs($0 - $1) }.max() ?? 1
        print(
            String(
                format:
                    "AST long audio (fp32 core, CPU+GPU): max logit diff %.4f, "
                    + "agg score diff max %.5f / mean %.5f",
                maxLogitDiff, maxDiff, meanDiff
            )
        )
        // wiring bugs move scores grossly; fp32 core + float32 fbank drift
        // stays well inside 3e-2
        XCTAssertLessThan(maxDiff, 3e-2, "aggregated max scores")
        XCTAssertLessThan(meanDiff, 3e-2, "aggregated mean scores")
        // semantic check: same top-1 class
        let gotTop = aggMax.enumerated().max(by: { $0.element < $1.element })!.offset
        let refTop = wantMax.enumerated().max(by: { $0.element < $1.element })!.offset
        XCTAssertEqual(gotTop, refTop, "top-1 class")
    }
}
