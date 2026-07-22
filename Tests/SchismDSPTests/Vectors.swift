import Foundation
import XCTest

/// Loads the published `test_vectors_*.npz` from the schism-audio Core ML
/// repos on Hugging Face (cached under Caches/schism-dsp). Set
/// `SCHISM_DSP_VECTORS_DIR` to a directory laid out as `<repo>/<file>` (e.g.
/// a schism-mlx `build/` tree) to run offline against local vectors.
enum Vectors {
    static func load(repo: String, file: String) throws -> Npz {
        if let dir = ProcessInfo.processInfo.environment["SCHISM_DSP_VECTORS_DIR"] {
            return try Npz(
                contentsOf: URL(fileURLWithPath: dir)
                    .appendingPathComponent(repo)
                    .appendingPathComponent(file)
            )
        }
        let cacheDir = FileManager.default.urls(
            for: .cachesDirectory, in: .userDomainMask
        )[0].appendingPathComponent("schism-dsp")
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let cached = cacheDir.appendingPathComponent("\(repo)--\(file)")
        if !FileManager.default.fileExists(atPath: cached.path) {
            let url = URL(
                string: "https://huggingface.co/schism-audio/\(repo)/resolve/main/\(file)"
            )!
            let data = try Data(contentsOf: url)
            try data.write(to: cached)
        }
        return try Npz(contentsOf: cached)
    }
}

/// `numpy.allclose`-style assertion: `|a - b| <= atol + rtol * |b|`
/// elementwise, reporting the worst offender.
func assertClose(
    _ got: [Float], _ want: [Float], atol: Float, rtol: Float,
    _ label: String, file: StaticString = #filePath, line: UInt = #line
) {
    XCTAssertEqual(got.count, want.count, "\(label): count", file: file, line: line)
    var worst: Float = 0
    var worstIdx = 0
    for i in 0..<min(got.count, want.count) {
        let excess = abs(got[i] - want[i]) - (atol + rtol * abs(want[i]))
        if excess > worst {
            worst = excess
            worstIdx = i
        }
    }
    XCTAssertLessThanOrEqual(
        worst, 0,
        "\(label): worst at [\(worstIdx)] got \(got[worstIdx]) want \(want[worstIdx])",
        file: file, line: line
    )
}
