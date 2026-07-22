import Foundation

/// Minimal .npz / .npy reader for the schism test vectors.
///
/// `np.savez` writes a zip with *stored* (uncompressed) entries, so no
/// inflate is needed. Entries are located via the end-of-central-directory
/// record and the central directory (local headers written by numpy use
/// streaming data descriptors, so their size fields are unreliable); the
/// name/extra lengths are re-read from each *local* header before the data,
/// since local extra fields differ from central ones.
struct NpyArray {
    let shape: [Int]
    let data: [Float]
}

struct Npz {
    private var arrays: [String: NpyArray] = [:]

    subscript(name: String) -> NpyArray? { arrays[name] }

    init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        for (name, bytes) in try Self.zipEntries(data) {
            guard name.hasSuffix(".npy") else { continue }
            if let array = try Self.parseNpy(bytes) {
                arrays[String(name.dropLast(4))] = array
            }
        }
    }

    private static func u16(_ d: Data, _ o: Int) -> Int {
        Int(d[d.startIndex + o]) | Int(d[d.startIndex + o + 1]) << 8
    }

    private static func u32(_ d: Data, _ o: Int) -> Int {
        var v = 0
        for i in (0..<4).reversed() { v = v << 8 | Int(d[d.startIndex + o + i]) }
        return v
    }

    private static func zipEntries(_ data: Data) throws -> [(String, Data)] {
        // find end-of-central-directory (PK\x05\x06) scanning backwards
        var eocd = -1
        let start = max(0, data.count - 65558)
        var i = data.count - 22
        while i >= start {
            if data[data.startIndex + i] == 0x50, u32(data, i) == 0x06054B50 {
                eocd = i
                break
            }
            i -= 1
        }
        guard eocd >= 0 else { throw NpzError.badZip("no end-of-central-directory") }
        let count = u16(data, eocd + 10)
        var offset = u32(data, eocd + 16)

        var entries: [(String, Data)] = []
        for _ in 0..<count {
            guard u32(data, offset) == 0x02014B50 else {
                throw NpzError.badZip("bad central directory entry")
            }
            let method = u16(data, offset + 10)
            let csize = u32(data, offset + 20)
            let nameLen = u16(data, offset + 28)
            let extraLen = u16(data, offset + 30)
            let commentLen = u16(data, offset + 32)
            let localOffset = u32(data, offset + 42)
            let name = String(
                decoding: data.subdata(
                    in: data.startIndex + offset + 46 ..< data.startIndex + offset + 46 + nameLen
                ),
                as: UTF8.self
            )
            guard method == 0 else { throw NpzError.badZip("\(name): not stored (method \(method))") }
            // local header carries its own name/extra lengths — do not reuse
            // the central directory's
            guard u32(data, localOffset) == 0x04034B50 else {
                throw NpzError.badZip("\(name): bad local header")
            }
            let localName = u16(data, localOffset + 26)
            let localExtra = u16(data, localOffset + 28)
            let dataStart = localOffset + 30 + localName + localExtra
            entries.append(
                (
                    name,
                    data.subdata(
                        in: data.startIndex + dataStart ..< data.startIndex + dataStart + csize
                    )
                )
            )
            offset += 46 + nameLen + extraLen + commentLen
        }
        return entries
    }

    /// Parse one .npy blob. Returns nil for dtypes the vectors never use for
    /// numeric payloads (e.g. the '<U…' params strings).
    private static func parseNpy(_ d: Data) throws -> NpyArray? {
        guard d.count > 10, d[d.startIndex] == 0x93 else { throw NpzError.badNpy("magic") }
        let major = d[d.startIndex + 6]
        let headerLen: Int
        let headerStart: Int
        if major == 1 {
            headerLen = u16(d, 8)
            headerStart = 10
        } else {
            headerLen = u32(d, 8)
            headerStart = 12
        }
        let header = String(
            decoding: d.subdata(
                in: d.startIndex + headerStart ..< d.startIndex + headerStart + headerLen
            ),
            as: UTF8.self
        )
        guard let descr = extract(header, key: "descr") else { throw NpzError.badNpy(header) }
        guard header.contains("'fortran_order': False") else {
            throw NpzError.badNpy("fortran order unsupported")
        }
        let shape = extractShape(header)
        let count = shape.reduce(1, *)
        let payload = d.subdata(in: d.startIndex + headerStart + headerLen ..< d.endIndex)

        switch descr {
        case "<f4":
            let values = payload.withUnsafeBytes {
                Array($0.bindMemory(to: Float32.self).prefix(count))
            }
            return NpyArray(shape: shape, data: values)
        case "<f8":
            let values = payload.withUnsafeBytes {
                Array($0.bindMemory(to: Float64.self).prefix(count))
            }
            return NpyArray(shape: shape, data: values.map { Float($0) })
        default:
            return nil // params strings etc.
        }
    }

    private static func extract(_ header: String, key: String) -> String? {
        guard let r = header.range(of: "'\(key)': '") else { return nil }
        let rest = header[r.upperBound...]
        guard let end = rest.firstIndex(of: "'") else { return nil }
        return String(rest[..<end])
    }

    private static func extractShape(_ header: String) -> [Int] {
        guard let r = header.range(of: "'shape': (") else { return [] }
        let rest = header[r.upperBound...]
        guard let end = rest.firstIndex(of: ")") else { return [] }
        return rest[..<end].split(separator: ",").compactMap {
            Int($0.trimmingCharacters(in: .whitespaces))
        }
    }
}

enum NpzError: Error {
    case badZip(String)
    case badNpy(String)
}
