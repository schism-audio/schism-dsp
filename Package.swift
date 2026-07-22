// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "schism-dsp",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [
        .library(name: "SchismDSP", targets: ["SchismDSP"])
    ],
    targets: [
        .target(name: "SchismDSP"),
        .testTarget(name: "SchismDSPTests", dependencies: ["SchismDSP"]),
    ]
)
