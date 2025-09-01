import CoreML
import Foundation
import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class AsrManagerExtensionTests: XCTestCase {

    var manager: AsrManager!

    override func setUp() {
        super.setUp()
        manager = AsrManager(config: ASRConfig.default)
    }

    override func tearDown() {
        manager = nil
        super.tearDown()
    }

    // MARK: - padAudioIfNeeded Tests

    func testPadAudioIfNeededNopadding() {
        let audioSamples: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5]
        let targetLength = 3

        let result = manager.padAudioIfNeeded(audioSamples, targetLength: targetLength)

        // Should return original array when it's already longer than target
        XCTAssertEqual(result, audioSamples)
        XCTAssertEqual(result.count, 5)
    }

    func testPadAudioIfNeededWithPadding() {
        let audioSamples: [Float] = [0.1, 0.2, 0.3]
        let targetLength = 7

        let result = manager.padAudioIfNeeded(audioSamples, targetLength: targetLength)

        // Should pad with zeros
        let expected: [Float] = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]
        XCTAssertEqual(result, expected)
        XCTAssertEqual(result.count, targetLength)
    }

    func testPadAudioIfNeededExactLength() {
        let audioSamples: [Float] = [0.1, 0.2, 0.3]
        let targetLength = 3

        let result = manager.padAudioIfNeeded(audioSamples, targetLength: targetLength)

        // Should return original array when exactly the target length
        XCTAssertEqual(result, audioSamples)
        XCTAssertEqual(result.count, 3)
    }

    func testPadAudioIfNeededEmptyArray() {
        let audioSamples: [Float] = []
        let targetLength = 5

        let result = manager.padAudioIfNeeded(audioSamples, targetLength: targetLength)

        // Should return array of zeros
        let expected: [Float] = [0.0, 0.0, 0.0, 0.0, 0.0]
        XCTAssertEqual(result, expected)
        XCTAssertEqual(result.count, targetLength)
    }

    func testPadAudioIfNeededZeroTarget() {
        let audioSamples: [Float] = [0.1, 0.2]
        let targetLength = 0

        let result = manager.padAudioIfNeeded(audioSamples, targetLength: targetLength)

        // Should return original array when target is 0
        XCTAssertEqual(result, audioSamples)
        XCTAssertEqual(result.count, 2)
    }

    func testPadAudioIfNeededLargeArray() {
        let audioSamples: [Float] = Array(repeating: 0.5, count: 1000)
        let targetLength = 1500

        let result = manager.padAudioIfNeeded(audioSamples, targetLength: targetLength)

        XCTAssertEqual(result.count, targetLength)
        // First 1000 should be 0.5
        XCTAssertEqual(Array(result.prefix(1000)), audioSamples)
        // Last 500 should be 0.0
        XCTAssertEqual(Array(result.suffix(500)), Array(repeating: 0.0, count: 500))
    }

    // MARK: - calculateStartFrameOffset Tests

    func testCalculateStartFrameOffsetFirstSegment() {
        let offset = manager.calculateStartFrameOffset(segmentIndex: 0, leftContextSeconds: 2.0)

        // First segment should have no offset
        XCTAssertEqual(offset, 0)
    }

    func testCalculateStartFrameOffsetSecondSegment() {
        let leftContext = 2.0
        let offset = manager.calculateStartFrameOffset(segmentIndex: 1, leftContextSeconds: leftContext)

        // 2 seconds at 12.5 fps = 25 frames
        XCTAssertEqual(offset, 25)
    }

    func testCalculateStartFrameOffsetThirdSegment() {
        let leftContext = 1.5
        let offset = manager.calculateStartFrameOffset(segmentIndex: 2, leftContextSeconds: leftContext)

        // 1.5 seconds at 12.5 fps = 18.75, rounded to 19 frames
        XCTAssertEqual(offset, 19)
    }

    func testCalculateStartFrameOffsetVariousContexts() {
        // Test different left context durations
        let testCases: [(leftContext: Double, expected: Int)] = [
            (0.0, 0),  // No context
            (0.08, 1),  // One frame (80ms)
            (0.16, 2),  // Two frames
            (1.0, 13),  // 1 second = 12.5 frames, rounded to 13
            (3.2, 40),  // 3.2 seconds = 40 frames
        ]

        for (leftContext, expected) in testCases {
            let offset = manager.calculateStartFrameOffset(segmentIndex: 1, leftContextSeconds: leftContext)
            XCTAssertEqual(offset, expected, "Failed for leftContext=\(leftContext)")
        }
    }

    func testCalculateStartFrameOffsetNegativeSegment() {
        let offset = manager.calculateStartFrameOffset(segmentIndex: -1, leftContextSeconds: 2.0)

        // Negative segment should still return 0
        XCTAssertEqual(offset, 0)
    }

    // MARK: - Token Merging Tests

    private func makeToken(_ id: Int, _ start: Double) -> AlignedToken {
        AlignedToken(id: id, start: start, duration: 0.08)
    }

    func testMergeLongestContiguous() throws {
        let a = [makeToken(1, 0.0), makeToken(2, 0.08), makeToken(3, 0.16)]
        let b = [makeToken(3, 0.16), makeToken(4, 0.24)]
        let merged = try mergeLongestContiguous(a, b, overlapDuration: 0.16)
        XCTAssertEqual(merged.map { $0.id }, [1, 2, 3, 4])
    }

    func testMergeLongestCommonSubsequence() {
        let a = [makeToken(1, 0.0), makeToken(2, 0.08), makeToken(3, 0.16), makeToken(4, 0.24)]
        let b = [makeToken(2, 0.08), makeToken(99, 0.16), makeToken(3, 0.24), makeToken(4, 0.32), makeToken(5, 0.40)]
        let merged = mergeLongestCommonSubsequence(a, b, overlapDuration: 0.16)
        XCTAssertEqual(merged.map { $0.id }, [1, 2, 99, 3, 4, 5])
    }

    // MARK: - Performance Tests

    func testPadAudioPerformance() {
        let audioSamples = Array(repeating: Float(0.5), count: 100_000)
        let targetLength = 240_000  // 15 seconds at 16kHz

        measure {
            for _ in 0..<100 {
                _ = manager.padAudioIfNeeded(audioSamples, targetLength: targetLength)
            }
        }
    }

    func testCalculateStartFrameOffsetPerformance() {
        measure {
            for i in 0..<10_000 {
                _ = manager.calculateStartFrameOffset(segmentIndex: i % 100, leftContextSeconds: 2.0)
            }
        }
    }

    func testMergeLongestContiguousPerformance() {
        let frameDuration = 0.08
        let a = (0..<1000).map { makeToken($0, Double($0) * frameDuration) }
        let b = (500..<1500).map { makeToken($0, Double($0) * frameDuration) }
        measure {
            for _ in 0..<100 {
                _ = try? mergeLongestContiguous(a, b, overlapDuration: frameDuration * 2)
            }
        }
    }
}
