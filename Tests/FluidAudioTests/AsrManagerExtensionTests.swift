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

    // MARK: - mergeTokens Tests

    func testMergeTokensNoDuplicates() {
        let previous = [1, 2, 3, 4, 5]
        let current = [6, 7, 8, 9, 10]

        let (merged, removedCount) = manager.mergeTokens(previous: previous, current: current)
        let deduped = Array(merged.dropFirst(previous.count))

        XCTAssertEqual(deduped, current, "No duplicates should return original current")
        XCTAssertEqual(removedCount, 0, "Should not remove any tokens")
    }

    func testMergeTokensWithOverlap() {
        let previous = [1, 2, 3, 4, 5]
        let current = [3, 4, 5, 6, 7, 8]

        let (merged, removedCount) = manager.mergeTokens(previous: previous, current: current)
        let deduped = Array(merged.dropFirst(previous.count))

        XCTAssertEqual(deduped, [6, 7, 8], "Should remove overlapping tokens")
        XCTAssertEqual(removedCount, 3, "Should remove 3 tokens")
    }

    func testMergeTokensPunctuation() {
        let previous = [1, 2, 3, 7883]  // 7883 is period
        let current = [7883, 4, 5, 6]

        let (merged, removedCount) = manager.mergeTokens(previous: previous, current: current)
        let deduped = Array(merged.dropFirst(previous.count))

        XCTAssertEqual(deduped, [4, 5, 6], "Should remove duplicate punctuation")
        XCTAssertEqual(removedCount, 1, "Should remove 1 punctuation token")
    }

    func testMergeTokensQuestionMark() {
        let previous = [1, 2, 7952]  // 7952 is question mark
        let current = [7952, 3, 4]

        let (merged, removedCount) = manager.mergeTokens(previous: previous, current: current)
        let deduped = Array(merged.dropFirst(previous.count))

        XCTAssertEqual(deduped, [3, 4], "Should remove duplicate question mark")
        XCTAssertEqual(removedCount, 1, "Should remove 1 token")
    }

    func testMergeTokensExclamation() {
        let previous = [1, 2, 7948]  // 7948 is exclamation
        let current = [7948, 3, 4]

        let (merged, removedCount) = manager.mergeTokens(previous: previous, current: current)
        let deduped = Array(merged.dropFirst(previous.count))

        XCTAssertEqual(deduped, [3, 4], "Should remove duplicate exclamation")
        XCTAssertEqual(removedCount, 1, "Should remove 1 token")
    }

    func testMergeTokensEmptyPrevious() {
        let previous: [Int] = []
        let current = [1, 2, 3, 4]

        let (merged, removedCount) = manager.mergeTokens(previous: previous, current: current)
        let deduped = Array(merged.dropFirst(previous.count))

        XCTAssertEqual(deduped, current, "Empty previous should return original current")
        XCTAssertEqual(removedCount, 0, "Should not remove any tokens")
    }

    func testMergeTokensEmptyCurrent() {
        let previous = [1, 2, 3, 4]
        let current: [Int] = []

        let (merged, removedCount) = manager.mergeTokens(previous: previous, current: current)
        let deduped = Array(merged.dropFirst(previous.count))

        XCTAssertEqual(deduped, current, "Empty current should return empty array")
        XCTAssertEqual(removedCount, 0, "Should not remove any tokens")
    }

    func testMergeTokensShortSequences() {
        let previous = [1, 2]
        let current = [2, 3]

        let (merged, removedCount) = manager.mergeTokens(previous: previous, current: current)
        let deduped = Array(merged.dropFirst(previous.count))

        XCTAssertEqual(deduped, [3], "Should remove overlapping token")
        XCTAssertEqual(removedCount, 1, "Should remove 1 token")
    }

    func testMergeTokensMaxOverlap() {
        // Test with overlap longer than default maxOverlap (12)
        let previous = Array(1...20)
        let current = Array(10...25)  // 10-21 overlaps with previous

        let (merged, removedCount) = manager.mergeTokens(previous: previous, current: current)
        let deduped = Array(merged.dropFirst(previous.count))

        XCTAssertGreaterThan(removedCount, 0, "Should find some overlap")
        XCTAssertLessThanOrEqual(removedCount, 12, "Should not exceed maxOverlap")
        XCTAssertEqual(deduped, Array(21...25), "Should append new tokens")
    }

    func testMergeTokensPartialOverlap() {
        let previous = [10, 11, 12, 13, 14]
        let current = [12, 13, 14, 15, 16]

        let (merged, removedCount) = manager.mergeTokens(previous: previous, current: current)
        let deduped = Array(merged.dropFirst(previous.count))

        XCTAssertEqual(deduped, [15, 16], "Should remove partial overlap")
        XCTAssertEqual(removedCount, 3, "Should remove 3 overlapping tokens")
    }

    func testMergeTokensCustomMaxOverlap() {
        let previous = [1, 2, 3, 4, 5, 6]
        let current = [4, 5, 6, 7, 8, 9]

        let (merged, removedCount) = manager.mergeTokens(
            previous: previous, current: current, maxOverlap: 2
        )
        let deduped = Array(merged.dropFirst(previous.count))

        XCTAssertEqual(deduped, [6, 7, 8, 9], "Should not find overlap with small maxOverlap \(deduped)")
        XCTAssertEqual(removedCount, 2, "Should remove 2 tokens with small maxOverlap")
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

    func testMergeTokensPerformance() {
        let previous = Array(1...1000)
        let current = Array(990...1500)  // 10-token overlap

        measure {
            for _ in 0..<1000 {
                _ = manager.mergeTokens(previous: previous, current: current)
            }
        }
    }
}
