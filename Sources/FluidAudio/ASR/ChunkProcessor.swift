import CoreML
import Foundation
import OSLog

struct ChunkProcessor {
    let audioSamples: [Float]
    let enableDebug: Bool

    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "ChunkProcessor")

    // Frame-aligned configuration: 8 + 3.2 + 3.2 seconds context at 16kHz
    // 8s center = exactly 100 encoder frames
    // 3.2s context = exactly 40 encoder frames each
    // Total: 14.4s (within 15s model limit, 180 total frames)
    private let sampleRate: Int = 16000
    private let centerSeconds: Double = 11.2  // Exactly 100 frames (8.0 * 12.5)
    private let leftContextSeconds: Double = 1.6  // Exactly 40 frames (3.2 * 12.5)
    private let rightContextSeconds: Double = 1.6  // Exactly 40 frames (3.2 * 12.5)

    private var centerSamples: Int { Int(centerSeconds * Double(sampleRate)) }
    private var leftContextSamples: Int { Int(leftContextSeconds * Double(sampleRate)) }
    private var rightContextSamples: Int { Int(rightContextSeconds * Double(sampleRate)) }
    private var maxModelSamples: Int { 240_000 }  // 15 seconds window capacity

    func process(
        using manager: AsrManager, decoderState: inout TdtDecoderState, startTime: Date
    ) async throws -> ASRResult {
        var allTokens: [AlignedToken] = []

        var centerStart = 0
        var segmentIndex = 0
        var lastProcessedFrame = 0  // Track the last frame processed by previous chunk

        while centerStart < audioSamples.count {
            // Determine if this is the last chunk
            let isLastChunk = (centerStart + centerSamples) >= audioSamples.count

            // Process chunk with explicit last chunk detection

            let (windowTokens, windowTimestamps, maxFrame) = try await processWindowWithTokens(
                centerStart: centerStart,
                segmentIndex: segmentIndex,
                lastProcessedFrame: lastProcessedFrame,
                isLastChunk: isLastChunk,
                using: manager,
                decoderState: &decoderState
            )

            // Update last processed frame for next chunk
            if maxFrame > 0 {
                lastProcessedFrame = maxFrame
            }

            let frameDuration: TimeInterval = 0.08
            let chunkTokens = zip(windowTokens, windowTimestamps).map {
                AlignedToken(id: $0.0, start: TimeInterval($0.1) * frameDuration, duration: frameDuration)
            }

            if segmentIndex > 0 && !allTokens.isEmpty && !chunkTokens.isEmpty {
                do {
                    allTokens = try mergeLongestContiguous(
                        allTokens, chunkTokens, overlapDuration: frameDuration * 2)
                } catch {
                    allTokens = mergeLongestCommonSubsequence(
                        allTokens, chunkTokens, overlapDuration: frameDuration * 2)
                }
            } else {
                allTokens = chunkTokens
            }
            centerStart += centerSamples
            segmentIndex += 1
        }
        let tokenIds = allTokens.map { $0.id }
        let timestamps = allTokens.map { Int($0.start / 0.08) }
        return manager.processTranscriptionResult(
            tokenIds: tokenIds,
            timestamps: timestamps,
            encoderSequenceLength: 0,  // Not relevant for chunk processing
            audioSamples: audioSamples,
            processingTime: Date().timeIntervalSince(startTime)
        )
    }

    private func processWindowWithTokens(
        centerStart: Int,
        segmentIndex: Int,
        lastProcessedFrame: Int,
        isLastChunk: Bool,
        using manager: AsrManager,
        decoderState: inout TdtDecoderState
    ) async throws -> (tokens: [Int], timestamps: [Int], maxFrame: Int) {
        // Use standard context for all chunks - the TdtDecoder handles last chunk finalization
        let adaptiveLeftContextSamples = leftContextSamples

        // Compute window bounds in samples: [leftStart, rightEnd)
        let leftStart = max(0, centerStart - adaptiveLeftContextSamples)
        let centerEnd = min(audioSamples.count, centerStart + centerSamples)
        let rightEnd = min(audioSamples.count, centerEnd + rightContextSamples)

        // If nothing to process, return empty
        if leftStart >= rightEnd {
            return ([], [], 0)
        }

        let chunkSamples = Array(audioSamples[leftStart..<rightEnd])

        // Pad to model capacity (15s) if needed; keep track of actual chunk length
        let paddedChunk = manager.padAudioIfNeeded(chunkSamples, targetLength: maxModelSamples)

        // Calculate encoder frame offset based on where previous chunk ended
        let actualLeftContextSeconds = Double(adaptiveLeftContextSamples) / Double(sampleRate)
        let startFrameOffset = manager.calculateStartFrameOffset(
            segmentIndex: segmentIndex,
            leftContextSeconds: actualLeftContextSeconds
        )

        let (tokens, timestamps, encLen) = try await manager.executeMLInferenceWithTimings(
            paddedChunk,
            originalLength: chunkSamples.count,
            enableDebug: false,
            decoderState: &decoderState,
            startFrameOffset: startFrameOffset,
            lastProcessedFrame: lastProcessedFrame,
            isLastChunk: isLastChunk
        )

        if tokens.isEmpty || encLen == 0 {
            return ([], [], 0)
        }

        // Take all tokens from decoder (it already processed only the relevant frames)
        let filteredTokens = tokens
        let filteredTimestamps = timestamps
        let maxFrame = timestamps.max() ?? 0

        return (filteredTokens, filteredTimestamps, maxFrame)
    }
}
