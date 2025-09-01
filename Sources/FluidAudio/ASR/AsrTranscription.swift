import CoreML
import Foundation
import OSLog

extension AsrManager {

    internal func transcribeWithState(
        _ audioSamples: [Float], decoderState: inout TdtDecoderState
    ) async throws -> ASRResult {
        guard isAvailable else { throw ASRError.notInitialized }
        guard audioSamples.count >= 16_000 else { throw ASRError.invalidAudioData }

        if config.enableDebug {
            logger.debug("transcribeWithState: processing \(audioSamples.count) samples")
            // Log decoder state values before processing
            let hiddenBefore = (
                decoderState.hiddenState[0].intValue, decoderState.hiddenState[1].intValue
            )
            let cellBefore = (
                decoderState.cellState[0].intValue, decoderState.cellState[1].intValue
            )
            logger.debug(
                "Decoder state before: hidden[\(hiddenBefore.0),\(hiddenBefore.1)], cell[\(cellBefore.0),\(cellBefore.1)]"
            )
        }

        let startTime = Date()

        // Route to appropriate processing method based on audio length

        if audioSamples.count <= 240_000 {
            let originalLength = audioSamples.count
            let paddedAudio: [Float] = padAudioIfNeeded(audioSamples, targetLength: 240_000)
            let (tokens, timestamps, encoderSequenceLength) = try await executeMLInferenceWithTimings(
                paddedAudio,
                originalLength: originalLength,
                enableDebug: config.enableDebug,
                decoderState: &decoderState
            )

            let result = processTranscriptionResult(
                tokenIds: tokens,
                timestamps: timestamps,
                encoderSequenceLength: encoderSequenceLength,
                audioSamples: audioSamples,
                processingTime: Date().timeIntervalSince(startTime)
            )

            if config.enableDebug {
                // Log decoder state values after processing
                let hiddenAfter = (
                    decoderState.hiddenState[0].intValue, decoderState.hiddenState[1].intValue
                )
                let cellAfter = (decoderState.cellState[0].intValue, decoderState.cellState[1].intValue)
                logger.debug(
                    "Decoder state after: hidden[\(hiddenAfter.0),\(hiddenAfter.1)], cell[\(cellAfter.0),\(cellAfter.1)]"
                )
                logger.debug("Transcription result: '\(result.text)'")
            }

            return result
        }

        // ChunkProcessor now uses the passed-in decoder state for continuity
        let processor = ChunkProcessor(audioSamples: audioSamples, enableDebug: config.enableDebug)
        return try await processor.process(using: self, decoderState: &decoderState, startTime: startTime)
    }

    internal func executeMLInferenceWithTimings(
        _ paddedAudio: [Float],
        originalLength: Int? = nil,
        enableDebug: Bool = false,
        decoderState: inout TdtDecoderState,
        startFrameOffset: Int = 0,
        lastProcessedFrame: Int = 0,
        isLastChunk: Bool = false
    ) async throws -> (tokens: [Int], timestamps: [Int], encoderSequenceLength: Int) {

        let melspectrogramInput = try await prepareMelSpectrogramInput(
            paddedAudio, actualLength: originalLength)

        guard
            let melspectrogramOutput = try melspectrogramModel?.prediction(
                from: melspectrogramInput,
                options: predictionOptions
            )
        else {
            throw ASRError.processingFailed("Mel-spectrogram model failed")
        }

        let encoderInput = try prepareEncoderInput(melspectrogramOutput)

        guard
            let encoderOutput = try encoderModel?.prediction(
                from: encoderInput,
                options: predictionOptions
            )
        else {
            throw ASRError.processingFailed("Encoder model failed")
        }

        let rawEncoderOutput = try extractFeatureValue(
            from: encoderOutput, key: "encoder_output", errorMessage: "Invalid encoder output")
        let encoderLength = try extractFeatureValue(
            from: encoderOutput, key: "encoder_output_length",
            errorMessage: "Invalid encoder output length")

        let encoderHiddenStates = rawEncoderOutput
        let encoderSequenceLength = encoderLength[0].intValue

        let (tokens, timestamps) = try await tdtDecodeWithTimings(
            encoderOutput: encoderHiddenStates,
            encoderSequenceLength: encoderSequenceLength,
            originalAudioSamples: paddedAudio,
            decoderState: &decoderState,
            startFrameOffset: startFrameOffset,
            lastProcessedFrame: lastProcessedFrame,
            isLastChunk: isLastChunk
        )

        return (tokens, timestamps, encoderSequenceLength)
    }

    /// Streaming-friendly chunk transcription that preserves decoder state and supports start-frame offset.
    /// This is used by both sliding window chunking and streaming paths to unify behavior.
    internal func transcribeStreamingChunk(
        _ chunkSamples: [Float],
        source: AudioSource,
        startFrameOffset: Int,
        lastProcessedFrame: Int,
        previousTokens: [Int] = [],
        enableDebug: Bool
    ) async throws -> (tokens: [Int], timestamps: [Int], encoderSequenceLength: Int) {
        // Select and copy decoder state for the source
        var state = (source == .microphone) ? microphoneDecoderState : systemDecoderState

        let originalLength = chunkSamples.count
        let padded = padAudioIfNeeded(chunkSamples, targetLength: 240_000)
        let (tokens, timestamps, encLen) = try await executeMLInferenceWithTimings(
            padded,
            originalLength: originalLength,
            enableDebug: enableDebug,
            decoderState: &state,
            startFrameOffset: startFrameOffset,
            lastProcessedFrame: lastProcessedFrame
        )

        // Persist updated state back to the source-specific slot
        if source == .microphone {
            microphoneDecoderState = state
        } else {
            systemDecoderState = state
        }

        // Apply token deduplication if previous tokens are provided
        if !previousTokens.isEmpty && !tokens.isEmpty {
            let (deduped, removedCount) = mergeTokens(previous: previousTokens, current: tokens)
            let adjustedTimestamps = removedCount > 0 ? Array(timestamps.dropFirst(removedCount)) : timestamps

            if enableDebug && removedCount > 0 {
                logger.debug("Streaming chunk: removed \(removedCount) duplicate tokens")
            }

            return (deduped, adjustedTimestamps, encLen)
        }

        return (tokens, timestamps, encLen)
    }

    internal func processTranscriptionResult(
        tokenIds: [Int],
        timestamps: [Int] = [],
        encoderSequenceLength: Int,
        audioSamples: [Float],
        processingTime: TimeInterval,
        tokenTimings: [TokenTiming] = []
    ) -> ASRResult {

        let (text, finalTimings) = convertTokensWithExistingTimings(tokenIds, timings: tokenTimings)
        let duration = TimeInterval(audioSamples.count) / TimeInterval(config.sampleRate)

        // Convert timestamps to TokenTiming objects if provided
        let timingsFromTimestamps = createTokenTimings(from: tokenIds, timestamps: timestamps)

        // Use existing timings if provided, otherwise use timings from timestamps
        let resultTimings = tokenTimings.isEmpty ? timingsFromTimestamps : finalTimings

        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && duration > 1.0 {
            logger.warning(
                "⚠️ Empty transcription for \(String(format: "%.1f", duration))s audio (tokens: \(tokenIds.count))"
            )
        }

        // Calculate confidence based on audio duration and token density
        let confidence = calculateConfidence(
            duration: duration,
            tokenCount: tokenIds.count,
            isEmpty: text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        )

        return ASRResult(
            text: text,
            confidence: confidence,
            duration: duration,
            processingTime: processingTime,
            tokenTimings: resultTimings
        )
    }

    internal func padAudioIfNeeded(_ audioSamples: [Float], targetLength: Int) -> [Float] {
        guard audioSamples.count < targetLength else { return audioSamples }
        return audioSamples + Array(repeating: 0, count: targetLength - audioSamples.count)
    }

    /// Calculate confidence score based on transcription characteristics
    /// Returns a value between 0.0 and 1.0
    private func calculateConfidence(duration: Double, tokenCount: Int, isEmpty: Bool) -> Float {
        // Empty transcription gets low confidence
        if isEmpty {
            return 0.1
        }

        // Base confidence starts at 0.3
        var confidence: Float = 0.3

        // Duration factor: longer audio generally means more confident transcription
        // Confidence increases with duration up to ~10 seconds, then plateaus
        let durationFactor = min(duration / 10.0, 1.0)
        confidence += Float(durationFactor) * 0.4  // Add up to 0.4

        // Token density factor: more tokens per second indicates richer content
        if duration > 0 {
            let tokensPerSecond = Double(tokenCount) / duration
            // Typical speech is 2-4 tokens per second
            let densityFactor = min(tokensPerSecond / 3.0, 1.0)
            confidence += Float(densityFactor) * 0.3  // Add up to 0.3
        }

        // Clamp between 0.1 and 1.0
        return max(0.1, min(1.0, confidence))
    }

    /// Convert frame timestamps to TokenTiming objects
    private func createTokenTimings(from tokenIds: [Int], timestamps: [Int]) -> [TokenTiming] {
        guard !tokenIds.isEmpty && !timestamps.isEmpty && tokenIds.count == timestamps.count else {
            return []
        }

        var timings: [TokenTiming] = []

        for i in 0..<tokenIds.count {
            let tokenId = tokenIds[i]
            let frameIndex = timestamps[i]

            // Convert encoder frame index to time (approximate: 80ms per frame)
            let startTime = TimeInterval(frameIndex) * 0.08
            let endTime = startTime + 0.08  // Approximate token duration

            // Get token text from vocabulary if available
            let tokenText = vocabulary[tokenId] ?? "token_\(tokenId)"

            // Token confidence based on duration (longer tokens = higher confidence)
            let tokenDuration = endTime - startTime
            let tokenConfidence = Float(min(max(tokenDuration / 0.5, 0.5), 1.0))  // 0.5 to 1.0 based on duration

            let timing = TokenTiming(
                token: tokenText,
                tokenId: tokenId,
                startTime: startTime,
                endTime: endTime,
                confidence: tokenConfidence
            )

            timings.append(timing)
        }

        return timings
    }

    /// Slice encoder output to remove left context frames (following NeMo approach)
    private func sliceEncoderOutput(
        _ encoderOutput: MLMultiArray,
        from startFrame: Int,
        newLength: Int
    ) throws -> MLMultiArray {
        let shape = encoderOutput.shape
        let batchSize = shape[0].intValue
        let hiddenSize = shape[2].intValue

        // Create new array with sliced dimensions
        let slicedArray = try MLMultiArray(
            shape: [batchSize, newLength, hiddenSize] as [NSNumber],
            dataType: encoderOutput.dataType
        )

        // Copy data from startFrame onwards
        let sourcePtr = encoderOutput.dataPointer.bindMemory(to: Float.self, capacity: encoderOutput.count)
        let destPtr = slicedArray.dataPointer.bindMemory(to: Float.self, capacity: slicedArray.count)

        for t in 0..<newLength {
            for h in 0..<hiddenSize {
                let sourceIndex = (startFrame + t) * hiddenSize + h
                let destIndex = t * hiddenSize + h
                destPtr[destIndex] = sourcePtr[sourceIndex]
            }
        }

        return slicedArray
    }

    /// Remove duplicate token sequences at the start of the current list that overlap
    /// with the tail of the previous accumulated tokens. Returns deduplicated current tokens
    /// and the number of removed leading tokens so caller can drop aligned timestamps.
    /// Ideally this is not needed. We need to make some more fixes to the TDT decoding logic,
    /// this should be a temporary workaround.
    private enum MergeError: Error {
        case noPairs
    }

    /// Merge token sequences using longest contiguous overlap with LCS fallback.
    /// Returns deduplicated current tokens and the number of removed leading tokens.
    internal func mergeTokens(
        previous: [Int], current: [Int], maxOverlap: Int = 12
    ) -> (deduped: [Int], removedCount: Int) {
        do {
            let merged = try mergeLongestContiguous(previous, current, maxOverlap: maxOverlap)
            let deduped = Array(merged.dropFirst(previous.count))
            let removed = current.count - deduped.count
            return (deduped, removed)
        } catch {
            let merged = mergeLongestCommonSubsequence(previous, current, maxOverlap: maxOverlap)
            let deduped = Array(merged.dropFirst(previous.count))
            let removed = current.count - deduped.count
            return (deduped, removed)
        }
    }

    /// Merge sequences by finding the longest contiguous overlap region.
    private func mergeLongestContiguous(
        _ a: [Int], _ b: [Int], maxOverlap: Int
    ) throws -> [Int] {
        if a.isEmpty { return b }
        if b.isEmpty { return a }

        let overlapA = Array(a.suffix(maxOverlap))
        let overlapB = Array(b.prefix(maxOverlap))

        if overlapA.count < 2 || overlapB.count < 2 {
            return a + b
        }

        var best: [(Int, Int)] = []

        for i in 0..<overlapA.count {
            for j in 0..<overlapB.count {
                if overlapA[i] == overlapB[j] {
                    var k = 0
                    var current: [(Int, Int)] = []
                    while i + k < overlapA.count && j + k < overlapB.count
                        && overlapA[i + k] == overlapB[j + k]
                    {
                        current.append((i + k, j + k))
                        k += 1
                    }
                    if current.count > best.count {
                        best = current
                    }
                }
            }
        }

        guard best.count >= 2 else {
            throw MergeError.noPairs
        }

        let aStartIdx = a.count - overlapA.count
        let indicesA = best.map { aStartIdx + $0.0 }
        let indicesB = best.map { $0.1 }

        var result: [Int] = []
        result.append(contentsOf: a[..<indicesA[0]])

        for idx in 0..<best.count {
            let idxA = indicesA[idx]
            let idxB = indicesB[idx]
            result.append(a[idxA])

            if idx < best.count - 1 {
                let nextIdxA = indicesA[idx + 1]
                let nextIdxB = indicesB[idx + 1]

                let gapA = Array(a[(idxA + 1)..<nextIdxA])
                let gapB = Array(b[(idxB + 1)..<nextIdxB])

                if gapB.count > gapA.count {
                    result.append(contentsOf: gapB)
                } else {
                    result.append(contentsOf: gapA)
                }
            }
        }

        result.append(contentsOf: b[(indicesB.last! + 1)...])
        return result
    }

    /// Merge sequences using longest common subsequence when contiguous overlap fails.
    private func mergeLongestCommonSubsequence(
        _ a: [Int], _ b: [Int], maxOverlap: Int
    ) -> [Int] {
        if a.isEmpty { return b }
        if b.isEmpty { return a }

        let overlapA = Array(a.suffix(maxOverlap))
        let overlapB = Array(b.prefix(maxOverlap))

        if overlapA.count < 2 || overlapB.count < 2 {
            return a + b
        }

        let m = overlapA.count
        let n = overlapB.count
        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

        for i in 1...m {
            for j in 1...n {
                if overlapA[i - 1] == overlapB[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }

        var i = m
        var j = n
        var lcsPairs: [(Int, Int)] = []

        while i > 0 && j > 0 {
            if overlapA[i - 1] == overlapB[j - 1] {
                lcsPairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            } else if dp[i - 1][j] > dp[i][j - 1] {
                i -= 1
            } else {
                j -= 1
            }
        }

        lcsPairs.reverse()

        if lcsPairs.isEmpty {
            return a + b
        }

        let aStartIdx = a.count - overlapA.count
        let indicesA = lcsPairs.map { aStartIdx + $0.0 }
        let indicesB = lcsPairs.map { $0.1 }

        var result: [Int] = []
        result.append(contentsOf: a[..<indicesA[0]])

        for idx in 0..<lcsPairs.count {
            let idxA = indicesA[idx]
            let idxB = indicesB[idx]
            result.append(a[idxA])

            if idx < lcsPairs.count - 1 {
                let nextIdxA = indicesA[idx + 1]
                let nextIdxB = indicesB[idx + 1]

                let gapA = Array(a[(idxA + 1)..<nextIdxA])
                let gapB = Array(b[(idxB + 1)..<nextIdxB])

                if gapB.count > gapA.count {
                    result.append(contentsOf: gapB)
                } else {
                    result.append(contentsOf: gapA)
                }
            }
        }

        result.append(contentsOf: b[(indicesB.last! + 1)...])
        return result
    }

    /// Calculate start frame offset for a sliding window segment
    internal func calculateStartFrameOffset(segmentIndex: Int, leftContextSeconds: Double) -> Int {
        guard segmentIndex > 0 else {
            return 0
        }
        // Use exact encoder frame rate: 80ms per frame = 12.5 fps
        let encoderFrameRate = 1.0 / 0.08  // 12.5 frames per second
        let leftContextFrames = Int(round(leftContextSeconds * encoderFrameRate))

        return leftContextFrames
    }

}
