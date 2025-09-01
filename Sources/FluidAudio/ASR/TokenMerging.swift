import Foundation

struct AlignedToken: Sendable {
    let id: Int
    var start: TimeInterval
    var duration: TimeInterval
    var end: TimeInterval { start + duration }
}

enum TokenMergeError: Error {
    case noPairs
}

func mergeLongestContiguous(
    _ a: [AlignedToken],
    _ b: [AlignedToken],
    overlapDuration: TimeInterval
) throws -> [AlignedToken] {
    if a.isEmpty { return b }
    if b.isEmpty { return a }

    let aEnd = a.last!.end
    let bStart = b.first!.start

    if aEnd <= bStart {
        return a + b
    }

    let overlapA = a.filter { $0.end > bStart - overlapDuration }
    let overlapB = b.filter { $0.start < aEnd + overlapDuration }

    let enoughPairs = overlapA.count / 2

    if overlapA.count < 2 || overlapB.count < 2 {
        let cutoff = (aEnd + bStart) / 2
        let left = a.filter { $0.end <= cutoff }
        let right = b.filter { $0.start >= cutoff }
        return left + right
    }

    var bestContiguous: [(Int, Int)] = []

    for i in overlapA.indices {
        for j in overlapB.indices {
            if overlapA[i].id == overlapB[j].id
                && abs(overlapA[i].start - overlapB[j].start) < overlapDuration / 2
            {
                var current: [(Int, Int)] = []
                var k = i
                var l = j
                while k < overlapA.count && l < overlapB.count
                    && overlapA[k].id == overlapB[l].id
                    && abs(overlapA[k].start - overlapB[l].start) < overlapDuration / 2
                {
                    current.append((k, l))
                    k += 1
                    l += 1
                }
                if current.count > bestContiguous.count {
                    bestContiguous = current
                }
            }
        }
    }

    if bestContiguous.count >= enoughPairs {
        let aStartIdx = a.count - overlapA.count
        let lcsIndicesA = bestContiguous.map { aStartIdx + $0.0 }
        let lcsIndicesB = bestContiguous.map { $0.1 }

        var result: [AlignedToken] = []
        result.append(contentsOf: a[..<lcsIndicesA[0]])

        for idx in 0..<bestContiguous.count {
            let idxA = lcsIndicesA[idx]
            let idxB = lcsIndicesB[idx]
            result.append(a[idxA])

            if idx < bestContiguous.count - 1 {
                let nextA = lcsIndicesA[idx + 1]
                let nextB = lcsIndicesB[idx + 1]
                let gapA = Array(a[(idxA + 1)..<nextA])
                let gapB = Array(b[(idxB + 1)..<nextB])
                if gapB.count > gapA.count {
                    result.append(contentsOf: gapB)
                } else {
                    result.append(contentsOf: gapA)
                }
            }
        }

        result.append(contentsOf: b[(lcsIndicesB.last! + 1)...])
        return result
    } else {
        throw TokenMergeError.noPairs
    }
}

func mergeLongestCommonSubsequence(
    _ a: [AlignedToken],
    _ b: [AlignedToken],
    overlapDuration: TimeInterval
) -> [AlignedToken] {
    if a.isEmpty { return b }
    if b.isEmpty { return a }

    let aEnd = a.last!.end
    let bStart = b.first!.start

    if aEnd <= bStart {
        return a + b
    }

    let overlapA = a.filter { $0.end > bStart - overlapDuration }
    let overlapB = b.filter { $0.start < aEnd + overlapDuration }

    if overlapA.count < 2 || overlapB.count < 2 {
        let cutoff = (aEnd + bStart) / 2
        let left = a.filter { $0.end <= cutoff }
        let right = b.filter { $0.start >= cutoff }
        return left + right
    }

    let m = overlapA.count
    let n = overlapB.count
    var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

    for i in 1...m {
        for j in 1...n {
            if overlapA[i - 1].id == overlapB[j - 1].id
                && abs(overlapA[i - 1].start - overlapB[j - 1].start) < overlapDuration / 2
            {
                dp[i][j] = dp[i - 1][j - 1] + 1
            } else {
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            }
        }
    }

    var lcsPairs: [(Int, Int)] = []
    var i = m
    var j = n
    while i > 0 && j > 0 {
        if overlapA[i - 1].id == overlapB[j - 1].id
            && abs(overlapA[i - 1].start - overlapB[j - 1].start) < overlapDuration / 2
        {
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
        let cutoff = (aEnd + bStart) / 2
        let left = a.filter { $0.end <= cutoff }
        let right = b.filter { $0.start >= cutoff }
        return left + right
    }

    let aStartIdx = a.count - overlapA.count
    let lcsIndicesA = lcsPairs.map { aStartIdx + $0.0 }
    let lcsIndicesB = lcsPairs.map { $0.1 }

    var result: [AlignedToken] = []
    result.append(contentsOf: a[..<lcsIndicesA[0]])

    for idx in 0..<lcsPairs.count {
        let idxA = lcsIndicesA[idx]
        let idxB = lcsIndicesB[idx]
        result.append(a[idxA])

        if idx < lcsPairs.count - 1 {
            let nextA = lcsIndicesA[idx + 1]
            let nextB = lcsIndicesB[idx + 1]
            let gapA = Array(a[(idxA + 1)..<nextA])
            let gapB = Array(b[(idxB + 1)..<nextB])
            if gapB.count > gapA.count {
                result.append(contentsOf: gapB)
            } else {
                result.append(contentsOf: gapA)
            }
        }
    }

    result.append(contentsOf: b[(lcsIndicesB.last! + 1)...])
    return result
}
