"""m23: problem-solving contest.

No hints about techniques this time. Above each solution, write a comment with:
the technique, why it applies, and the time/space complexity.
"""


# 1. Longest consecutive run.
#    Given an unsorted list of integers, return the length of the longest run of
#    consecutive values (in value, not position). Must be O(n): no sorting.
#    longest_consecutive([100, 4, 200, 1, 3, 2]) -> 4   (1, 2, 3, 4)
def longest_consecutive(nums):
    raise NotImplementedError


# 2. Trapping rain water.
#    heights[i] is the height of a bar of width 1. After rain, how many units of water
#    are trapped between the bars? O(n) time.
#    trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) -> 6
def trap(heights):
    raise NotImplementedError


# 3. Ways to decode a message.
#    Letters are encoded A=1, B=2, ..., Z=26. Given a string of digits, how many ways
#    can it be decoded? "0" alone can't be decoded, and "06" isn't valid for F.
#    num_decodings("226") -> 3 ("BZ", "VF", "BBF"), num_decodings("06") -> 0
def num_decodings(digits):
    raise NotImplementedError


# 4. Can you finish all courses?
#    There are n courses, 0..n-1. prereqs is a list of [course, required_first] pairs.
#    Return True if it's possible to take every course.
#    can_finish(2, [[1, 0]]) -> True, can_finish(2, [[1, 0], [0, 1]]) -> False
def can_finish(n, prereqs):
    raise NotImplementedError


# 5. Shipping within D days.
#    Packages must be shipped IN ORDER; weights[i] is the i-th package. Each day you load
#    packages onto the ship (in order) without going over its capacity. What is the
#    smallest capacity that ships everything within `days` days?
#    ship_capacity([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5) -> 15
def ship_capacity(weights, days):
    raise NotImplementedError


# 6. k-th largest in a stream.
#    Build an object that receives numbers one at a time and always knows the k-th
#    largest so far. add() returns the current k-th largest (or None if fewer than k
#    numbers have arrived). Each add() must be O(log k).
class KthLargest:
    def __init__(self, k):
        raise NotImplementedError

    def add(self, value):
        raise NotImplementedError


# 7. Network delay.
#    times is a list of (u, v, t): a signal goes from node u to node v in t ms.
#    Nodes are 1..n. A signal is sent from node k. How long until ALL nodes have it?
#    Return -1 if some node never receives it.
#    network_delay([(2, 1, 1), (2, 3, 1), (3, 4, 1)], 4, 2) -> 2
def network_delay(times, n, k):
    raise NotImplementedError


# 8. Largest rectangle in a histogram.
#    heights[i] is a bar of width 1. Return the area of the largest rectangle that fits
#    entirely inside the histogram. O(n).
#    largest_rectangle([2, 1, 5, 6, 2, 3]) -> 10   (heights 5 and 6, width 2)
def largest_rectangle(heights):
    raise NotImplementedError


# 9. Minimum window substring.
#    Return the shortest substring of s that contains every character of t (including
#    duplicates), or "" if none. If several have the same length, return the leftmost.
#    min_window("ADOBECODEBANC", "ABC") -> "BANC"
def min_window(s, t):
    raise NotImplementedError


# 10. Equal-sum partition.
#     Can the positive integers in nums be split into two groups with equal sums?
#     can_partition([1, 5, 11, 5]) -> True ([1, 5, 5] and [11]); can_partition([1, 2, 3, 5]) -> False
def can_partition(nums):
    raise NotImplementedError
