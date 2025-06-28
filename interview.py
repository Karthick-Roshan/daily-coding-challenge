# Question: 01
#  You’re given an array A of n integers and q queries.
#  Each query can be one of the following two types:
#  • Type 1 Query: (1, l, r) - Replace A[i] with 
# (i-l+1)*A[l] for each index i, where l <= i <= r.
#  • Type 2 Query: (2, l, r) - Calculate the sum of the 
# elements in A from index l to index r.
#  Find the sum of answers to all type 2 queries. Since 
# answer can be large, return it modulo 109+7

def calculate(n: int, arr: list[int], q: int, queries: list[list[int]]) -> int:
    sums = 0
    for query in queries:
        if query[0] == 1:
            for i in range(query[1], query[2] + 1):
                arr[i] = (i - query[1] + 1) * arr[query[1]]
        elif query[0] == 2:
            for j in range(query[1], query[2] + 1):
                sums += arr[j]
    return sums

sample_input = [
    7,
    [1, 8, 6, 10, 5, 6, 9],
    5,
    [
        [2, 0, 3],
        [1, 2, 3],
        [1, 0, 6],
        [2, 1, 4],
        [2, 6, 6]
    ]
]

# print(calculate(*sample_input))


# Question: 02
# Given a number x and an array of integers arr, 
# find the smallest subarray with sum greater than the given value. 
# If such a subarray do not exist return 0 in that case.

# Examples:

# Input: x = 51, arr[] = [1, 4, 45, 6, 0, 19]
# Output: 3
# Explanation: Minimum length subarray is [4, 45, 6]

def smallestSubWithSum(x: int, arr: list[int]) -> int:
    n = len(arr)
    mini = n + 1
    l = 0
    curr = 0
    
    for r in range(n):
        curr += arr[r]
        
        while curr > x:
            mini = min(mini, r - l + 1)
            curr -= arr[l]
            l += 1
            
    return 0 if mini == n + 1 else mini

# print(smallestSubWithSum(51, [1, 4, 45, 6, 0, 19]))



# 2. EASY
#  You are given an array A of length N and an 
# integer k.
#  It is given that a subarray from l to r is considered 
# good, if the number of distinct elements in that 
# subarray doesn’t exceed k. Additionally, an empty 
# subarray is also a good subarray and its sum is 
# considered to be zero.
#  Find the maximum sum of a good subarray.
#  Sample Output Description 1
# 
#  Here, N = 11, k = 2
#  A = [1, 2, 2, 3, 2, 3, 5, 1, 2, 1, 1]
#  We can select the subarray = [2, 2, 3, 2, 3]
#  It is a good subarray because it contains at most k 
# distinct elements.
#  Its sum = 2+2+3+2+3 = 12
#  So, our answer is 12.

def max_sum(N: int, k: int, arr: list[int]) -> int:
    from collections import defaultdict

    freq = defaultdict(int)
    curr_sum = 0
    max_sum = 0
    left = 0

    for right in range(N):
        freq[arr[right]] += 1
        curr_sum += arr[right]

        while len(freq) > k:
            freq[arr[left]] -= 1
            curr_sum -= arr[left]
            if freq[arr[left]] == 0:
                del freq[arr[left]]
            left += 1

        if curr_sum < 0:
            freq.clear()
            curr_sum = 0
            left = right + 1  

        max_sum = max(max_sum, curr_sum)

    return max_sum

# print(max_sum(11, 2, [1, 2, 2, 3, 2, 3, 5, 1, 2, 1, 1]))
# print(max_sum(5, 5, [-1, 1, 3, 2, -1]))



# Given an integer array nums, return all the triplets 
# [nums[i], nums[j], nums[k]] 
# such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

# Notice that the solution set must not contain duplicate triplets.

# Example 1:

# Input: nums = [-1,0,1,2,-1,-4]
# Output: [[-1,-1,2],[-1,0,1]]
# Explanation: 
# nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
# nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
# nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
# The distinct triplets are [-1,0,1] and [-1,-1,2].
# Notice that the order of the output and the order of the triplets does not matter.

def threeSum(nums: list[int]) -> list[list[int]]:
    nums.sort()
    res = []
    n = len(nums)
    for i in range(n-2):
        if i > 0 and nums[i] == nums[i-1]: continue
        l, r = i+1, n-1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s == 0:
                res.append([nums[i], nums[l], nums[r]])
                while l < r and nums[l] == nums[l+1]: l += 1
                while l < r and nums[r] == nums[r-1]: r -= 1
                l += 1; r -= 1
            elif s < 0: l += 1
            else: r -= 1
    return res

print(threeSum([-1,0,1,2,-1,-4]))