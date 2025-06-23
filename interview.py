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