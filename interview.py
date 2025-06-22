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

print(calculate(*sample_input))