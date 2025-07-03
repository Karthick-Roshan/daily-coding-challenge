# Kadane Algorithm

def kadane(arr: list[int]) -> int:
    summ = 0
    maxx = float('-inf')

    for i in range(len(arr)):
        summ += arr[i]

        if summ > maxx:
            maxx = summ

        if summ < 0:
            summ = 0

    return maxx

print(kadane([-2,-3,4,-1,-2,1,5,-3]))


# Flood Fill

def floodFill(image: list[list[int]], sr: int, sc: int, color: int) -> list[list[int]]:
    original_color = image[sr][sc]
    
    if original_color == color:
        return image  

    def dfs(r, c):
        if (r < 0 or r >= len(image) or
            c < 0 or c >= len(image[0]) or
            image[r][c] != original_color):
            return

        image[r][c] = color

        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    dfs(sr, sc)
    return image

print(floodFill([[1,1,1],[1,1,0],[1,0,1]], 1, 1, 2))


# Best Time to Buy and Sell Stock

def maxProfit(prices: list[int]) -> int:
    mins = prices[0]
    res = 0

    for i in prices:
        mins = min(i, mins)
        res = max(res, i - mins)

    return res 

print(maxProfit([7, 10, 1, 3, 6, 9, 2]))


# Next larger Element

def nextLargerElement(arr: list[int]) -> list[int]:
    n = len(arr)
    result = [-1] * n
    stack = []

    for i in range(n):
        while stack and arr[i] > arr[stack[-1]]:
            idx = stack.pop()
            result[idx] = arr[i]
        stack.append(i)

    return result

print(nextLargerElement([1,3,2,4]))


# Sort012

def sort012(arr: list[int]):
    li = [0] * 3
    n = len(arr)
    
    for i in range(n):
        li[arr[i]] += 1

    id, val = 0, 0
    for i in li:
        for _ in range(i):
            arr[id] = val
            id += 1
        val += 1

    return arr

print(sort012([0,1,2,0,1,2]))

# Given an array of integers arr[] that is first strictly increasing and then maybe strictly decreasing, 
# find the bitonic point, that is the maximum element in the array.
# Bitonic Point is a point before which elements are strictly increasing and 
# after which elements are strictly decreasing.

# Examples:
# Input: arr[] = [1, 2, 4, 5, 7, 8, 3]
# Output: 8

def findMaximum(arr: list[int]) -> int:
    n = len(arr)
    
    l, r = 0, n - 1
    
    while l <= r:
        mid = l + (r - l) // 2
        
        if mid > 0 and mid < n - 1:
            if arr[mid] > arr[mid - 1] and arr[mid] > arr[mid + 1]:
                return arr[mid]
            elif arr[mid] > arr[mid - 1] and arr[mid] < arr[mid + 1]:
                l = mid + 1
            else:
                r = mid - 1
                
        elif mid == 0:
                return arr[0] if arr[0] > arr[1] else arr[1]
        elif mid == n - 1:
                return arr[-1] if arr[-1] > arr[-2] else arr[-2]
        
print(findMaximum([1, 2, 4, 5, 7, 8, 3]))

# Count Triplet

def countTriplets(n: int, sum: int, arr: list[int]) -> list:
    arr.sort()
    count = 0

    for i in range(n - 2):
        left = i + 1
        right = n - 1

        while left < right:
            current_sum = arr[i] + arr[left] + arr[right]

            if current_sum < sum:
                count += (right - left)
                left += 1
            else:
                right -= 1

    return count

# print(countTriplets(n = 5, sum = 12, arr = [5, 1, 3, 4, 7]))



# Majority Element II
# Given an integer array of size n, find all elements that appear more than ⌊ n/3 ⌋ times.

def majorityElement(nums: list[int]) -> list:
    if not nums:
        return []

    c1, c2 = None, None
    count1, count2 = 0, 0

    for num in nums:
        if num == c1:
            count1 += 1
        elif num == c2:
            count2 += 1
        elif count1 == 0:
            c1, count1 = num, 1
        elif count2 == 0:
            c2, count2 = num, 1
        else:
            count1 -= 1
            count2 -= 1

    result = []
    for candidate in (c1, c2):
        if nums.count(candidate) > len(nums) // 3:
            result.append(candidate)

    return result

# print(majorityElement([3,2,3]))


# Leetcode 1652: Defuse the bomb

def decrypt(code: list[int], k: int) -> list[int]:
    n = len(code)
    if k == 0:
        return [0] * n
    
    result = [0] * n
    code = code * 2  

    window_sum = 0
    if k > 0:
        for i in range(1, k + 1):
            window_sum += code[i]
        for i in range(n):
            result[i] = window_sum
            window_sum -= code[i + 1]
            window_sum += code[i + k + 1]
    else:
        k = -k
        for i in range(n - k, n):
            window_sum += code[i]
        for i in range(n):
            result[i] = window_sum
            window_sum -= code[i + n - k]
            window_sum += code[i + n]

    return result

print(decrypt([5,7,1,4], 3))



# Given an array arr of distinct elements, the task is to rearrange the elements of the array 
# in a zig-zag fashion so that the converted array should be in the below form: 

# arr[0] < arr[1]  > arr[2] < arr[3] > arr[4] < . . . . arr[n-2] < arr[n-1] > arr[n]. 

def zigZag(arr: list[int]) -> list[int]:
    n = len(arr)
    
    zig = 1
    for i in range(n-1):
        if zig:
            if arr[i] > arr[i+1]:
                arr[i], arr[i+1] = arr[i+1], arr[i]
            zig = 0
        else:
            if arr[i] < arr[i+1]:
                arr[i], arr[i+1] = arr[i+1], arr[i]
            zig = 1

    return arr

print(zigZag([4, 3, 7, 8, 6, 2, 1]))