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

# print(kadane([-2,-3,4,-1,-2,1,5,-3]))


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

# print(floodFill([[1,1,1],[1,1,0],[1,0,1]], 1, 1, 2))


# Best Time to Buy and Sell Stock

def maxProfit(prices: list[int]) -> int:
    mins = prices[0]
    res = 0

    for i in prices:
        mins = min(i, mins)
        res = max(res, i - mins)

    return res 

# print(maxProfit([7, 10, 1, 3, 6, 9, 2]))


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

# print(nextLargerElement([1,3,2,4]))


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

# print(sort012([0,1,2,0,1,2]))

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
        
# print(findMaximum([1, 2, 4, 5, 7, 8, 3]))

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

# print(decrypt([5,7,1,4], 3))


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

# print(zigZag([4, 3, 7, 8, 6, 2, 1]))


def secondLargest(arr: list[int]) -> int:
    n = len(arr)
    l, sl = 0 , 0

    l = arr[0]
    for i in range(1, n):
        if arr[i] > sl and arr[i] > l:
            sl = l
            l = arr[i]
            
        if arr[i] > sl and arr[i] < l:
            sl = arr[i]

    return sl if sl != 0 else -1

# print(secondLargest([10, 10, 10]))


# Indexes of Subarray Sum
def subarraySum(arr, target):
    n = len(arr)
    ssum = 0
    l = 0

    for r in range(n):
        ssum += arr[r]

        while ssum > target and l <= r:
            ssum -= arr[l]
            l += 1

        if ssum == target:
            return [l + 1, r + 1]

    return [-1]

# print(subarraySum([1, 2, 3, 7, 5], 12))


# K largest Element
import heapq

def k_largest_elements(arr, k):
    if k == 0:
        return []

    min_heap = arr[:k]
    heapq.heapify(min_heap)  

    for num in arr[k:]:
        if num > min_heap[0]:
            heapq.heappushpop(min_heap, num)  

    return sorted(min_heap, reverse=True)  

# print(k_largest_elements([3, 1, 5, 12, 2, 11], 3))


# Push Zeros to End
def pushZerosToEnd(self,arr):
    n = len(arr)
    
    count = 0
    for i in range(n):
        if arr[i] != 0:
            arr[i], arr[count] = arr[count], arr[i]
            
            count += 1

    return arr

# print(pushZerosToEnd([1, 2, 0, 4, 3, 0, 5, 0]))


# Next Permutation
def nextPermutation(arr: list[int]) -> list[int]:
    pivot = -1
    n = len(arr)

    for i in range(n - 2, -1, -1):
        if arr[i] < arr[i + 1]:
            pivot = i
            break

    if pivot == -1:
        arr.reverse()
        return arr
    
    for i in range(n - 1, -1, -1):
        if arr[i] > arr[pivot]:
            arr[i], arr[pivot] = arr[pivot], arr[i]
            break

    arr[pivot + 1:] = reversed(arr[pivot + 1:])
    
    return arr

# print(nextPermutation([2, 4, 1, 7, 5, 0]))


# Smallest positive missing element
def missingNumber(arr):
    n = len(arr)

    for i in range(n):
        while 1 <= arr[i] <= n and arr[arr[i] - 1] != arr[i]:
            crt_idx = arr[i] - 1
            arr[i], arr[crt_idx] = arr[crt_idx], arr[i]

    for i in range(n):
        if arr[i] != i + 1:
            return i + 1

    return n + 1

# print(missingNumber([2, -3, 4, 1, 1, 7]))   


# Search in Rotated Sorted Array
def search(arr, key):
    n = len(arr)
    
    l, r = 0, n - 1
    while l <= r:
        mid = (l + r) // 2
        if arr[mid] == key:
            return mid
            
        if arr[l] <= arr[mid]:
            if arr[l] <= key < arr[mid]:
                r = mid - 1
            else:
                l = mid + 1
                
        else:
            if arr[mid] < key <= arr[r]:
                l = mid + 1
            else:
                r = mid - 1
                
    return -1

# print(search([4,5,6,7,0,1,2], 0))


# Given an amount, 
# find the minimum number of notes of different denominations 
# that sum up to the given amount. 

def minNotes(amount):
    notes = [2000, 500, 200, 100, 20, 10, 5, 2, 1]
    noteCount = {}
    for note in notes:
        if amount >= note:
            noteCount[note] = amount // note
            amount %= note
    return noteCount

# print(minNotes(868))


# Max Subarray 
def maxSubArray(nums):
    max_curr = max_g = nums[0]

    for i in range(1, len(nums)):
        max_curr = max(nums[i], max_curr + nums[i])
        max_g = max(max_g, max_curr)

    return max_g

# print(maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))


# Inversion Count
def inversionCount(arr: list[int]) -> int:
    def sort(arr):
        if len(arr) < 2: return arr, 0
        mid = len(arr) // 2
        left, invL = sort(arr[:mid])
        right, invR = sort(arr[mid:])
        merged, inv = [], invL + invR
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i]); i += 1
            else:
                merged.append(right[j]); j += 1
                inv += len(left) - i
        merged += left[i:] + right[j:]
        return merged, inv
    return sort(arr)[1]

# print(inversionCount([2, 4, 1, 3, 5]))

# Triplet sum in an array
def find3Numbers(arr, target):
    arr.sort()
    n = len(arr)
    
    for i in range(n - 2):
        left = i + 1
        right = n - 1
        
        while left < right:
            current_sum = arr[i] + arr[left] + arr[right]
            
            if current_sum == target:
                return True
            elif current_sum < target:
                left += 1
            else:
                right -= 1
                
    return False

# print(find3Numbers([1, 4, 45, 6, 10, 8], 22))

#longest Consecutive Number
def longestConsecutive(arr):
    arr = sorted(set(arr))
    n = len(arr)
    
    maxlen = currlen = 1
    
    for i in range(1, n):
        if arr[i] == arr[i - 1] + 1:
            currlen += 1
            maxlen = max(maxlen, currlen)
        else:
            currlen = 1
            
    return maxlen

# print(longestConsecutive([100, 4, 200, 1, 3, 2]))


def pairs(arr: list[int], k: int) -> list[int]:
    n = len(arr)

    l, r = 0, n - 1

    while arr[r] >= k:
        r -= 1
    print(arr[r])
    li = []
    while l < r:
        total = arr[l] + arr[r]
        if total > k:
            r -= 1
        else:
            for i in range(l+1, r+1):
                li.append((l, i))
            l += 1
    return li

# print(pairs([3,4,7,11,15,19], 15))


# KOKO eating Bananas
from math import ceil
def minEatingSpeed(piles, h):
    l, r = 1, max(piles)
    res = r

    while l <= r:
        mid = (l + r) // 2

        speed = 0
        for pile in piles:
            speed += ceil(pile / float(mid))

        if speed <= h:
            res = mid
            r = mid - 1
        else:
            l = mid + 1

    return res

# print(minEatingSpeed([30,11,23,4,20], 5))

# 3487. Maximum Unique Subarray Sum After Deletion
def maxSum(nums: list[int]) -> int:
    dic = {}
    for num in nums:
        dic[num] = dic.get(num, 0) + 1

    unique = [key for key in dic if key >= 0]

    if unique:
        return sum(unique)
    else:
        uni_el = [key for key in dic if key < 0]
        return max(uni_el) if uni_el else 0
    
# print(maxSum([1,2,-1,-2,1,0,-1]))
# print(maxSum([1,1,0,1,1]))
# print(maxSum([1,2,3,4,5]))
# print(maxSum([-100, -1]))
# print(maxSum([-100, -1, 0, 0]))


# 2210. Count Hills and Valleys in an Array
def countHillValley(nums):
    count = 0

    filtered = [nums[0]]
    for i in nums[1:]:
        if i != filtered[-1]:
            filtered.append(i)

    for i in range(1, len(filtered) - 1):
        if filtered[i - 1] < filtered[i] and filtered[i + 1] < filtered[i]:
            count += 1
        if filtered[i - 1] > filtered[i] and filtered[i + 1] > filtered[i]:
            count += 1
    
    return count

# print(countHillValley([6,6,5,5,4,1]))

def findEquilibrium(arr):
    total = sum(arr)
    left = 0

    for i in range(len(arr)):
        if left == total - arr[i] - left:
            return i
        left += arr[i]

    return -1

# print(findEquilibrium([1, 2, 3, 4, 5]))
# print(findEquilibrium([1, 2, 0, 3]))

# Longest subarray with length k
def longestSubArray(arr, k):
    n = len(arr)
    lsa, pref = 0, 0
    dic = {}

    for i in range(n):
        pref += arr[i]
        
        if pref == k:
            lsa = i + 1
        
        if pref - k in dic:
            lsa = max(lsa, i - dic[pref - k])
        
        if pref not in dic:
            dic[pref] = i

    return lsa

# arr = [10, 5, 2, 7, 1, -10]
# print(longestSubArray(arr, 15))


# Largest subarray of 0's and 1's
def maxLen(arr: list[int]) -> int:
        n = len(arr)
        dic = {}
        pref, res = 0, 0
        
        for i in range(n):
            pref += -1 if arr[i] == 0 else 1
            
            if pref == 0:
                res = i + 1
                
            if pref in dic:
                res = max(res, i - dic[pref])
            else:
                dic[pref] = i
                
        return res

# print(maxLen([1, 0, 1, 1, 1, 0, 0]))


# Product array puzzle
def productExceptSelf(arr):
    zeros = 0
    prod = 1
    idx = -1
    
    n = len(arr)
    
    for i in range(n):
        if arr[i] == 0:
            zeros += 1
            idx = i
        else:
            prod *= arr[i]
            
    res = [0] * n
            
    if zeros == 0:
        for i in range(n):
            res[i] = prod // arr[i]
            
    elif zeros == 1:
        res[idx] = prod
        
    return res

# print(productExceptSelf([10, 3, 5, 6, 2]))