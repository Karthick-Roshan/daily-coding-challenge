# Next Greater Element

def nextGreaterElement(nums1: list[int], nums2: list[int]) -> list[int]:
    stk = []
    ans = []
    dic = {}

    for i in range(len(nums2) - 1, -1, -1):
        while stk and stk[-1] <= nums2[i]:
            stk.pop()
        if not stk:
            dic[nums2[i]] = -1
        else:
            dic[nums2[i]] = stk[-1]
            
        stk.append(nums2[i])

    for val in nums1:
        ans.append(dic[val])

    return ans

# print(nextGreaterElement(nums1 = [4,1,2], nums2 = [1,3,4,2]))

def finalPrices(prices: list[int]) -> list[int]:
    stack = []
    for i, price in enumerate(prices):
        print(i, price)
        while stack and prices[stack[-1]] >= price:
            idx = stack.pop()
            prices[idx] -= price
        stack.append(i)
    return prices

# print(finalPrices([8,4,6,2,3]))