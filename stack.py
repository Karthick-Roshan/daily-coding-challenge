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

print(nextGreaterElement(nums1 = [4,1,2], nums2 = [1,3,4,2]))