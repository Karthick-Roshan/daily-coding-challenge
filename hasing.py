# Count pairs that add upto the target
def countPairs(arr, target):
    dic = {}
    count = 0
    n = len(arr)
    
    for i in range(n):
        if target - arr[i] in dic:
            count = count + dic.get(target - arr[i], 0)

        dic[arr[i]] = dic.get(arr[i], 0) + 1
            
    return count

# print(countPairs([1, 5, 7, -1, 5], 6)) 


# First unique character in a given string
def first_unique_char(s):
    char_count = {}
    
    for char in s:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    
    for i, char in enumerate(s):
        if char_count[char] == 1:
            return i

    return -1

# print(first_unique_char("leetcode"))  