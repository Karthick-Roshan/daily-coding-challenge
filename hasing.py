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

print(countPairs([1, 5, 7, -1, 5], 6)) 