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