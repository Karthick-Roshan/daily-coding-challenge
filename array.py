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