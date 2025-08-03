# Given a square matrix mat[][] of size n x n. 
# *The task is to rotate it by 90 degrees in an anti-clockwise direction without using any extra space. 

# Examples:

# Input: mat[][] = [[0, 1, 2], 
#                 [3, 4, 5], 
#                 [6, 7, 8]] 
# Output: [[2, 5, 8],
#         [1, 4, 7],
#         [0, 3, 6]]


def rotateMatrix(mat):
    print("Original Matrix:")
    for i in mat:
        print(i)

    n = len(mat[0])
    
    def rev(arr, n):
        l, r = 0, n - 1
        while l < r:
            arr[l], arr[r] = arr[r], arr[l]
            l += 1
            r -= 1
            
    for i in mat:
        rev(i, n)
        
    for i in range(n):
        for j in range(i):
            mat[i][j], mat[j][i] = mat[j][i], mat[i][j]
            
    print("\nRotated Matrix:") 
    for i in mat:
        print(i)

# rotateMatrix([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

# Toeplitz Matrix
def isToeplitzMatrix(matrix):
    m = len(matrix)
    n = len(matrix[0])

    for i in range(m - 1):
        for j in range(n - 1):
            if matrix[i][j] != matrix[i + 1][j + 1]:
                return False

    return True

# print(isToeplitzMatrix([[1,2,3,4],[5,1,2,3],[9,5,1,2]]))