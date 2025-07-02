# Reverse a Linked List
# 1 -> 2 -> 3 -> 4 -> None
# 4 -> 3 -> 2 -> 1 -> None

# Code
def reverseList(head):
    stk = []
    
    curr = head
    while curr.next:
        stk.append(curr)
        curr = curr.next
        
    head = curr
    while stk:
        curr.next = stk.pop()
        curr = curr.next
        
    curr.next = None
    return head

# Given the head of a linked list and the number k, Your task is to find the kth node from the end. 
# If k is more than the number of nodes, then the output should be -1.

def getKthFromLast(self, head, k):
    curr = head
    temp = head
    
    count = 0
    while curr and count < k:
        curr = curr.next
        count += 1
        
    if count < k:
        return -1
        
    while curr:
        temp = temp.next
        curr = curr.next
        
    return temp.data if temp else -1