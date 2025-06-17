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