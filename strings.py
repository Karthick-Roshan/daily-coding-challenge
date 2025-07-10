# group Anagram

from collections import defaultdict
def groupAnagrams(strs: list[str]) -> list[list[str]]:
    res = defaultdict(list)

    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - 97] += 1  

        res[tuple(count)].append(s)

    return list(res.values())

# print(groupAnagrams(["act","pots","tops","cat","stop","hat"]))

def isIsomorphic(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False

    map_s_t = {}
    map_t_s = {}

    for ch_s, ch_t in zip(s, t):
        if ch_s in map_s_t:
            if map_s_t[ch_s] != ch_t:
                return False
        else:
            map_s_t[ch_s] = ch_t

        if ch_t in map_t_s:
            if map_t_s[ch_t] != ch_s:
                return False
        else:
            map_t_s[ch_t] = ch_s

    return True

# print(isIsomorphic("egg", "add"))


# Implement Atoi
# Given a string s, the objective is to convert it into integer format without utilizing any built-in functions. 

# Cases for atoi() conversion:

# Skip any leading whitespaces.
# Check for a sign (‘+’ or ‘-‘), default to positive if no sign is present.
# Read the integer by ignoring leading zeros until a non-digit character is encountered or end of the string is reached. If no digits are present, return 0.
# If the integer is greater than 231 – 1, then return 231 – 1 and if the integer is smaller than -231, then return -231.

def myAtoi(s: str) -> int:
    n = len(s)
    sign = 1
    idx = 0
    res = 0
    
    while (idx < n) and s[idx] == " ":
        idx += 1
        
    if idx < n and (s[idx] == "-" or s[idx] == "+"):
        if s[idx] == "-":
            sign = -1
        idx += 1
    
    while idx < n and '0' <= s[idx] <= '9':
        
        res = 10 * res + (ord(s[idx]) - ord('0')) 
        
        if res > (2 ** 31 - 1):
            return sign * (2 ** 31 - 1) if sign == 1 else -2 ** 31
            
        idx += 1
        
    return sign * res

# print(myAtoi("42"))


# Given two non-empty strings s1 and s2, consisting only of lowercase English letters, 
# determine whether they are anagrams of each other or not. 
# Two strings are considered anagrams if they contain the same characters 
# with exactly the same frequencies, regardless of their order.

def areAnagrams(s1, s2):
    freq = [0] * 26
    
    for i in s1:
        freq[ord(i) - ord('a')] += 1
        
    for i in s2:
        freq[ord(i) - ord('a')] -= 1 
        
    for c in freq:
        if c != 0:
            return False
        
    return True

# print(areAnagrams("anagram", "nagaram"))