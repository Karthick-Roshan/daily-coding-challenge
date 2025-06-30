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
        # Check mapping from s to t
        if ch_s in map_s_t:
            if map_s_t[ch_s] != ch_t:
                return False
        else:
            map_s_t[ch_s] = ch_t

        # Check reverse mapping from t to s
        if ch_t in map_t_s:
            if map_t_s[ch_t] != ch_s:
                return False
        else:
            map_t_s[ch_t] = ch_s

    return True

print(isIsomorphic("egg", "add"))