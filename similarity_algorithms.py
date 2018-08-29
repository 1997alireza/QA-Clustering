def euclidean_distance(str1, str2):
    words1 = str1.split(' ')
    words2 = str2.split(' ')
    set1 = set(words1)
    set2_minus1 = set(words2) - set1
    s = 0
    for word in set1:
        s1 = 0
        s2 = 0
        for w in words1:
            if w == word:
                s1 = s1+1
        for w in words2:
            if w == word:
                s2 = s2+1
        s = s + (s2-s1)**2
    for word in set2_minus1:
        s2 = 0
        for w in words2:
            if word == w:
                s2 = s2+1
        s = s + s2**2
    return s**(1/2.0)


def edit_distance(str1, str2):
    # w_ins == w_del == w_sub
    l1 = len(str1)
    l2 = len(str2)
    diff = [[0 for i in range(0, l2+1)] for j in range(0, l1+1)]
    diff[0][0] = 0
    for i in range(1, l1+1):
        diff[i][0] = i
    for i in range(1, l2+1):
        diff[0][i] = i
    for i in range(1, l1+1):
        for j in range(1, l2+1):
            if str1[i-1] == str2[j-1]:
                diff[i][j] = diff[i-1][j-1]
            else:
                diff[i][j] = min(
                    diff[i-1][j],
                    diff[i][j-1],
                    diff[i-1][j-1]
                ) + 1
    return diff[l1][l2]
