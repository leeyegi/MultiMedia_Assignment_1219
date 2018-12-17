'''
def move(i, j):
    if j < (8 - 1):
        return max(0, i - 1), j + 1
    else:
        return i + 1, j

def zigzag(n,m,scale):
    a = [[0] * 8 for _ in range(8)]

    tmp=[]
    tmp.append(16)
    for i in range(1,64):
        if i<m:
            tmp.append(scale)
        else:
            tmp.append(n*scale)

    print(tmp)

    x, y = 0, 0
    for v in range(64):
        a[y][x] =tmp[v]
        print(y,x)
        if (x + y) & 1:
            x, y = move(x, y)
        else:
            y, x = move(y, x)
    return a


from pprint import pprint

pprint(zigzag(10,10,10))
'''
