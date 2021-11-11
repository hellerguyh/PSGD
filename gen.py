def myGen():
    l = [1,2,3,4,5,6]
    for i in l: 
        yield i

itr = myGen()
print(next(itr))
print(next(itr))
print(next(itr))
print(next(itr))
print(next(itr))
print(next(itr))
print(next(itr))
