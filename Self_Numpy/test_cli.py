from ctypes import *
import ctypes as ctypes

test = cdll.LoadLibrary('E:\stupideer\C&C++\Practice\cmake-build-debug\libpractice.dll')
print test
print "np_sum : {0}".format(test.np_sum(10, 20))
print "np_avg : {0}".format(test.np_avg(10, 20))

array = c_int * 10
ii = array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
pi = pointer(ii)

ts = test.twoSum
ts.restype = POINTER(ctypes.c_int * 2)
result = ts(pi, 10, 17)

print "two-sum array indices -> [{0},{1}]".format(result.contents[0], result.contents[1])

array1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
print array1[::2]

print "np_hello => not in order process "
test.np_hello()
