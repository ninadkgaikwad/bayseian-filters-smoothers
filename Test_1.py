print("hello world!")

#import bayesian_filters_smoothers.bayesian_filters_smoothers

import numpy as np

""" A=np.matrix([[1,1],[1,1]])
B=np.matrix([[1,1],[1,1]])
D=np.matrix([[1],[2]])

C=np.dot(A,D)
C1=A*D
C2=D+D """

#print(C)

class Boy:

    def __init__(self, A):
        self.A=A

    def Squared(self, a,b):
        return a**2, b**2

    def B(self, b,d):
        b1, b2 = self.Squared(b,d)
        c = self.A(b1,b2)
        return c
    
def AA(a,b):
    c=a+b
    return c

Boy1 = Boy(AA)

cc = Boy1.B(10,5)

print(cc)

w1 = np.ones((3,1))
w2 = w1.transpose()

w3 = np.dot(w1,w2)

print(w3)

from scipy import linalg
a = np.array([[41, 12], [12, 34]])
w, vr = np.linalg.eig(a)

aa = np.diag(w)

aa = (np.sqrt(aa))

bb = np.dot(vr,np.dot(aa, np.linalg.inv(vr)))

cc = (np.dot(bb,bb.transpose()))



print(cc)
