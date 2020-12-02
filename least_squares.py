import numpy as np
import random

def regress(indVecs, depVec):
    for x in indVecs:
        if len(x) != len(depVec):
            print("All predictor vectors must be the same length as predicted vector")
            return None, None
    A = [[0 for i in range(len(indVecs)+2)] for j in range(len(indVecs)+2)]
    for i in range(len(indVecs)):
        for j in range(len(indVecs)+2):
            if i == j:
                A[i][j] = -1
            elif j < len(indVecs):
                A[i][j] = -1.0*sum([indVecs[j][k]*indVecs[i][k] for k in range(len(depVec))])/sum([x**2 for x in indVecs[i]])
            elif j == len(indVecs):
                A[i][j] = -1.0*sum(indVecs[i])/sum([x**2 for x in indVecs[i]])
            else:
                A[i][j] = 1.0*sum([indVecs[i][k]*depVec[k] for k in range(len(depVec))])/sum([x**2 for x in indVecs[i]])
    for j in range(len(indVecs)):
        A[len(indVecs)][j] = -1.0*sum(indVecs[j])/len(depVec)
    A[len(indVecs)][len(indVecs)] = -1
    A[len(indVecs)][len(indVecs)+1] = 1.0*sum(depVec)/len(depVec)
    A[len(indVecs)+1][len(indVecs)+1] = 1
    b = [0 for i in range(len(indVecs)+2)]
    b[-1] = 1
    #print(np.array(A))
    coeff = np.linalg.solve(np.array(A), np.array(b))
    msqe = sum([(depVec[i]-sum([coeff[j]*indVecs[j][i] for j in range(len(coeff)-2)])+coeff[-2])**2 for i in range(len(depVec))])
    return coeff, msqe

if __name__=="__main__":
    #indVecs = [[random.randint(0,9) for i in range(10)] for j in range(10)]
    #depVec = [random.randint(0,9) for i in range(10)]
    indVecs = [[x-random.uniform(-.5,.5) for x in range(10)]]
    #indVecs = [[x for x in range(10)]]
    depVec = [x for x in range(10)]
    print(regress(indVecs, depVec))

    indVecs = [[-1, 1, -1, 1], [-1, -1, 1, 1]]
    depVec = [-1, 1, 1, -1]

    print(regress(indVecs, depVec))
