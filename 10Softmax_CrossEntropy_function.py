import numpy as np
L=[5,6,7]
# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.

required_list = []
def softmax(L):
    List  = np.exp(L)
    print(List)
    sum_of_denominator = 0
    for i in range(len(List)):
        sum_of_denominator = sum_of_denominator+ List[i]
    #print(sum_of_denominator)

    for j in List:
        required_list.append(j* 1.0/sum_of_denominator)
    return required_list

print(softmax(L))
P = 0.13
Y = 1
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

print(cross_entropy(Y,P))