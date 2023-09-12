import random



def diceRoll():
    return round(random.random()*6)



# random.seed(21321)


def arrDiv(arr1, num):
    
    for i in range (0, len(arr1)):
        arr1[i] /= num
    return arr1



x = [0,0,0,0,0,0,0,0,0,0,0,0]


sampleNum = 100


for i in range (0, sampleNum):
    num = diceRoll() + diceRoll() -1
    x[num] += 1

# x = arrDiv(x, sampleNum)
print(x)

sum = 0
for i in range (0, 12):
    sum += x[i]*(i+1)

print(sum/sampleNum)



import matplotlib.pyplot as plt


plt.figure()
plt.show()