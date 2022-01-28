from matplotlib.pyplot import *
#from math import *

with open("ploplo.dat", "r") as f:
    contenu = f.read().split()

#print(contenu)
np = len(contenu)//2
#print(np)

x = [float(contenu[2*i]) for i in range(np)]
y = [float(contenu[2*i+1]) for i in range(np)]
plot(x, y, color="blue")
show()