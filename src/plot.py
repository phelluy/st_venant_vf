from matplotlib.pyplot import *
from math import *

with open("ploplo.dat", "r") as f:
    contenu = f.read().split()
np = len(contenu)/3

xi = [float(contenu[3*i]) for i in range(np)]
unum = [float(contenu[3*i+1]) for i in range(np)]
uex = [float(contenu[3*i+2]) for i in range(np)]
plot(xi, unum, color="blue")
plot(xi, uex, color="red")
show()