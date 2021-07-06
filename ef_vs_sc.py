# Author: Nicholas Mosca
# ejection fraction vs serum creatine
from data_editing import * 
import matplotlib.pyplot as plt





# scatter plot 
# plt.scatter(death_sc, death_ef, marker='^',label = 'dead',color = 'blue')
# plt.scatter(living_sc, living_ef, marker='o', label = 'survived',color = 'orange')
# plt.plot([death_sc.min(), death_sc.max()], [living_ef.min(), living_ef.max()], 'r--')
# plt.legend(loc = ' upper center')


fig = plt.figure(figsize=(30,20),dpi= 800)
ax1 = fig.add_subplot(221)
ax1.scatter(living_sc, living_ef, marker='o', label = 'survived',color = 'orange')
ax1.scatter(death_sc, death_ef, marker='^',label = 'dead',color = 'blue')
ax1.plot([death_sc.min(), death_sc.max()], [living_ef.min(), living_ef.max()], 'r--')
ax1.set_xlabel("serum creatine")
ax1.set_ylabel("ejection fraction")
ax1.legend(loc = 'lower right')

plt.savefig("Figure_3.png")
