import matplotlib.pyplot as plt
import csv
import train_ged_distribution as tr_gd
import test_ged_distribution as te_gd
from scipy.stats import skew

test_bins = 170
train_bins = 212

test_avg = 36.64
train_avg = 37.14

train_skew = 2.841
test_skew = 2.887
tr_x = []
te_x=[]

tr_x = tr_gd.my_list
te_x = te_gd.my_list

print(skew(tr_x))

plt.style.use('ggplot')
plt.hist(tr_x, bins=train_bins, color = "blue", lw =10)
plt.xlabel('Approx. GED')
plt.ylabel('Count')
plt.title('Train Dataset Size - 40500')
plt.axvline(train_avg, color='k', linestyle='dashed', linewidth=1.2)

min_ylim, max_ylim = plt.ylim()
plt.text(train_avg, max_ylim*0.9, ' Mean: {:.2f}'.format(train_avg))

plt.text(170, 1092, 'Skew = 2.841', style='italic',bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 10})

plt.show()
