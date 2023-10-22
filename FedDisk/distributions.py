# import packages
import scipy.stats as stats 
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
 
# generate data
data =stats.norm(scale=1, loc=0).rvs(1000)
 
# plotting a histogram
ax = sns.distplot(data,
                  bins=50,
                  kde=True,
                  color='red',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Normal Distribution', ylabel='Frequency')
 
plt.savefig('dist.jpg')
plt.close()


def KL(p, q):
    p = np.array(p) + 1e-9
    q = np.array(q) + 1e-9
    return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)))


def stats(data):
    return np.mean(data),  np.std(data)

data = np.array(data)
mean, std = stats(data)
print("* mean , std: ",mean,std)

# Parition data into 3 clietns
n1= 800
n2 = 200
n=1000
x1,x2 = sorted(data)[:n1], sorted(data)[-n2:]
m1,sd1 = stats(x1)
m2,sd2 = stats(x2)
# m3,sd3 = stats(x3)

# distribution 
nspace,step = np.linspace(-4, +4, 100, retstep=True)
d1 =  norm.pdf(nspace, m1, sd1)*step
d2 =  norm.pdf(nspace, m2, sd2)*step
# d3 =  norm.pdf(nspace, m3, sd3)*step
d =   norm.pdf(nspace, mean, std)*step

# compute alpha weight
a1 = d/d1
a2 = d/d2
d_avg = n1/n*d1 + n2/n*d2
d_alpha = n1/n*a1*d1 + n2/n*a2*d2

print("KL d1-d", KL(d1,d))
print("KL d2-d", KL(d2,d))
print("KL weighted_d1-d", KL( a1*d1,d))
print("KL weighted_d2-d", KL( a2*d2,d))
print("KL davg-d", KL(d_avg,d))
print("KL alphaApprox-d", KL(d_alpha,d))

# plot
bins = np.linspace(-4, 4, 50)
plt.plot(nspace,d1, label='p1')
plt.plot(nspace,d2, label = 'p2')
plt.plot(nspace,a1*d1, label = 'w-p1')
plt.plot(nspace,a2*d2, label = 'w-p2')
plt.plot(nspace,d, label= 'org')
# plt.plot(nspace,d_avg, label = 'avg')
# plt.plot(nspace,d_alpha, label = 'weighted')
plt.legend()
plt.savefig('dist1.jpg')
breakpoint()