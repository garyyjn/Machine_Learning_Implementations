import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt


data = pd.read_csv('./billboard.csv')
plt.hist(data['Weeks'], bins=50)
plt.show()

from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate
grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.1, 1.0, 30)},
                    cv=20) # 20-fold cross-validation
grid.fit(np.array(data['Weeks']).reshape(-1, 1))
print(grid.best_params_)
x_grid = np.linspace(1, 101, 100)
kde = grid.best_estimator_
pdf = np.exp(kde.score_samples(x_grid[:, None]))

fig, ax = plt.subplots()
ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
ax.hist(data['Weeks'], 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
ax.legend(loc='upper left')
ax.set_xlim(1, 100);
plt.show()