import seaborn as sns
import matplotlib.pyplot as plt

fitted_vals = reg.predict(X_test)
resids = y_test - fitted_vals

fig, ax = plt.subplots(1,2)

sns.regplot(x=fitted_vals, y=y_test, lowess=True, ax=ax[0], line_kws={'color': 'red'})
ax[0].set_title('Observed vs. Predicted Values', fontsize=16)
ax[0].set(xlabel='Predicted', ylabel='Observed')

sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})
ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
ax[1].set(xlabel='Predicted', ylabel='Residuals')