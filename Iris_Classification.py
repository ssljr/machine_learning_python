from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
import matplotlib.pyplot as plt

"""模拟一个鸢尾花分类模型"""

# 加载鸢尾花数据集
data_iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(data_iris['data'], data_iris['target'], random_state=0)

iris_dataframe = pd.DataFrame(X_train, columns=data_iris.feature_names)

grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(10, 10), marker='o', hist_kwds={'bins': 20},
                                 s=60,
                                 alpha=.8, cmap=mglearn.cm3)

plt.show()
