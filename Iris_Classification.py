import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

"""模拟一个鸢尾花分类模型"""

# 加载鸢尾花数据集
data_iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(data_iris['data'], data_iris['target'], random_state=0)

"""训练模型
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

joblib.dump(knn, "saved_models/knn.bin")
"""
# 测试模型
X_new = np.array([[6.0, 2, 5, 1.8]])
print("X_new shape {}".format(X_new.shape))
model = joblib.load("saved_models/knn.bin")

prediction = model.predict(X_new)
print("X_new prediction: {}".format(prediction))
print("predicted target name: {}".format(data_iris['target_names'][prediction]))

# 评估模型
print("score: {}".format(model.score(X_test, y_test)))
