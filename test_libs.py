import matplotlib.pyplot as plt
import numpy as np

# 构建数据集
import pandas as pd

df4 = pd.DataFrame({
    "a": np.random.randn(1000) + 1,
    "b": np.random.randn(1000),
    "c": np.random.randn(1000) - 1,
    "d": np.random.randn(1000) - 2,
}, columns=['a', 'b', 'c', 'd'])

df4.plot.hist(alpha=0.5)  # 指定图形透明度
df4.plot.hist(stacked=True, bins=20)  # 堆叠并指定箱数为20
df4.diff().hist()  # 通过diff给每一列数据都绘制一个直方图