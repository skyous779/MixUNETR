'''
Author: skyous 1019364238@qq.com
Date: 2024-02-28 17:47:32
LastEditors: skyous 1019364238@qq.com
LastEditTime: 2024-02-28 17:47:38
FilePath: /prostate158-main/config/prostatex_copy/split.py
Description: 

Copyright (c) 2024 by 1019364238@qq.com, All Rights Reserved. 
'''
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取CSV文件
data = pd.read_csv('all.csv')

# 随机划分训练集和其余部分，比例为8:2
train_data, rest_data = train_test_split(data, test_size=0.2, random_state=42)

# 随机划分测试集和验证集，比例为1:1
test_data, val_data = train_test_split(rest_data, test_size=0.5, random_state=42)

# 保存为单独的CSV文件
train_data.to_csv('train_.csv', index=False)
test_data.to_csv('test_.csv', index=False)
val_data.to_csv('val_.csv', index=False)