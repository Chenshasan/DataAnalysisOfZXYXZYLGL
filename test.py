#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd
data = [[1,2,3],[4,5,6],[7,8,9]]
index = [0,1,2]
columns=['a','b','c']
df = pd.DataFrame(data=data, index=index, columns=columns)
print(df)
print(df.iloc[0:2,1:])