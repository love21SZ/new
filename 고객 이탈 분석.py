#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import seaborn as sns
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import konlpy
import koreanize_matplotlib

df = pd.read_csv("Bank-Customer-Attrition-Insights-Data.csv")
df.head()


# In[2]:


df.info()
df.describe()


# In[3]:


# 이탈 여부 퍼센트 계산

churn_counts = df['Exited'].value_counts()
percent = churn_counts / churn_counts.sum() * 100
labels = ['유지 고객', '이탈 고객']

# 시각화
plt.bar(labels, percent, color=['blue', 'orange'])
plt.title('고객 이탈 비율')
plt.ylabel('비율 (%)')

for i, v in enumerate(percent):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center')

plt.ylim(0, 100)
plt.show()


# In[4]:


gender_churn = df.groupby('Gender')['Exited'].mean() * 100
print(gender_churn)


# In[5]:


geo_churn = df.groupby('Geography')['Exited'].mean() * 100
print(geo_churn)


# In[6]:


card_churn = df.groupby('Card Type')['Exited'].mean().sort_values(ascending=False) * 100
print(card_churn)


# In[7]:


# 나이, 잔액, 추정 월급
numeric_cols = ['Age', 'Balance', 'EstimatedSalary']

for col in numeric_cols:
    plt.figure()
    sns.histplot(data=df, x=col, hue='Exited', kde=True, element='step')
    plt.title(f'{col} 분포 - 이탈 여부별')
    plt.xlabel(col)
    plt.ylabel('빈도')
    plt.show()


# In[8]:


# 숫자형 컬럼만 추출
numeric_df = df.select_dtypes(include='number')

# 상관계수 계산
correlation = numeric_df.corr()

# 이탈(Exited)과의 상관계수만 보기
correlation['Exited'].sort_values(ascending=False)


# In[9]:


summary = pd.DataFrame({
    '전체 평균': df.mean(numeric_only=True),
    '이탈자 평균': df[df['Exited'] == 1].mean(numeric_only=True),
    '유지 고객 평균': df[df['Exited'] == 0].mean(numeric_only=True)
})
print(summary)


# In[10]:


satis = df.groupby('Satisfaction Score')['Exited'].mean() * 100
print(satis)
satis.plot(kind='bar', title='만족도별 이탈률 (%)')
plt.ylabel('이탈률 (%)')
plt.show()


# In[11]:


complain = df.groupby('Complain')['Exited'].mean() * 100
complain.plot(kind='bar', title='불만 제기 여부에 따른 이탈률')
plt.xticks([0,1], ['불만 없음', '불만 있음'], rotation=0)
plt.ylabel('이탈률 (%)')
plt.show()


# In[ ]:




