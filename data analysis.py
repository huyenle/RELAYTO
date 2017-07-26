import json
from bson import json_util
import re
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# pre-process the data
with open("main.json", encoding="utf8") as data_file:
    bsondata = data_file.read()


def preProcess(bsondata):
    jsondata=re.sub(r'ObjectId\s*\(\s*\"(\S+)\"\s*\)',
                      r'"\1"',
                      bsondata)

    jsondata=re.sub(r'NumberLong\s*\(\s*(\S+)\s*\)',
                      r'"\1"',
                      jsondata)

    jsondata=re.sub(r'NumberInt\s*\(\s*(\S+)\s*\)',
                      r'"\1"',
                      jsondata)

    jsondata=re.sub(r'ISODate\s*\(\s*\"(\S+)\"\s*\)',
                      r'"\1"',
                      jsondata)
    return jsondata


jsondata=json.loads(preProcess(bsondata), object_hook=json_util.object_hook)

# Separate the data
data=pd.DataFrame(jsondata)

# Reformat the data
data['views']=pd.to_numeric(data['views'])
data['created']=pd.to_datetime(data['created'])
# add daily view up to 2017-06-09
now = pd.to_datetime('2017-06-09')
data['lifetime'] = data['created'].apply(lambda x: (now - x).days)
data['daily_views'] = data['views']/data['lifetime']
# separate data
meta=data.drop(['boardResources', 'items'], 1) # only contain information of the documents, not contents


# 1.The most 10 viewed documents
# sort data by number of views
meta_view_sorted=data.sort_values(by='views', ascending=False)
top10 = meta_view_sorted.loc[:, ['title', 'views']].head(10)

# Distribution of view numbers
# create pie chart for top 10 and non-top 10 documents
top_col = list(np.repeat('top10',10)) + list(np.repeat('others',190))
meta_view_sorted['top'] = top_col

summary = meta_view_sorted.views.groupby(meta_view_sorted.top).sum()
summary.plot.pie(figsize=(6, 6), autopct='%.1f')

# Kde plot of number of views
meta_view_sorted.plot(kind='kde', y='views', xlim=[0,33000])

# top 10 document with the most average daily views for each document
top10_daily=meta_view_sorted.sort_values(by='daily_views', ascending=False).loc[:, ['title', 'daily_views']].head(10)

# 2. Docuemnts progression over dates

# convert the time stamp into id of the dataframe to be easily manipulated
meta.set_index(meta['created'], inplace=True)
# number of document created over the month/year:
monthly_created = pd.DataFrame(meta['title'].resample('M').count())
# cumulative line chart for number of documents created
monthly_created.cumsum().plot(legend = False, title = 'Cumulative number of documents over time on RELAYTO')

# 3. Calculate the number of pages
# Because only the last 100 documents have more than 1 page
# I separated this part as contents_items
contents_items = data[100:].drop('boardResources',1)
contents_items['pages'] = contents_items['items'].apply(lambda x: len(x))
# distribution of pages
sb.kdeplot(contents_items['pages'], shade=True, legend=False)

# Correlation between number of pages and number of views
pages_max = contents_items['pages'].max()
pages_min = contents_items['pages'].min()

corr_coef = contents_items['pages'].corr(contents_items['views'])
contents_items.plot(kind='scatter', x='pages', y='views', alpha=0.5)
plt.title('The correlation coef is only {0:.2f}'.format(corr_coef))

# the results show that the correlation is not so high and
# also there are some extreme value for the most popular documents


# I tried remove the extreme values with top five viewed documents to see the correlation
contents_items.sort_values(by='views', ascending=False, inplace=True)

corr_coef2 = contents_items[5:]['pages'].corr(contents_items[5:]['views'])
contents_items[5:].plot(kind='scatter', x='pages', y='views', alpha=0.5)
sb.lmplot(data=contents_items[5:], x='pages', y='views', ci = None)
plt.title('The correlation coef is now {0:.2f}'.format(corr_coef2))

# 5. Length of the title
data['title_len'] = data['title'].apply(lambda x: len(x.strip()))
data_sort_title = data.sort_values(by='views', ascending=False)

data_sort_title.plot(kind='scatter', x='title_len', y='views', logy=True).axvspan(20, 50, alpha=0.2, color='blue')
sb.kdeplot(data['title_len'], cumulative=False, shade=True)

# Any relationship with number of views?
sb.lmplot(data=data_sort_title[10:], x='title_len', y='views', robust=True)
# the chart showed no significant relationship
