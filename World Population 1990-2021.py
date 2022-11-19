#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import folium
import plotly.graph_objects as go

import geopandas as gpd
# import sys
# sys.maxsize


# ### 1. Import libraries

# In[2]:


# import sys, csv, ctypes as ct


# In[3]:


# "v{:d}.{:d}.{:d}".format(*sys.version_info[:3]), sys.platform, sys.maxsize, ct.sizeof(ct.c_void_p) * 8, ct.sizeof(ct.c_long) * 8


# In[4]:


# csv.field_size_limit()


# In[5]:


# csv.field_size_limit(int(ct.c_ulong(-1).value // 2))


# In[6]:


# limit1 = csv.field_size_limit()
# limit1


# In[7]:


# "0x{0:016X}".format(limit1)
# '0x7FFFFFFFFFFFFFFF'


# ### 2. Data Check

#  About dataset:
# 
# 

# In[8]:


# https://datacatalog.worldbank.org/search/dataset/0037712
#https://databank.worldbank.org/reports.aspx?source=2&series=SP.POP.TOTL&country=#


# In[9]:


df = pd.read_excel("C:\\Users\\kurzy\\Desktop\\Studia podyplomowe\\moje projekciki\\World_Development_Indicators_tys.xlsx")
data = df.copy()
data.head(10)


# In[10]:


data.columns


# In[11]:


data.info()


# In[12]:


data.shape


# In[13]:


data.isna().sum()


# In[14]:


data.describe()


# ### 3. Data cleaning and preprocessing

# In[15]:


# drop duplicates
data.drop_duplicates(inplace=True)


# In[16]:


# drop NaN
data.dropna(inplace=True)


# In[17]:


# rename columns with year 
data.rename(columns={'1990 [YR1990]': '1990',
                   '2000 [YR2000]': '2000', 
                   '2012 [YR2012]': '2012',
                   '2013 [YR2013]': '2013',
                   '2014 [YR2014]': '2014',
                   '2015 [YR2015]': '2015',
                   '2016 [YR2016]': '2016',
                   '2017 [YR2017]': '2017',
                   '2018 [YR2018]': '2018',
                   '2019 [YR2019]': '2019',
                   '2020 [YR2020]': '2020',
                   '2021 [YR2021]': '2021'}, inplace=True)


# In[18]:


data['Series Name'].unique()


# In[19]:


data.drop(columns=['Series Name', 'Series Code'], inplace= True)


# In[20]:


data['Country Name'] =data['Country Name'].astype('str') 
data['Country Code'] =data['Country Code'].astype('str') 


# In[21]:


# Add column region 

#country_code  = pd.read_csv('https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv', usecols = ['name', 'alpha-3', 'region'])
country_code = pd.read_excel('C:\\Users\\kurzy\\Desktop\Studia podyplomowe\\moje projekciki\\country_coordinates.xlsx')
#country_code.rename(columns={'Alpha-3 code' : 'Country Code', 'Country' : 'Country Name'}, inplace = True)
#country_code['Latitude (average)'].astype('int')
country_code.head()


# In[22]:



country_code.info()


# In[23]:


# filter data and obtaine information only about countries

final_data = pd.merge(data, country_code, how= 'inner')
#final_data.drop(columns = '', inplace = True)
final_data


# In[24]:


# convert from thousnands to milions

final_data['1990'] = round(final_data['1990']/1000, 2)
final_data['2000'] = round(final_data['2000']/1000, 2)
final_data['2012'] = round(final_data['2012']/1000, 2)
final_data['2013'] = round(final_data['2013']/1000, 2)
final_data['2014'] = round(final_data['2014']/1000, 2)
final_data['2015'] = round(final_data['2015']/1000, 2)
final_data['2016'] = round(final_data['2016']/1000, 2)
final_data['2017'] = round(final_data['2017']/1000, 2)
final_data['2018'] = round(final_data['2018']/1000, 2)
final_data['2019'] = round(final_data['2019']/1000, 2)
final_data['2020'] = round(final_data['2020']/1000, 2)
final_data['2021'] = round(final_data['2021']/1000, 2)


# In[25]:


# wzrost liczby ludnosci od 1990-2021

final_data['Population growth 1990-2021'] = final_data['2021'] - final_data['1990']
final_data.head()


# In[26]:


# przesunac region przed 1990


# In[27]:


# Transpose data frame for some analysis
final_dataa = final_data.copy()
Tdata = final_dataa.drop(columns=['Country Code', 'Population growth 1990-2021', 'Latitude', 'Longitude'], inplace= True)
Tdata = final_dataa.set_index('Country Name')
Tdata = Tdata.T
# Tdata.reset_index(inplace=True)
#Tdata= Tdata.rename(columns={"index":"year"})
Tdata


# ### 4. Exploring data

# In[28]:


# Data frame with information about world population

world_population = data[data['Country Name'] == 'World']
world_population = world_population.drop(columns=['Country Name', 'Country Code', '2000' , '1990'])
world_population = world_population.T
world_population.reset_index(inplace=True)
world_population = world_population.rename(columns = {'index':'Year', 263 : 'Population [mln]'})
world_population['Population [mln]'] = round(world_population['Population [mln]']/1000, 2)
world_population


# In[29]:


# Data frame with information about EU population

EU_population = data[data['Country Name'] == 'European Union']
EU_population = EU_population.drop(columns=['Country Name', 'Country Code', '1990', '2000'])
EU_population = EU_population.T
EU_population.reset_index(inplace=True)
EU_population = EU_population.rename(columns = {'index':'Year', 229 : 'Population [mln]'})
EU_population['Population [mln]'] = round(EU_population['Population [mln]']/1000, 2)
EU_population


# In[30]:


# Data frame with top 30 Countries in 2021

top30 = final_data.sort_values('2021', ascending= False)
top30 = top30[:30]
top30


# In[31]:


# Top 30 population growth countries

top30_population_growth = final_data.sort_values('Population growth 1990-2021', ascending= False)[:30]
top30_population_growth


# ### 5. Visualizations

# In[32]:


# world population LINE CHART in plt

year = world_population['Year']
population = world_population['Population [mln]']
    
csfont = {'fontname' : 'Verdana'}

fig, ax = plt.subplots(figsize=(17, 10))

plt.plot(year, population, 
         color ='#cc00ff',
         linestyle = '-',
         marker ='p',
        markerfacecolor='#ff8080',
        markeredgecolor='#ff8080',
        markersize = 10,
        )

            
plt.title('World population from 2012 to 2021', fontsize=20, **csfont)
plt.xlabel('Year', fontsize = 15, labelpad=15, **csfont)
plt.ylabel('Population [mln]', fontsize=15, labelpad=15, **csfont)
plt.xticks(rotation=0, fontsize=12, **csfont)
plt.yticks(rotation=0, fontsize=12, **csfont)
plt.rcParams['axes.facecolor']='#ebfaeb'

for index in range(len(year)):
  ax.text(year[index], population[index], population[index], size=12)


plt.show()


# In[33]:


# European Union population LINE CHART in plt

year_EU= EU_population['Year']
population_EU = EU_population['Population [mln]']


csfont = {'fontname' : 'Verdana'}

fig, ax = plt.subplots(figsize=(15, 6))

plt.plot(year_EU, population_EU, 
         color ='#cc00ff',
         linestyle = '--',
         marker ='p',
        markerfacecolor ='#ff8080',
        markeredgecolor= '#ff8080',
        markersize = 10,
        )
            
plt.title('European Union population from 2012 to 2021', fontsize = 20, **csfont)
plt.xlabel('Year', fontsize = 15, labelpad = 15, **csfont)
plt.ylabel('Population [mln]', fontsize= 15, labelpad = 15, **csfont)
plt.xticks(rotation=0, fontsize= 12, **csfont)
plt.yticks(rotation=0, fontsize= 12, **csfont)
plt.rcParams['axes.facecolor'] = '#ebfaeb'

for index in range(len(year_EU)):
  ax.text(year_EU[index], population_EU[index], population_EU[index], size=12)


plt.show()


# In[34]:


# European Union population  PLOTLY EXPRESS CHART LINE

fig = px.line(EU_population, x='Year', y='Population [mln]', 
              title='European Union population',
              labels={'index' : 'Year', 'value' : 'Population [mln]'},
              markers=True, )
            

fig.update_traces(patch={'line': {'dash':'solid',
                                  'shape':'spline', 
                                  'color':'green',
                                  'width':1}})
#fig.update_layout('line':{})


fig.show()


# In[35]:


# Top 30 countries by population in 2021  HISTOGRAM 

#sns.set_style('darkgrid')

csfont = {'fontname' : 'Verdana'}

plt.figure(figsize=(17, 10))

ax = sns.barplot(
    x='Country Name', 
    y='2021', 
    data=top30,
    estimator=sum, 
    ci=None, 
    palette='coolwarm'
)

plt.title('Top 30 countries by population in 2021', fontsize=20, **csfont)
plt.xlabel("")
plt.ylabel('Population [mln]', fontsize=15, labelpad=15, **csfont)
plt.xticks(rotation=90, fontsize=12, **csfont)
plt.yticks(rotation=90, fontsize=12, **csfont)
plt.rcParams['axes.facecolor']='#ebfaeb'

#add values to bars
for i in ax.containers:
    ax.bar_label(i,)
  


# In[36]:


#top30 countries with the highest population growth

#sns.set_style('darkgrid')

csfont={'fontname' : 'Verdana'}

plt.figure(figsize=(17, 10))

ax = sns.barplot(
    x='Country Name', 
    y='Population growth 1990-2021', 
    data=top30_population_growth,
    estimator=sum, 
    ci=None, 
    palette= 'coolwarm'
)

plt.title('Top 30 countries with the highest population growth from 1990 to 2021', fontsize = 20, **csfont)
plt.xlabel("")
plt.ylabel('Population [mln]', fontsize=15, labelpad=15, **csfont)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(rotation=90, fontsize=12)
plt.rcParams['axes.facecolor'] = '#ebfaeb'

#add values to bars
for i in ax.containers:
    ax.bar_label(i,)
  


# In[37]:


# Population growth in European countries

fig = px.line(Tdata, x= Tdata.index, y=['Poland', 'France', 'Spain', 'Germany', 'Finland', 
                                        'Italy', 'Greece', 'Austria', 'Denmark', 'Croatia', 'Netherlands'],
             title='Population growth in European countries',
             labels={'index' : 'Year', 'value' : 'Population [mln]', 'variable' : 'Country'},
              markers= True,
                )
                           

fig.update_traces(patch={'line': {'dash':'solid',
                                  'shape':'spline', 
                                  'width':1}})


fig.show()


# In[43]:


# map of Top 30 countries by population in 2021 
# with Folium and using Latitude and Longitude

m = folium.Map(location=[10,0], tiles='OpenStreetMap', zoom_start=2)

for i in range(0,len(top30)):
   folium.Circle(
      location=[top30_population_growth.iloc[i]['Latitude'], top30_population_growth.iloc[i]['Longitude']],
      popup=top30_population_growth.iloc[i]['Country Name'],
      radius=float(top30_population_growth.iloc[i]['2021']*1000),
      color='crimson',
      fill=False,
      fill_color='crimson',
    
   ).add_to(m)

m


# In[44]:


# map of Top 30 countries with the highest population growth from 1990 to 2021 
# with Choropleth and using County Code 

fig = go.Figure(data=go.Choropleth(
    locations = top30_population_growth['Country Code'],
    z = top30_population_growth['Population growth 1990-2021'],
    text = top30_population_growth['Country Code'],
    colorscale = 'plasma',
    autocolorscale=False,
    reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_title = 'Population [mln]')
               )

fig.update_layout(
    title_text='Top 30 countries with the highest population growth from 1990 to 2021',
    title_x = 0.70,
    geo=dict(showframe=False, showcoastlines=False)) 

fig.show()


# In[40]:


# machine learning - ile bedzie ludnosci w 2050, 2100, 2500, 3000 w konkretnych Panstwach i na swiecie

