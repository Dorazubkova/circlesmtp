#!/usr/bin/env python
# coding: utf-8

# ## 1. Импорт библиотек

# In[128]:


import bokeh
from bokeh.server.server import Server as server
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, show, output_notebook
from bokeh.tile_providers import Vendors, get_provider
import pandas as pd
import numpy as np
from bokeh.models import ColumnDataSource, HoverTool, LassoSelectTool, Label, Title, ZoomInTool, ZoomOutTool
from bokeh.models import Arrow, OpenHead, NormalHead, VeeHead
from bokeh.layouts import gridplot
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import ColumnDataSource, Figure
from bokeh.models.widgets import PreText, Paragraph, Select, Dropdown, RadioButtonGroup, RangeSlider, Slider, CheckboxGroup,HTMLTemplateFormatter,TableColumn
import bokeh.layouts as layout
from bokeh.models.widgets import Tabs, Panel, RadioGroup
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import shapely 
from shapely.geometry import Point
import geopandas as gpd
tile_provider = get_provider(Vendors.CARTODBPOSITRON)


# In[129]:


tile_provider


# ## 2. Чтение данных

# ### 2.1 Onoffmatrix

# In[130]:


onoffmatrix = pd.read_csv('myapp/onoffmatrix_avg.csv', sep = ';', encoding='cp1251')
onoffmatrix = onoffmatrix[['stop_id_from','stop_id_to','movements_norm', 'hour_on']]

#остановки-суперсайты
stops_supers = pd.read_csv('myapp/stops_supers.csv', sep = ';', encoding='cp1251')

onoffmatrix = pd.merge(onoffmatrix, stops_supers, how='inner', left_on = ['stop_id_from'], right_on = ['stop_id'])
onoffmatrix = pd.merge(onoffmatrix, stops_supers, how='inner', left_on = ['stop_id_to'], right_on = ['stop_id'])
onoffmatrix = onoffmatrix[['super_site_x','super_site_y','movements_norm','hour_on']].rename(columns = {'super_site_x':'super_site_from',
                                                                                   'super_site_y':'super_site_to'})
onoffmatrix = onoffmatrix.groupby(['super_site_from','super_site_to','hour_on']).sum().reset_index()

supers_Moscow = pd.read_csv('myapp/supers_Mercator.csv', sep = ';')
supers_Moscow = supers_Moscow.drop_duplicates()

onoffmatrix = pd.merge(onoffmatrix, supers_Moscow, how = 'inner', 
              left_on = ['super_site_from'], right_on=['super_site']).rename(columns={'X':'X_from','Y':'Y_from'})
onoffmatrix = pd.merge(onoffmatrix,  supers_Moscow, how = 'inner', 
           left_on = ['super_site_to'], right_on=['super_site']).rename(columns={'X':'X_to','Y':'Y_to'})
onoffmatrix = onoffmatrix[['super_site_from','super_site_to','movements_norm','X_from','Y_from','X_to','Y_to','hour_on']]
onoffmatrix['movements_norm'] = round(onoffmatrix['movements_norm'],2)

#сайты Тушино из
supers_T = pd.read_csv('myapp/supersites_Tushino.csv', sep = ';')

#onoffmatrix = pd.merge(onoffmatrix, supers_T, how='inner',left_on=['super_site_from'], right_on=['super_site'])
#onoffmatrix = onoffmatrix[onoffmatrix['movements_norm']>0.5]
onoffmatrix['movesize'] = round(onoffmatrix['movements_norm']/1, 0)
onoffmatrix_7 = onoffmatrix[onoffmatrix['hour_on'] == 7]
onoffmatrix_8 = onoffmatrix[onoffmatrix['hour_on'] == 8]


# ### 2.2 Odmatrix

# In[131]:


odmatrix = pd.read_csv('myapp/odmatrix_avg.csv', sep = ';', encoding='cp1251')
odmatrix = odmatrix[['site_id_from','site_id_to','movements_norm', 'hour_start']]

#сайты-суперсайты
sited_supers = pd.read_csv('myapp/sites_supers.csv', sep = ';', encoding='cp1251')

odmatrix = pd.merge(odmatrix, sited_supers, how='inner', left_on = ['site_id_from'], right_on = ['site_id'])
odmatrix = pd.merge(odmatrix, sited_supers, how='inner', left_on = ['site_id_to'], right_on = ['site_id'])
odmatrix = odmatrix[['super_site_x','super_site_y','movements_norm','hour_start']].rename(columns = {'super_site_x':'super_site_from',
                                                                              'super_site_y':'super_site_to', 'hour_start':'hour_on'})
odmatrix = odmatrix.groupby(['super_site_from','super_site_to','hour_on']).sum().reset_index()


odmatrix = pd.merge(odmatrix, supers_Moscow, how = 'inner', 
              left_on = ['super_site_from'], right_on=['super_site']).rename(columns={'X':'X_from','Y':'Y_from'})
odmatrix = pd.merge(odmatrix,  supers_Moscow, how = 'inner', 
           left_on = ['super_site_to'], right_on=['super_site']).rename(columns={'X':'X_to','Y':'Y_to'})
odmatrix = odmatrix[['super_site_from','super_site_to','movements_norm','X_from','Y_from','X_to','Y_to','hour_on']]
odmatrix['movements_norm'] = round(odmatrix['movements_norm'],2)

#odmatrix = pd.merge(odmatrix, supers_T, how='inner',left_on=['super_site_from'], right_on=['super_site'])
#odmatrix = odmatrix[odmatrix['movements_norm']>0.5]
odmatrix['movesize'] = round(odmatrix['movements_norm']/1, 0)
odmatrix_7 = odmatrix[odmatrix['hour_on'] == 7]
odmatrix_8 = odmatrix[odmatrix['hour_on'] == 8]


# In[182]:


odmatrix_7[odmatrix_7['super_site_from'] == 2703]['movements_norm'].sum()


# ### 2.3 Округа

# In[171]:


#соответствие округов и супер сайтов
supers_okrugs = pd.read_csv('myapp/supers_okrugs.csv', sep = ';', encoding='cp1251')
supers_okrugs = supers_okrugs.sort_values(['name_okrug'])
supers_okrugs['id'] = supers_okrugs.groupby(['name_okrug']).ngroup()

#названия округов
okrugs_names = list(supers_okrugs['name_okrug'].sort_values().drop_duplicates())

#названия супер сайтов
supers_names = pd.read_csv('myapp/supers_names.csv', sep = ';', encoding='cp1251')
supers_labels = pd.merge(supers_Moscow, supers_names, how = 'inner', on = ['super_site'])
supers_Moscow = pd.merge (supers_Moscow, supers_names, how = 'inner', on = ['super_site'])


# ## 3. Добавление карт, слоев и штучек

# ### 3.1 Создаем ColumnDataSourse

# #### 3.1.1 DataSourse для подписей

# In[134]:


cds_lb_from = dict(X=list(supers_Moscow['X'].values), 
                    Y=list(supers_Moscow['Y'].values),
                    super_site=list(supers_Moscow['super_site'].values),
                    super_site_name=list(supers_Moscow['super_site_name'].values))

source_lb_from = ColumnDataSource(data = cds_lb_from)

cds_lb_to = dict(X=list(supers_Moscow['X'].values), 
                    Y=list(supers_Moscow['Y'].values),
                    super_site=list(supers_Moscow['super_site'].values),
                    super_site_name=list(supers_Moscow['super_site_name'].values))

source_lb_to = ColumnDataSource(data = cds_lb_to)


# #### 3.1.2 DataSourse для матриц

# In[135]:


#пустой sourse, в который при выборе матрицы и округа запишутся данные
cds = dict(
                        X=[], 
                        Y=[],
                        super_site = [])

source_from = ColumnDataSource(data = cds)

source_to = ColumnDataSource(data = cds)

source_from2 = ColumnDataSource(data = cds)

source_to2 = ColumnDataSource(data = cds)

source_links_from = ColumnDataSource(data = cds)

source_links_to = ColumnDataSource(data = cds)


# ### 3.2 Создаем инструменты

# In[136]:


#инструменты
lasso_from = LassoSelectTool(select_every_mousemove=False)
lasso_to = LassoSelectTool(select_every_mousemove=False)

lasso_from2 = LassoSelectTool(select_every_mousemove=False)
lasso_to2 = LassoSelectTool(select_every_mousemove=False)

hover = HoverTool(tooltips=[("super_site_name", "@super_site_name")], names=["label"])

toolList_from = [lasso_from,  'reset',  'pan','wheel_zoom', hover]
toolList_to = [lasso_to,  'reset',  'pan', 'wheel_zoom', hover]

toolList_from2 = [lasso_from2, 'reset', 'pan','wheel_zoom', hover]
toolList_to2 = [lasso_to2,  'reset',  'pan','wheel_zoom', hover]

toolList_links = [lasso_to2,  'reset',  'pan','wheel_zoom', hover]


# ### 3.3 Рисуем MAP

# #### 3.3.1 График матрицы "ИЗ"

# ##### 3.3.1.1 График from

# In[137]:


#рисуем графики
p = figure(x_range=(3948598, 4354485), y_range=(7307581, 7725406), x_axis_type="mercator", y_axis_type="mercator", tools=toolList_from)
p.add_tile(tile_provider)
# p.add_layout(Title(text='Фильтр корреспонденций "ИЗ"', text_font_size='10pt', text_color = 'blue'), 'above')


#слой подписей
lb = p.circle(x = 'X',
         y = 'Y',
         source=source_lb_from,
        size=8,
        fill_color = 'lightgray',
        fill_alpha = 0.5,
        line_color = 'lightgray',
        line_alpha = 0.5,
        name = "label",
        nonselection_fill_color = 'lightgray',
        nonselection_fill_alpha = 0.5,
        nonselection_line_color = 'lightgray',
        nonselection_line_alpha = 0.5 )

#слой сайтов from
r = p.circle(x = 'X',
         y = 'Y',
         source=source_from,
        fill_color='navy',
        size=10,
        fill_alpha = 1,
        nonselection_fill_alpha=1,
        nonselection_fill_color='gray')

Time_Title1 = Title(text='Матрица: ', text_font_size='10pt', text_color = 'grey')
p.add_layout(Time_Title1, 'above')


# ##### 3.3.1.2 График to

# In[138]:


p_to = figure(x_range=(3948598, 4354485), y_range=(7307581, 7725406), x_axis_type="mercator", y_axis_type="mercator", tools=toolList_to)
p_to.add_tile(tile_provider)

lb_to = p_to.circle(x = 'X',
         y = 'Y',
         source=source_lb_to,
        size=8,
        fill_color = 'lightgray',
        fill_alpha = 0.5,
        line_color = 'lightgray',
        line_alpha = 0.5,
        name = "label",
        nonselection_fill_color = 'lightgray',
        nonselection_fill_alpha = 0.5,
        nonselection_line_color = 'lightgray',
        nonselection_line_alpha = 0.5 )


t = p_to.circle(x = 'X', 
                y = 'Y', 
                fill_color='papayawhip', 
                fill_alpha = 0.6, 
                line_color='tan', 
                line_alpha = 0.8, 
                size=6 , 
                source = source_to,
                nonselection_fill_alpha = 0.6, 
                nonselection_fill_color = 'papayawhip', 
                nonselection_line_color = None)


ds = r.data_source
tds = t.data_source


t_to = p_to.circle(x = [], y = [], fill_color=[], fill_alpha = 0.6, 
                    line_color= None, line_alpha = 0.8, size=[], nonselection_line_color = None,
                   nonselection_fill_color = 'papayawhip') 

l = p_to.text(x = [], y = [], text_color='black', text =[], text_font_size='8pt',
                         text_font_style = 'bold')

tds_to = t_to.data_source
lds=l.data_source


# #### 3.3.2 График матрицы "В"

# ##### 3.3.2.1 График to

# In[139]:


p2 = figure(x_range=(3948598, 4354485), y_range=(7307581, 7725406), x_axis_type="mercator", y_axis_type="mercator", tools=toolList_from2)
p2.add_tile(tile_provider)

# p2.add_layout(Title(text='Фильтр корреспонденций "В"', text_font_size='10pt', text_color = 'purple'), 'above')

lb2 = p2.circle(x = 'X',
         y = 'Y',
         source=source_lb_from,
        size=8,
        fill_color = 'lightgray',
        fill_alpha = 0.5,
        line_color = 'lightgray',
        line_alpha = 0.5,
        name = "label",
        nonselection_fill_color = 'lightgray',
        nonselection_fill_alpha = 0.5,
        nonselection_line_color = 'lightgray',
        nonselection_line_alpha = 0.5 )

r2 = p2.circle(x = 'X',
         y = 'Y',
         source=source_to2,
        fill_color='purple',
        size=10,
        fill_alpha = 1,
        line_color = 'purple',
        nonselection_fill_alpha=1,
        nonselection_fill_color='gray')

Time_Title2 = Title(text='Матрица: ', text_font_size='10pt', text_color = 'grey')
p2.add_layout(Time_Title2, 'above')


# ##### 3.3.2.2 График from

# In[140]:


p_from = figure(x_range=(3948598, 4354485), y_range=(7307581, 7725406), x_axis_type="mercator", y_axis_type="mercator", tools=toolList_to2)
p_from.add_tile(tile_provider)

lb_from = p_from.circle(x = 'X',
         y = 'Y',
         source=source_lb_to,
        size=8,
        fill_color = 'lightgray',
        fill_alpha = 0.5,
        line_color = 'lightgray',
        line_alpha = 0.5,
        name = "label",
        nonselection_fill_color = 'lightgray',
        nonselection_fill_alpha = 0.5,
        nonselection_line_color = 'lightgray',
        nonselection_line_alpha = 0.5 )

t2 = p_from.circle(x = 'X', 
                   y = 'Y', 
                   fill_color='papayawhip', 
                   fill_alpha = 0.6, 
                    line_color='tan', 
                   line_alpha = 0.8, 
                   size=6 , 
                   source = source_from2,
                  nonselection_fill_alpha = 0.6, 
                   nonselection_fill_color = 'papayawhip', 
                   nonselection_line_color = None)

t_from = p_from.circle(x = [], y = [], fill_color=[], fill_alpha = 0.6, 
                                line_color= None, line_alpha = 0.8, size=[], nonselection_line_color = None, 
                               nonselection_fill_alpha = 0.6, nonselection_fill_color = 'papayawhip' ) 
l_from = p_from.text(x = [], y = [], text_color='black', text =[], text_font_size='8pt',
                         text_font_style = 'bold')

tds_from = t_from.data_source
lds_from=l_from.data_source


ds2 = r2.data_source
tds2 = t2.data_source


# ### 3.4 Widgets

# In[142]:


#widgets
stats = Paragraph(text='', width=500, style={'color': 'blue'})
stats2 = Paragraph(text='', width=500, style={'color': 'purple'})
menu = [('onoffmatrix_7', 'onoffmatrix_7'), ('onoffmatrix_8', 'onoffmatrix_8'), ('odmatrix_7', 'odmatrix_7'),
       ('odmatrix_8', 'odmatrix_8')]
select1 = RadioGroup(labels=["Матрица ON_OFF", "Матрица OD"])
select2 = RadioGroup(labels=["Матрица ON_OFF", "Матрица OD"])
text_okrug = Paragraph(text='2. ВЫБЕРИТЕ ОКРУГ:', width=500, height=10, style={'color': 'white', 'background':'green'})
text_func = Paragraph(text='3. ВЫБЕРИТЕ ДЕЙСТВИЕ:', width=500, height=10, style={'color': 'white', 'background':'steelblue'})
button1 = RadioButtonGroup(labels=['Нарисовать кружочки','Посмотреть корреспонденции'], css_classes =['custom_button'])
button2 = RadioButtonGroup(labels=['Нарисовать кружочки','Посмотреть корреспонденции'], button_type  = 'primary')
slider1 = RangeSlider(start=0, end=100000, value=(0,10000), step=50, title="Сайты, с которых корреспонденции в диапазоне:", callback_policy="mouseup")
slider2 = RangeSlider(start=0, end=100000, value=(0,10000), step=50, title="Сайты, с которых корреспонденции в диапазоне:")
checkbox_group1 = CheckboxGroup(labels=okrugs_names, active=[])
checkbox_group2 = CheckboxGroup(labels=okrugs_names, active=[])


select3 = Dropdown(label="1. ВЫБЕРИТЕ МАТРИЦУ:", menu = menu, button_type  = 'danger')
text_okrug = Paragraph(text='2. ВЫБЕРИТЕ ОКРУГ:', width=500, height=10, style={'color': 'white', 'background':'green'})
text_func = Paragraph(text='3. ВЫБЕРИТЕ ДЕЙСТВИЕ:', width=500, height=10, style={'color': 'white', 'background':'steelblue'})
checkbox_group3 = CheckboxGroup(labels=okrugs_names, active=[])
slider3 = Slider(start=0, end=50, value=0, step=5, title="Топ связей:")


# ### 3.5 Вспомогательные функции

# #### 3.5.1 Запись названий матрицы

# In[143]:


prev_matrix_from = ['matrix']
def previous_matrix_from(matrix):
    prev_matrix_from.append(matrix)
    return prev_matrix_from

prev_matrix_to = ['matrix']
def previous_matrix_to(matrix):
    prev_matrix_to.append(matrix)
    return prev_matrix_to

prev_matrix_links = ['matrix']
def previous_matrix_links(matrix):
    prev_matrix_links.append(matrix)
    return prev_matrix_links


# #### 3.5.2 Запись названий округов

# In[144]:


prev_ok_from = ['okrug']
def previous_okrug_from(okrug):
    prev_ok_from.append(okrug)
    return prev_ok_from

prev_ok_to = ['okrug']
def previous_okrug_to(okrug):
    prev_ok_to.append(okrug)
    return prev_ok_to

prev_ok_links = ['okrug']
def previous_okrug_links(okrug):
    prev_ok_links.append(okrug)
    return prev_ok_links


# #### 3.5.3 Запись значений zoom

# In[145]:


dd_to = [600000]
def previous_to(d):    
    dd_to.append(d)
    return dd_to 

dd_from = [600000]
def previous_from(d):    
    dd_from.append(d)
    return dd_from   


# #### 3.5.4 Запись значений selected index

# In[146]:


index_to = [[0]]
def previous_idx_to(idx):
    index_to.append(idx)
    return index_to

index_from = [[0]]
def previous_idx_from(idx):
    index_from.append(idx)
    return index_from


# #### 3.5.5 Запись значений button

# In[147]:


bttn = [2]
def previous_but(but):
    bttn.append(but)
    return bttn


# #### 3.5.6 Определение уровня zoom

# In[148]:


def zoom_groups(x):
    if x > 601000:
        group = 0
    elif x >= 40000:
        group = 1
    elif x >= 20000:
        group = 2
    else:
        group = 3
    return group    


# #### 3.5.7 Вычисление "центра масс"

# In[149]:


def center(x, y, mass):
    sumMass = sum(mass)
    momentX = sum([x*y for x, y in zip(x, mass)])
    momentY = sum([x*y for x, y in zip(y, mass)])
    xCentr = momentX / sumMass
    yCentr = momentY / sumMass
    return [xCentr, yCentr]


# #### 3.5.8 Кластеризация координат

# In[150]:


def cluster_to(test, X, n, color):
    
    X_to_new = []
    Y_to_new = []
    
    kmeans = KMeans(n_clusters=int(np.ceil(len(test)/n)))
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    group_id = pd.Series(y_kmeans)
    test = test.reset_index(drop = True)
    test['group_id'] = group_id

    groups = test.groupby(['group_id'])
    
    test1 = gpd.GeoDataFrame(test)
    
    test1 = test1.dissolve(by = 'group_id')
    test1.geometry = test1.geometry.centroid
    test1['X_to_new'] = test1.geometry.x
    test1['Y_to_new'] = test1.geometry.y

#     for i, gr in groups:
#         w = center(gr.X_to, gr.Y_to, gr.text_sum)
#         X_to_new.append(w[0])
#         Y_to_new.append(w[1]) 

#     test1['X_to_new'] = X_to_new
#     test1['Y_to_new'] = Y_to_new
    test1['text_sum_new'] = list(test.groupby(['group_id'])['text_sum'].sum())
    test1['size_sum_new'] = list(test.groupby(['group_id'])['size_sum'].sum())

    new_data_text1 = dict()
    new_data_text1['x'] = list(test1['X_to_new'])
    new_data_text1['y'] = list(test1['Y_to_new'])
    new_data_text1['text'] = list(round(test1['text_sum_new'],2))

    new_data1 = dict()
    new_data1['x'] = list(test1['X_to_new'])
    new_data1['y'] = list(test1['Y_to_new'])
    new_data1['size'] = [x/3 for x in new_data_text1['text']]
    new_data1['fill_color'] = [color]*len(test1)
        
    return new_data1, new_data_text1



def cluster_from(test, X, n, color):
    
    X_to_from = []
    Y_to_from = []
    
    kmeans = KMeans(n_clusters=int(np.ceil(len(test)/n)))
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    group_id = pd.Series(y_kmeans)
    test = test.reset_index(drop = True)
    test['group_id'] = group_id

    groups = test.groupby(['group_id'])
    
    test1 = gpd.GeoDataFrame(test)
    
    test1 = test1.dissolve(by = 'group_id')
    test1.geometry = test1.geometry.centroid
    test1['X_from_new'] = test1.geometry.x
    test1['Y_from_new'] = test1.geometry.y

#     for i, gr in groups:
#         w = center(gr.X_to, gr.Y_to, gr.text_sum)
#         X_to_new.append(w[0])
#         Y_to_new.append(w[1]) 

#     test1['X_to_new'] = X_to_new
#     test1['Y_to_new'] = Y_to_new
    test1['text_sum_new'] = list(test.groupby(['group_id'])['text_sum'].sum())
    test1['size_sum_new'] = list(test.groupby(['group_id'])['size_sum'].sum())

    new_data_text1 = dict()
    new_data_text1['x'] = list(test1['X_from_new'])
    new_data_text1['y'] = list(test1['Y_from_new'])
    new_data_text1['text'] = list(round(test1['text_sum_new'],2))

    new_data1 = dict()
    new_data1['x'] = list(test1['X_from_new'])
    new_data1['y'] = list(test1['Y_from_new'])
    new_data1['size'] = [x/3 for x in new_data_text1['text']]
    new_data1['fill_color'] = [color]*len(test1)
        
    return new_data1, new_data_text1


# #### 3.5.9 Очистка ColumnDataSourse

# In[151]:


def clear():
    new_data_text1 = dict()
    new_data_text1['x'] = []
    new_data_text1['y'] = []
    new_data_text1['text'] = []

    new_data1 = dict()
    new_data1['x'] = []
    new_data1['y'] = []
    new_data1['size'] = []
    new_data1['fill_color'] = []
    
    return new_data1, new_data_text1


# #### 3.5.10 Очистка selected index

# In[152]:


def null_selection_to():
    source_to.selected.update(indices=[]) 

def null_selection_from():
    source_from.selected.update(indices=[]) 
    
def null_selection_to2():
    source_to2.selected.update(indices=[]) 

def null_selection_from2():
    source_from2.selected.update(indices=[]) 


# ### 3.6 CALLBACKS

# #### 3.6.1 Добавление сайтов на карту при выборе округа

# In[153]:


def update1(attrname, old, new):
    
    ok = checkbox_group1.active
    previous_okrug_from(ok)
        
    if prev_ok_from[-1] != prev_ok_from[-2]:
        new_data1, new_data_text1 = clear()  
        null_selection_from()
        null_selection_to()

    df = pd.merge(supers_Moscow, supers_okrugs, how = 'inner', on = ['super_site'])

    df1 = df[df['id'].isin(ok)]

    print(len(df1))
    print(len(df))
    
    print(df1.columns)

    cds_upd1 = dict(     X=list(df1['X'].values), 
                        Y=list(df1['Y'].values),
                   super_site = list(df1['super_site'].values))

    cds_upd2 = dict(     X=list(supers_Moscow['X'].values), 
                        Y=list(supers_Moscow['Y'].values),
                   super_site = list(supers_Moscow['super_site'].values))

    #1
    source_from_sl = ColumnDataSource(data = cds_upd1)
    source_from.data = source_from_sl.data

    #2
    source_to_sl = ColumnDataSource(data = cds_upd2)
    source_to.data = source_to_sl.data

checkbox_group1.on_change('active', update1)


# In[154]:


def update2(attrname, old, new):

    ok = checkbox_group2.active
    previous_okrug_to(ok)

    if prev_ok_to[-1] != prev_ok_to[-2]:
        new_data1, new_data_text1 = clear()  
        null_selection_from2()
        null_selection_to2()

    df = pd.merge(supers_Moscow, supers_okrugs, how = 'inner', on = ['super_site'])

    df1 = df[df['id'].isin(ok)]

    print(len(df1))

    cds_upd1 = dict(     X=list(df1['X'].values), 
                        Y=list(df1['Y'].values),
                   super_site = list(df1['super_site'].values))

    cds_upd2 = dict(     X=list(supers_Moscow['X'].values), 
                        Y=list(supers_Moscow['Y'].values),
                   super_site = list(supers_Moscow['super_site'].values))

    #3
    source_from_sl2 = ColumnDataSource(data = cds_upd2)
    source_from2.data = source_from_sl2.data

    #4
    source_to_sl2 = ColumnDataSource(data = cds_upd1)
    source_to2.data = source_to_sl2.data

checkbox_group2.on_change('active', update2)


# In[155]:


def update1_matrix_name(attrname, old, new):
    
    sl = select1.active
    print(sl)
    previous_matrix_from(sl)
    
    if prev_matrix_from[-1] != prev_matrix_from[-2]:
        new_data1, new_data_text1 = clear()  
        null_selection_from()
        null_selection_to()
    
    if sl == 0:
        matrix_name = 'ON_OFF'
    elif sl == 1:
        matrix_name = 'OD'

    Time_Title1.text = "Матрица " + matrix_name 

select1.on_change('active', update1_matrix_name)


# In[156]:


def update2_matrix_name(attrname, old, new):
    
    sl = select2.active
    print(sl)
    previous_matrix_to(sl)
    
    if prev_matrix_to[-1] != prev_matrix_to[-2]:
        new_data1, new_data_text1 = clear()  
        null_selection_from2()
        null_selection_to2()

    if sl == 0:
        matrix_name = 'ON_OFF'
    elif sl == 1:
        matrix_name = 'OD'
    
    Time_Title2.text = "Матрица " + matrix_name 

select2.on_change('active', update2_matrix_name)


# #### 3.6.2 Рисование кружков при выборе сайтов

# ##### 3.6.2.1 Графики from

# In[159]:


def callback(attrname, old, new): 

    but = button1.active
        
    if but == 0:
        
        print(but) 
        
        previous_but(but)
        
        if bttn[-1] != bttn[-2]:
            
            new_data1, new_data_text1 = clear()
            inters_idx = []   
            null_selection_from()
            null_selection_to()
            
        else:      
    
            idx = source_from.selected.indices

            if not idx:
                previous_idx_to([])
            else:
                previous_idx_to(idx)
                
            sl = select1.active
            print(sl)
            
            if sl == 0:
                matrix = onoffmatrix_7
            elif sl == 1:
                matrix = odmatrix_7

            #таблица с выбранными индексами 
            df = pd.DataFrame(data=ds.data).iloc[idx]
            
            print(df)
            
            df = pd.merge(df, matrix, how = 'inner', left_on = ['super_site'], 
                          right_on = ['super_site_from'])
            
            print(df.columns.values)
            print(df['movesize'].sum())

            #сумма movements по выделенным индексам
            df['size_sum'] = df.groupby(['X_to','Y_to'])['movesize'].transform(sum)
            df['text_sum'] = df.groupby(['X_to','Y_to'])['movesize'].transform(sum)

            x1 = p_to.x_range.start
            x2 = p_to.x_range.end
            y1 = p_to.y_range.start
            y2 = p_to.y_range.end

            d = ((x2-x1)**2 + (y2-y1)**2)**0.5 
            print('d = ', d)

            test = df.drop_duplicates(['X_to','Y_to'])

            stats.text = " "

            previous_to(d)

            new_data1 = dict()
            new_data_text1 = dict() 

            test = gpd.GeoDataFrame(test, geometry=[Point(xy) for xy in zip(test.X_to, test.Y_to)])

            X = test[['X_to', 'Y_to']].values

            if (zoom_groups(dd_to[-1]) == 0) | ((zoom_groups(dd_to[-1]) == 1) & (zoom_groups(dd_to[-2]) != 
                            zoom_groups(dd_to[-1]))) | ((zoom_groups(dd_to[-1]) == 1) & (index_to[-1] != 
                            index_to[-2])) | (index_to[-1] == []):

                try:
                    new_data1, new_data_text1 = cluster_to(test, X, 6, 'red')
                except:
                    new_data1, new_data_text1 = clear()
                    

            elif ((zoom_groups(dd_to[-1]) == 2) & (zoom_groups(dd_to[-2]) != zoom_groups(dd_to[-1])))  | ((zoom_groups(dd_to[-1]) == 
                        2) & (index_to[-1] != index_to[-2])) | (index_to[-1] == []):

                try:                
                    new_data1, new_data_text1 = cluster_to(test, X, 2, 'blue')                
                except:               
                    new_data1, new_data_text1 = clear()

            elif ((zoom_groups(dd_to[-1]) == 3) & (zoom_groups(dd_to[-2]) != zoom_groups(dd_to[-1])))  | ((zoom_groups(dd_to[-1]) == 
                                3) & (index_to[-1] != index_to[-2])) | (index_to[-1] == []):

                test1 = test

                new_data_text1 = dict()
                new_data_text1['x'] = list(test1['X_to'])
                new_data_text1['y'] = list(test1['Y_to'])
                new_data_text1['text'] = list(round(test1['text_sum'],2))

                new_data1 = dict()
                new_data1['x'] = list(test1['X_to'])
                new_data1['y'] = list(test1['Y_to'])
                new_data1['size'] = [x/3 for x in new_data_text1['text']]
                new_data1['fill_color'] = ['orange']*len(test1)

        if new_data1:

            tds_to.data = new_data1
            lds.data = new_data_text1

source_from.selected.on_change('indices', callback)
button1.on_change('active', callback)  
p_to.x_range.on_change('start', callback)  


# ##### 3.6.2.2 Графики to

# In[160]:


def callback2(attrname, old, new):
    
    idx = source_to2.selected.indices
    print(idx)
    
    but = button2.active
        
    if but == 0:
        
        print(but) 
        
        previous_but(but)
        
        if bttn[-1] != bttn[-2]:
            
            new_data1, new_data_text1 = clear()
            inters_idx = []   
            null_selection_from()
            null_selection_to()
            
        else:      
    
            idx = source_to2.selected.indices
            print(idx)

            if not idx:
                previous_idx_from([])
            else:
                previous_idx_from(idx)
                
            sl = select2.active
            print(sl)
            
            if sl == 0:
                matrix = onoffmatrix_7
            elif sl == 1:
                matrix = odmatrix_7
                
            print(matrix.head())

            #таблица с выбранными индексами 
            df = pd.DataFrame(data=ds2.data).iloc[idx]
            
            print(df)
            
            df = pd.merge(df, matrix, how = 'inner', left_on = ['super_site'], 
                          right_on = ['super_site_to'])
            
            print(df.columns.values)
            print(df['movesize'].sum())

            #сумма movements по выделенным индексам
            df['size_sum'] = df.groupby(['X_from','Y_from'])['movesize'].transform(sum)
            df['text_sum'] = df.groupby(['X_from','Y_from'])['movesize'].transform(sum)

            x1 = p_from.x_range.start
            x2 = p_from.x_range.end
            y1 = p_from.y_range.start
            y2 = p_from.y_range.end

            d = ((x2-x1)**2 + (y2-y1)**2)**0.5 
            print('d = ', d)

            test = df.drop_duplicates(['X_from','Y_from'])

            stats2.text = " "

            previous_from(d)

            new_data1 = dict()
            new_data_text1 = dict() 

            test = gpd.GeoDataFrame(test, geometry=[Point(xy) for xy in zip(test.X_from, test.Y_from)])

            X = test[['X_from', 'Y_from']].values

            if (zoom_groups(dd_from[-1]) == 0) | ((zoom_groups(dd_from[-1]) == 1) & (zoom_groups(dd_from[-2]) != 
                            zoom_groups(dd_from[-1]))) | ((zoom_groups(dd_from[-1]) == 1) & (index_from[-1] != 
                            index_from[-2])) | (index_from[-1] == []):

                try:
                    new_data1, new_data_text1 = cluster_from(test, X, 6, 'red')
                except:
                    new_data1, new_data_text1 = clear()


            elif ((zoom_groups(dd_from[-1]) == 2) & (zoom_groups(dd_from[-2]) != zoom_groups(dd_from[-1])))  | ((zoom_groups(dd_from[-1]) == 
                        2) & (index_from[-1] != index_from[-2])) | (index_from[-1] == []):

                try:                
                    new_data1, new_data_text1 = cluster_from(test, X, 2, 'blue')                
                except:               
                    new_data1, new_data_text1 = clear()

            elif ((zoom_groups(dd_from[-1]) == 3) & (zoom_groups(dd_from[-2]) != zoom_groups(dd_from[-1])))  | ((zoom_groups(dd_from[-1]) == 
                                3) & (index_from[-1] != index_from[-2])) | (index_from[-1] == []):

                test1 = test

                new_data_text1 = dict()
                new_data_text1['x'] = list(test1['X_from'])
                new_data_text1['y'] = list(test1['Y_from'])
                new_data_text1['text'] = list(round(test1['text_sum'],2))

                new_data1 = dict()
                new_data1['x'] = list(test1['X_from'])
                new_data1['y'] = list(test1['Y_from'])
                new_data1['size'] = [x/3 for x in new_data_text1['text']]
                new_data1['fill_color'] = ['orange']*len(test1)

        if new_data1:

            tds_from.data = new_data1
            lds_from.data = new_data_text1
            
        print(tds_from.data)


source_to2.selected.on_change('indices', callback2)
button2.on_change('active', callback2)  
p_from.x_range.on_change('start', callback2)  


# #### 3.6.3 Корреспонденции текстом

# ##### 3.6.3.1 Графики to

# In[161]:


def callback_to(attrname, old, new):
    
    but = button1.active
        
    if but == 1:
        
        previous_but(but)
        print ('2',bttn)
        if bttn[-1] != bttn[-2]:
            new_data1, new_data_text1 = clear()
            inters_idx = []   
            null_selection_from()
            null_selection_to()
        
        else:            

            idx2 = source_from.selected.indices
            idx_to = source_to.selected.indices

            inters_idx = list(set(idx2) & set(idx_to))

            print("Length of selected circles to: ", idx2)
            print("Length of selected circles to: ", inters_idx)

            #таблица с выбранными индексами 
            dff = pd.DataFrame(data=tds.data).loc[inters_idx]

            test = dff.drop_duplicates(['X_to','Y_to'])

            #сумма movements по выделенным индексам
            aaa = dff['text'].sum()
            print("size to: ", aaa)

            #сайты из
            sitesfrom = dff['sitesfrom'].drop_duplicates()
            sitesto = dff['sitesto'].drop_duplicates()

            new_data1 = dict()
            new_data1['x'] = list(test['X_to'])
            new_data1['y'] = list(test['Y_to'])
            new_data1['size'] = ['10']*len(test)
            new_data1['fill_color']=['lightsalmon']*len(test)
            new_data_text1 = dict()
            new_data_text1['x'] = []
            new_data_text1['y'] = []
            new_data_text1['text'] = []
            stats.text = "Из сайтов " + str(list(sitesfrom)) + " в сайты " + str(list(sitesto)) + " едет " + str(aaa) + " человек(а) в час"

        tds_to.data = new_data1
        lds.data = new_data_text1

button1.on_change('active', callback_to) 
source_to.selected.on_change('indices', callback_to)
source_from.selected.on_change('indices', callback_to)


# ##### 3.6.3.2 Графики from

# In[162]:


def callback_to2(attrname, old, new):
    
    but = button2.active
    
    if but == 1:

        previous_but(but)
        print ('2',bttn)
        if bttn[-1] != bttn[-2]:
            new_data1, new_data_text1 = clear()
            inters_idx = []   
            null_selection_from2()
            null_selection_to2()
            
        else:            

            idx2 = source_to2.selected.indices
            idx_to = source_from2.selected.indices

            inters_idx = list(set(idx2) & set(idx_to))

            print("Length of selected circles to: ", idx2)
            print("Length of selected circles to: ", inters_idx)

            #таблица с выбранными индексами 
            dff = pd.DataFrame(data=tds2.data).loc[inters_idx]
            print("Length of selected circles to: ", dff)

            test = dff.drop_duplicates(['X_from','Y_from'])

            #сумма movements по выделенным индексам
            aaa = dff['text'].sum()
            print("size to: ", aaa)

            #сайты из
            sitesfrom = dff['sitesfrom'].drop_duplicates()
            sitesto = dff['sitesto'].drop_duplicates() 
            
            new_data1 = dict()
            new_data1['x'] = list(test['X_from'])
            new_data1['y'] = list(test['Y_from'])
            new_data1['size'] = ['10']*len(test)
            new_data1['fill_color']=['lightsalmon']*len(test)
            new_data_text1 = dict()
            new_data_text1['x'] = []
            new_data_text1['y'] = []
            new_data_text1['text'] = []
            stats2.text = "В сайты " + str(list(sitesto)) + " из сайтов " + str(list(sitesfrom)) + " едет " + str(aaa) + " человек(а) в час"

        tds_from.data = new_data1
        lds_from.data = new_data_text1

source_from2.selected.on_change('indices', callback_to2) 
source_to2.selected.on_change('indices', callback_to2)
button2.on_change('active', callback_to2)


# ### 3.7 Layouts

# In[166]:


layout1 = layout.row(p,p_to)
layout2 = layout.row(p2, p_from)
layout3 = layout.column(select1, text_okrug, checkbox_group1, text_func, button1, stats)
layout4 = layout.column(select2, text_okrug, checkbox_group2, text_func, button2, stats2)

layout5 = layout.row(layout1, layout3)
layout6 = layout.row(layout2, layout4)


# In[168]:


tab1 = Panel(child=layout5,title='Фильтр корреспонденций "ИЗ"')
tab2 = Panel(child=layout6,title='Фильтр корреспонденций "В"')
tabs = Tabs(tabs=[tab1, tab2])

doc = curdoc() 
doc.add_root(tabs)


# In[169]:


# apps = {'/': Application(FunctionHandler(make_document))}

# server = server(apps, port=5001)
# server.start()


# if __name__ == '__main__':
#     print('Opening Bokeh application on http://localhost:5006/')

# server.io_loop.add_callback(server.show, "/")


# In[ ]:




