#!/usr/bin/env python
# coding: utf-8

# In[169]:


from bokeh.io import show, output_notebook
from bokeh.plotting import figure, show, output_notebook
from bokeh.tile_providers import CARTODBPOSITRON
import bokeh
import pandas as pd
import os
import sys
from bokeh.models import ColumnDataSource, HoverTool, LassoSelectTool
from bokeh.layouts import gridplot
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import ColumnDataSource, Figure
from bokeh.models import Title
import bokeh.layouts as layout


# In[170]:


matrix = pd.read_csv('myapp/onoffmatrix_avg.csv', sep = ';', encoding='cp1251')
matrix = matrix[matrix['hour_on']==7]
matrix = matrix[['stop_id_from','stop_id_to','movements_norm']]

#остановки-суперсайты
stops_supers = pd.read_csv('myapp/stops_supers.csv', sep = ';', encoding='cp1251')

matrix = pd.merge(matrix, stops_supers, how='inner', left_on = ['stop_id_from'], right_on = ['stop_id'])
matrix = pd.merge(matrix, stops_supers, how='inner', left_on = ['stop_id_to'], right_on = ['stop_id'])
matrix = matrix[['super_site_x','super_site_y','movements_norm']].rename(columns = {'super_site_x':'super_site_from',
                                                                                   'super_site_y':'super_site_to'})
matrix = matrix.groupby(['super_site_from','super_site_to']).sum().reset_index()

supers_Moscow = pd.read_csv('myapp/supers_Mercator.csv', sep = ';')
supers_Moscow = supers_Moscow.drop_duplicates()

links = pd.merge(matrix, supers_Moscow, how = 'inner', 
              left_on = ['super_site_from'], right_on=['super_site']).rename(columns={'X':'X_from','Y':'Y_from'})
links = pd.merge(links,  supers_Moscow, how = 'inner', 
           left_on = ['super_site_to'], right_on=['super_site']).rename(columns={'X':'X_to','Y':'Y_to'})
links = links[['super_site_from','super_site_to','movements_norm','X_from','Y_from','X_to','Y_to']]
links['movements_norm'] = links['movements_norm']/3
links['movements_norm'] = round(links['movements_norm'],0)

#сайты Тушино из
supers_T = pd.read_csv('myapp/supersites_Tushino.csv', sep = ';')

links = pd.merge(links, supers_T, how='inner',left_on=['super_site_from'], right_on=['super_site'])
links = links[links['movements_norm']>10]


# In[171]:


source_from = ColumnDataSource(data = dict(
                        X_from=list(links['X_from'].values), 
                        Y_from=list(links['Y_from'].values),
                        size=list(links['movements_norm'].values),
                        X_to=list(links['X_to'].values), 
                        Y_to=list(links['Y_to'].values)
))


source_to = ColumnDataSource(data = dict(
                        X_from=list(links['X_from'].values), 
                        Y_from=list(links['Y_from'].values),
                        size=list(links['movements_norm'].values),
                        X_to=list(links['X_to'].values), 
                        Y_to=list(links['Y_to'].values)
))



source_from2 = ColumnDataSource(data = dict(
                        X_from=list(links['X_from'].values), 
                        Y_from=list(links['Y_from'].values),
                        size=list(links['movements_norm'].values),
                        X_to=list(links['X_to'].values), 
                        Y_to=list(links['Y_to'].values)
))


source_to2 = ColumnDataSource(data = dict(
                        X_from=list(links['X_from'].values), 
                        Y_from=list(links['Y_from'].values),
                        size=list(links['movements_norm'].values),
                        X_to=list(links['X_to'].values), 
                        Y_to=list(links['Y_to'].values)
))


source_from_labels = ColumnDataSource(data = dict(
                        X_from=list(links[['X_from', 'Y_from','super_site_from']].drop_duplicates()['X_from'].values), 
                        Y_from=list(links[['X_from', 'Y_from','super_site_from']].drop_duplicates()['Y_from'].values),
                        label=list(links[['X_from', 'Y_from','super_site_from']].drop_duplicates()['super_site_from'].values)
))






hover = HoverTool(tooltips=[('site_id','@label')])

lasso_from = LassoSelectTool(select_every_mousemove=True)
lasso_to = LassoSelectTool(select_every_mousemove=False)


toolList_from = [lasso_from, 'tap', 'reset', 'save', 'pan','wheel_zoom']
toolList_to = [lasso_to, 'tap', 'reset', 'save', 'pan','wheel_zoom']


p = figure(x_range=(4157975.01546188769862056 , 4173827.06850233720615506), y_range=(7521739.63348639197647572,  7533621.55124872922897339),
          x_axis_type="mercator", y_axis_type="mercator", tools=toolList_from)
p.add_tile(CARTODBPOSITRON)

p.add_layout(Title(text='Фильтр корреспонденций "ИЗ"', text_font_size='10pt'), 'above')

p.circle(x = 'X_from',
         y = 'Y_from',
         source=source_from_labels,
        fill_color='black',
        size=5)

r = p.circle(x = 'X_from',
         y = 'Y_from',
         source=source_from,
        fill_color='navy',
        size=10,
        fill_alpha = 1,
        nonselection_fill_alpha=0.4)



p_to = figure(x_range=(4157975.01546188769862056 , 4173827.06850233720615506), y_range=(7521739.63348639197647572,  7533621.55124872922897339),
          x_axis_type="mercator", y_axis_type="mercator", tools=toolList_to)
p_to.add_tile(CARTODBPOSITRON)
t = p_to.circle(x = 'X_to', y = 'Y_to', fill_color='red', fill_alpha = 0.6, 
                            line_color='red', line_alpha = 0.8, size=6 , source = source_to)

ds = r.data_source
tds = t.data_source


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[172]:


def callback(attrname, old, new):

    idx = source_from.selected.indices

    print("Indices of selected circles from: ", idx)
    print("Lenght of selected circles from: ", len(idx))

    #таблица с выбранными индексами 
    df = pd.DataFrame(data=ds.data).iloc[idx]
    #сумма movements по выделенным индексам
    aa = df.groupby(['X_to','Y_to'])['size'].transform(sum)


    p_to = figure(x_range=(4157975.01546188769862056 , 4173827.06850233720615506), 
                  y_range=(7521739.63348639197647572,  7533621.55124872922897339),
                  x_axis_type="mercator", y_axis_type="mercator", tools=toolList_to)
    p_to.add_tile(CARTODBPOSITRON)
    t = p_to.circle(x = 'X_to', y = 'Y_to', fill_color='red', fill_alpha = 0.6, 
                            line_color='red', line_alpha = 0.8, size=6 , source = source_to)


    if not idx: #если пустое выделение

        layout1.children[1] = p_to #обновить график справа

    else: #если не пустое выделение

        for x in idx: #для каждого выделенного индекса рисуем site_to и его параметры

            new_data = dict()
            new_data['x'] = [ds.data['X_to'][x]]
            new_data['y'] = [ds.data['Y_to'][x]]
            new_data['size'] = [aa[x]]
            new_data['index'] = [x]


            t_to = p_to.circle(x = [], y = [], fill_color='orange', fill_alpha = 0.6, 
                            line_color='red', line_alpha = 0.8, size=[] )
            tds_to=t_to.data_source
            tds_to.data = new_data

            #текст

            new_data_text = dict()
            new_data_text['x'] = [ds.data['X_to'][x]]
            new_data_text['y'] = [ds.data['Y_to'][x]]
            new_data_text['text'] = [aa[x]]


            l = p_to.text(x = [], y = [], text_color='black', text =[], text_font_size='8pt',
                         text_font_style = 'bold')
            lds=l.data_source
            lds.data = new_data_text


            layout1.children[1] = p_to #обновить график справа
    
            
        def callback_to(attrname, old, new):

            idx_to = source_to.selected.indices

            inters_idx = list(set(idx) & set(idx_to))


            print("Indices of selected circles to: ", inters_idx)
            print("Length of selected circles to: ", len(inters_idx))


            #таблица с выбранными индексами 
            dff = pd.DataFrame(data=tds.data).loc[inters_idx]
            print("Length of selected circles to: ", dff)
            
            #сумма movements по выделенным индексам
#             aaa = dff['size'].sum()
#             print("size to: ", aaa)
            
#             xsum = dff['X_to'].sum()
#             ysum = dff['Y_to'].sum()
#             count = len(dff)
            
#             if count != 0:
                

#                 xcentr = xsum/count
#                 ycentr = ysum/count
                
#             else:
#                 xcentr = 0
#                 ycentr = 0
                   
                
#             new_data_to = dict()
#             new_data_to['x'] = [xcentr]
#             new_data_to['y'] = [ycentr]
#             new_data_to['size'] = [aaa]

#             t_to2 = p_to.circle(x = [], y = [], fill_color='green', fill_alpha = 0.6, 
#                             line_color='red', line_alpha = 0.8, size=[] )
#             tds_to2=t_to2.data_source
#             tds_to2.data = new_data_to

#             layout1.children[1] = p_to #обновить график справа
                
                
      

        source_to.selected.on_change('indices', callback_to)






source_from.selected.on_change('indices', callback)



layout1 = layout.row(p,p_to)

# Create Tabs
line = layout.column(layout1)


curdoc().add_root(line)


# In[173]:



# In[ ]:




