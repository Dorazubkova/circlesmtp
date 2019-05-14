#!/usr/bin/env python
# coding: utf-8

# In[135]:


from bokeh.server.server import Server as server
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, show, output_notebook
from bokeh.tile_providers import CARTODBPOSITRON
import bokeh
import pandas as pd
import os
import sys
from bokeh.models import ColumnDataSource, HoverTool, LassoSelectTool, Label, Title
from bokeh.layouts import gridplot
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import ColumnDataSource, Figure
from bokeh.models.widgets import PreText, Paragraph, Select, Dropdown, RadioButtonGroup
import bokeh.layouts as layout
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
output_notebook()


# In[136]:


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
onoffmatrix['movements_norm'] = round(onoffmatrix['movements_norm'],0)

#сайты Тушино из
supers_T = pd.read_csv('myapp/supersites_Tushino.csv', sep = ';')

onoffmatrix = pd.merge(onoffmatrix, supers_T, how='inner',left_on=['super_site_from'], right_on=['super_site'])
onoffmatrix = onoffmatrix[onoffmatrix['movements_norm']>20]
onoffmatrix['movesize'] = round(onoffmatrix['movements_norm']/3, 0)
onoffmatrix_7 = onoffmatrix[onoffmatrix['hour_on'] == 7]
onoffmatrix_8 = onoffmatrix[onoffmatrix['hour_on'] == 8]


# In[ ]:





# In[137]:


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
odmatrix['movements_norm'] = round(odmatrix['movements_norm'],0)

odmatrix = pd.merge(odmatrix, supers_T, how='inner',left_on=['super_site_from'], right_on=['super_site'])
odmatrix = odmatrix[odmatrix['movements_norm']>20]
odmatrix['movesize'] = round(odmatrix['movements_norm']/3, 0)
odmatrix_7 = odmatrix[odmatrix['hour_on'] == 7]
odmatrix_8 = odmatrix[odmatrix['hour_on'] == 8]


# In[138]:


supers_names = pd.read_csv('myapp/supers_names.csv', sep = ';', encoding='cp1251')
supers_labels = pd.merge(supers_Moscow, supers_names, how = 'inner', on = ['super_site'])


# In[139]:


cds_lb_from = dict(X_from=list(supers_labels['X'].values), 
            Y_from=list(supers_labels['Y'].values),
            super_site=list(supers_labels['super_site'].values),
            super_site_name=list(supers_labels['super_site_name'].values))

source_lb_from = ColumnDataSource(data = cds_lb_from)



cds_lb_to = dict(X_to=list(supers_labels['X'].values), 
            Y_to=list(supers_labels['Y'].values),
            super_site=list(supers_labels['super_site'].values),
            super_site_name=list(supers_labels['super_site_name'].values))

source_lb_to = ColumnDataSource(data = cds_lb_to)


# In[140]:


cds = dict(
                        X_from=[], 
                        Y_from=[],
                        size=[],
                        X_to=[], 
                        Y_to=[],
                        sitesfrom=[],
                        sitesto=[])


# In[141]:


source_from = ColumnDataSource(data = cds)

source_to = ColumnDataSource(data = cds)

source_from2 = ColumnDataSource(data = cds)

source_to2 = ColumnDataSource(data = cds)


# In[142]:


lasso_from = LassoSelectTool(select_every_mousemove=False)
lasso_to = LassoSelectTool(select_every_mousemove=False)

lasso_from2 = LassoSelectTool(select_every_mousemove=False)
lasso_to2 = LassoSelectTool(select_every_mousemove=False)

hover = HoverTool(tooltips=[("super_site_name", "@super_site_name")], names=["label"])

toolList_from = [lasso_from, 'tap', 'reset', 'save', 'pan','wheel_zoom', hover]
toolList_to = [lasso_to, 'tap', 'reset', 'save', 'pan','wheel_zoom', hover]

toolList_from2 = [lasso_from2, 'tap', 'reset', 'save', 'pan','wheel_zoom', hover]
toolList_to2 = [lasso_to2, 'tap', 'reset', 'save', 'pan','wheel_zoom', hover]


p = figure(x_range=(4157975.01546188769862056 , 4173827.06850233720615506), y_range=(7521739.63348639197647572,  7533621.55124872922897339),
          x_axis_type="mercator", y_axis_type="mercator", tools=toolList_from)
p.add_tile(CARTODBPOSITRON)

p.add_layout(Title(text='Фильтр корреспонденций "ИЗ"', text_font_size='10pt', text_color = 'blue'), 'above')

lb = p.circle(x = 'X_from',
         y = 'Y_from',
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

r = p.circle(x = 'X_from',
         y = 'Y_from',
         source=source_from,
        fill_color='navy',
        size=10,
        fill_alpha = 1,
        nonselection_fill_alpha=1,
        nonselection_fill_color='gray')

p_to = figure(x_range=(4157975.01546188769862056 , 4173827.06850233720615506), y_range=(7521739.63348639197647572,  7533621.55124872922897339),
          x_axis_type="mercator", y_axis_type="mercator", tools=toolList_to)
p_to.add_tile(CARTODBPOSITRON)

Time_Title = Title(text='Матрица: ', text_font_size='10pt', text_color = 'grey')
p.add_layout(Time_Title, 'above')


lb_to = p_to.circle(x = 'X_to',
         y = 'Y_to',
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


t = p_to.circle(x = 'X_to', y = 'Y_to', fill_color='red', fill_alpha = 0.6, 
                line_color='red', line_alpha = 0.8, size=6 , source = source_to,
                   nonselection_fill_alpha = 0.6, nonselection_fill_color = 'red')



ds = r.data_source
tds = t.data_source




p2 = figure(x_range=(4157975.01546188769862056 , 4173827.06850233720615506), y_range=(7521739.63348639197647572,  7533621.55124872922897339),
              x_axis_type="mercator", y_axis_type="mercator", tools=toolList_from2)
p2.add_tile(CARTODBPOSITRON)

p2.add_layout(Title(text='Фильтр корреспонденций "В"', text_font_size='10pt', text_color = 'purple'), 'above')

lb2 = p2.circle(x = 'X_from',
         y = 'Y_from',
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

r2 = p2.circle(x = 'X_to',
         y = 'Y_to',
         source=source_to2,
        fill_color='purple',
        size=10,
        fill_alpha = 1,
        line_color = 'purple',
        nonselection_fill_alpha=1,
        nonselection_fill_color='gray')


p_from = figure(x_range=(4157975.01546188769862056 , 4173827.06850233720615506), y_range=(7521739.63348639197647572,  7533621.55124872922897339),
          x_axis_type="mercator", y_axis_type="mercator", tools=toolList_to2)
p_from.add_tile(CARTODBPOSITRON)


lb_from = p_from.circle(x = 'X_to',
         y = 'Y_to',
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

t2 = p_from.circle(x = 'X_from', y = 'Y_from', fill_color='red', fill_alpha = 0.6, 
                            line_color='red', line_alpha = 0.8, size=6 , source = source_from2)


ds2 = r2.data_source
tds2 = t2.data_source


# In[143]:


#widgets
stats = Paragraph(text='', width=250)
stats2 = Paragraph(text='', width=250)
menu = [('onoffmatrix_7', 'onoffmatrix_7'), ('onoffmatrix_8', 'onoffmatrix_8'), ('odmatrix_7', 'odmatrix_7'),
       ('odmatrix_8', 'odmatrix_8')]
select = Dropdown(label="Выберите матрицу: ", menu = menu)

button1 = RadioButtonGroup(labels=['Нарисовать кружочки','Посмотреть корреспонденции'])


def update(attrname, old, new):
    
    sl = select.value
    print(sl)
    df = pd.DataFrame(data = eval(sl))
    print(df.columns.values)
    
    cds_upd = dict(     X_from=list(df['X_from'].values), 
                        Y_from=list(df['Y_from'].values),
                        size=list(df['movesize'].values),
                        X_to=list(df['X_to'].values), 
                        Y_to=list(df['Y_to'].values),
                        sitesfrom=list(df['super_site_from'].values),
                        sitesto=list(df['super_site_to'].values),
                        text=list(df['movements_norm'].values))

    #1
    source_from_sl = ColumnDataSource(data = cds_upd)
    source_from.data = source_from_sl.data

    #2
    source_to_sl = ColumnDataSource(data = cds_upd)
    source_to.data = source_to_sl.data

    #3
    source_from_sl2 = ColumnDataSource(data = cds_upd)
    source_from2.data = source_from_sl2.data

    #4
    source_to_sl2 = ColumnDataSource(data = cds_upd)
    source_to2.data = source_to_sl2.data

    Time_Title.text = "Матрица: " + sl


select.on_change('value', update)


# In[144]:


def update_selection_to(idx_to):
    source_to.selected.update(indices=idx_to) 

def update_selection_from(idx2):
    source_from.selected.update(indices=idx2)    
    

def callback(attrname, old, new):

    but = button1.active
    
    idx = source_from.selected.indices

    print("Indices of selected circles from: ", idx)
    print("Length of selected circles from: ", len(idx))

    #таблица с выбранными индексами 
    df = pd.DataFrame(data=ds.data).iloc[idx]
    #сумма movements по выделенным индексам
    aa = df.groupby(['X_to','Y_to'])['size'].transform(sum)
    aat = df.groupby(['X_to','Y_to'])['text'].transform(sum)
    df['size_sum'] = aa
    df['text_sum'] = aat

    p_to = figure(x_range=(4157975.01546188769862056 , 4173827.06850233720615506), 
                  y_range=(7521739.63348639197647572,  7533621.55124872922897339),
                  x_axis_type="mercator", y_axis_type="mercator", tools=toolList_to)
    p_to.add_tile(CARTODBPOSITRON)
    
    
    lb_to = p_to.circle(x = 'X_to',
         y = 'Y_to',
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
    
    
    t = p_to.circle(x = 'X_to', y = 'Y_to', fill_color='red', fill_alpha = 1, 
                            line_color='red', line_alpha = 1, size=6 , source = source_to,
                           nonselection_fill_alpha = 1, nonselection_fill_color = 'red', 
                            nonselection_line_color='red', nonselection_line_alpha = 1)
    

    test = df.drop_duplicates(['X_to','Y_to'])
    
    test = test[test['text_sum'] > 30]
    
    if but == 0:
        
        stats.text = " "

        if not idx: #если пустое выделение

            layout1.children[1] = p_to #обновить график справа

        else: #если не пустое выделение

            for x in range(len(test)): #для каждого выделенного индекса рисуем site_to и его параметры
                
                new_data = dict()
                new_data_text = dict()
                
                new_data['x'] = [list(test['X_to'])[x]]
                new_data['y'] = [list(test['Y_to'])[x]]
                new_data['size'] = [list(test['size_sum'])[x]]

                new_data_text['x'] = [list(test['X_to'])[x]]
                new_data_text['y'] = [list(test['Y_to'])[x]]
                new_data_text['text'] = [list(test['text_sum'])[x]]

                t_to = p_to.circle(x = [], y = [], fill_color='orange', fill_alpha = 0.6, 
                                line_color='red', line_alpha = 0.8, size=[] )
                tds_to=t_to.data_source
                tds_to.data = new_data


                l = p_to.text(x = [], y = [], text_color='black', text =[], text_font_size='8pt',
                             text_font_style = 'bold')
                lds=l.data_source
                lds.data = new_data_text

                layout1.children[1] = p_to #обновить график справа
                                      
                
    else:
        
        layout1.children[1] = p_to #обновить график справа
        

source_from.selected.on_change('indices', callback) 


# In[145]:


def callback2(attrname, old, new):
    
    but = button1.active
    
    idx = source_to2.selected.indices
    
    print("Indices of selected circles: ", idx)
    print("Lenght of selected circles: ", len(idx))

    #таблица с выбранными индексами 
    df = pd.DataFrame(data=ds2.data).iloc[idx]
    
    #сумма movements по выделенным индексам
    aa = df.groupby(['X_from','Y_from'])['size'].transform(sum)
    aat = df.groupby(['X_from','Y_from'])['text'].transform(sum)
    df['size_sum'] = aa
    df['text_sum'] = aat

    p_from = figure(x_range=(4157975.01546188769862056 , 4173827.06850233720615506), 
                  y_range=(7521739.63348639197647572,  7533621.55124872922897339),
                  x_axis_type="mercator", y_axis_type="mercator", tools=toolList_to2)
    p_from.add_tile(CARTODBPOSITRON)
    
    lb_from = p_from.circle(x = 'X_from',
         y = 'Y_from',
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
    
    t2 = p_from.circle(x = 'X_from', y = 'Y_from', fill_color='red', fill_alpha = 1, 
                            line_color='red', line_alpha = 1, size=6 , source = source_from2,
                           nonselection_fill_alpha = 1, nonselection_fill_color = 'red', 
                            nonselection_line_color='red', nonselection_line_alpha = 1)

    test = df.drop_duplicates(['X_from','Y_from'])
    
    test = test[test['text_sum'] > 30]
      
    
    if but == 0:
        
        stats2.text = " "

        if not idx: #если пустое выделение

            layout2.children[1] = p_from #обновить график справа

        else: #если не пустое выделение

            for x in range(len(test)): #для каждого выделенного индекса рисуем site_to и его параметры
                
                new_data = dict()
                new_data_text = dict()

                new_data['x'] = [list(test['X_from'])[x]]
                new_data['y'] = [list(test['Y_from'])[x]]
                new_data['size'] = [list(test['size_sum'])[x]]

                new_data_text['x'] = [list(test['X_from'])[x]]
                new_data_text['y'] = [list(test['Y_from'])[x]]
                new_data_text['text'] = [list(test['text_sum'])[x]]

                
                t_from = p_from.circle(x = [], y = [], fill_color='orange', fill_alpha = 0.6, 
                                line_color='red', line_alpha = 0.8, size=[] )
                tds2=t_from.data_source
                tds2.data = new_data


                l2 = p_from.text(x = [], y = [], text_color='black', text =[], text_font_size='8pt',
                             text_font_style = 'bold')
                lds2=l2.data_source
                lds2.data = new_data_text

                layout2.children[1] = p_from #обновить график справа
                
    else:
        
        layout2.children[1] = p_from #обновить график справа


source_to2.selected.on_change('indices', callback2)


# In[146]:


def callback_to(attrname, old, new):
    
    but = button1.active

    idx2 = source_from.selected.indices
    idx_to = source_to.selected.indices
    
    update_selection_to(idx_to)
    update_selection_from(idx2)

    inters_idx = list(set(idx2) & set(idx_to))

    print("Length of selected circles to: ", idx2)
    print("Length of selected circles to: ", inters_idx)

    #таблица с выбранными индексами 
    dff = pd.DataFrame(data=tds.data).loc[inters_idx]
    print("Length of selected circles to: ", dff)
    
    
    p_to = figure(x_range=(4157975.01546188769862056 , 4173827.06850233720615506), 
                  y_range=(7521739.63348639197647572,  7533621.55124872922897339),
                  x_axis_type="mercator", y_axis_type="mercator", tools=toolList_to)
    p_to.add_tile(CARTODBPOSITRON)
    
    
    lb_to = p_to.circle(x = 'X_to',
     y = 'Y_to',
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
    
    
    t = p_to.circle(x = 'X_to', y = 'Y_to', fill_color='red', fill_alpha = 1, 
                            line_color='red', line_alpha = 1, size=6 , source = source_to,
                   nonselection_fill_alpha = 1, nonselection_fill_color = 'red', 
                                nonselection_line_color='red', nonselection_line_alpha = 1)
    

    
    
    test = dff.drop_duplicates(['X_to','Y_to'])

    #сумма movements по выделенным индексам
    aaa = dff['text'].sum()
    print("size to: ", aaa)

    #сайты из
    sitesfrom = dff['sitesfrom'].drop_duplicates()
    sitesto = dff['sitesto'].drop_duplicates()
    
    if but == 1:
        
        if not inters_idx:
            
            stats.text = "Никто не едет"
            layout1.children[1] = p_to #обновить график справа         
            
        else:
            
            for x in range(len(test)): #для каждого выделенного индекса рисуем site_to и его параметры
                
                new_data = dict()
                new_data['x'] = [list(test['X_to'])[x]]
                new_data['y'] = [list(test['Y_to'])[x]]

                t_to = p_to.circle(x = [], y = [], fill_color='red', fill_alpha = 0.5, 
                                line_color='red', line_alpha = 0.8, size=15)
                tds_to=t_to.data_source
                tds_to.data = new_data

                layout1.children[1] = p_to #обновить график справа

                stats.text = "Из сайтов " + str(list(sitesfrom)) + " в сайты " + str(list(sitesto)) + " едет " + str(aaa) + " человек(а) в час"

    

source_to.selected.on_change('indices', callback_to)


# In[147]:


def update_selection_from2(idx2):
    source_to2.selected.update(indices=idx2) 

def update_selection_to2(idx_to):
    source_from2.selected.update(indices=idx_to)

def callback_to2(attrname, old, new):
    
    but = button1.active

    idx2 = source_to2.selected.indices
    idx_to = source_from2.selected.indices
    
    update_selection_to2(idx_to)
    update_selection_from2(idx2)

    inters_idx = list(set(idx2) & set(idx_to))

    print("Length of selected circles to: ", idx2)
    print("Length of selected circles to: ", inters_idx)

    #таблица с выбранными индексами 
    dff = pd.DataFrame(data=tds2.data).loc[inters_idx]
    print("Length of selected circles to: ", dff)
    
    p_from = figure(x_range=(4157975.01546188769862056 , 4173827.06850233720615506), 
                  y_range=(7521739.63348639197647572,  7533621.55124872922897339),
                  x_axis_type="mercator", y_axis_type="mercator", tools=toolList_to2)
    p_from.add_tile(CARTODBPOSITRON)
    
    lb_from = p_from.circle(x = 'X_from',
         y = 'Y_from',
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
    
    t2 = p_from.circle(x = 'X_from', y = 'Y_from', fill_color='red', fill_alpha = 1, 
                            line_color='red', line_alpha = 1, size=6 , source = source_from2,
                           nonselection_fill_alpha = 1, nonselection_fill_color = 'red', 
                            nonselection_line_color='red', nonselection_line_alpha = 1)

    test = dff.drop_duplicates(['X_from','Y_from'])
    
    #сумма movements по выделенным индексам
    aaa = dff['text'].sum()
    print("size to: ", aaa)

    #сайты из
    sitesfrom = dff['sitesfrom'].drop_duplicates()
    sitesto = dff['sitesto'].drop_duplicates()
    
    if but == 1:
        
        if not inters_idx:
            
            stats2.text = "Никто не едет"
            layout2.children[1] = p_from #обновить график справа         
            
        else:
            
            for x in range(len(test)): #для каждого выделенного индекса рисуем site_to и его параметры
                
                new_data = dict()
                new_data['x'] = [list(test['X_from'])[x]]
                new_data['y'] = [list(test['Y_from'])[x]]

                t_from = p_from.circle(x = [], y = [], fill_color='red', fill_alpha = 0.5, 
                                line_color='red', line_alpha = 0.8, size=15)
                tds_from=t_from.data_source
                tds_from.data = new_data

                layout2.children[1] = p_from #обновить график справа
    

                stats2.text = "В сайты " + str(list(sitesto)) + " из сайтов " + str(list(sitesfrom)) + " едет " + str(aaa) + " человек(а) в час"


source_from2.selected.on_change('indices', callback_to2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[148]:


layout1 = layout.row(p,p_to,stats,select)
layout2 = layout.row(p2, p_from,stats2,button1)

# Create Tabs
box = layout.column(layout1, layout2)


curdoc().add_root(box)


# In[149]:


# apps = {'/': Application(FunctionHandler(make_document))}

# server = server(apps, port=5001)
# server.start()


# if __name__ == '__main__':
#     print('Opening Bokeh application on http://localhost:5006/')

# server.io_loop.add_callback(server.show, "/")


# In[ ]:





# In[ ]:




