# -*- coding: utf-8 -*-
"""
Created on Tue May 19 02:42:29 2020

@author: bhavdeep singh
"""

from flask import Flask, render_template, stream_with_context, request, Response, jsonify, make_response
import pickle 
import os 
import pandas as pd
import numpy as np
import json



import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
from collections import deque


app = Flask(__name__)


pwd = os.getcwd()

df = pd.read_csv(pwd +"/Combined.csv")
#print(df.head())
#print(df.describe)
#print(df.dtypes)
#for i in ['Mode', 'Sample Number', 'Seconds', 'Minutes', 'Hours', 'Date', 'Month']:
#    df[i] = pd.to_numeric(df[i], downcast='integer')
#print(df.iloc[0]['pCut::Motor_Torque'])

useful_features = ['pCut::Motor_Torque',
                   'pCut::CTRL_Position_controller::Lag_error',
                   'pCut::CTRL_Position_controller::Actual_position',
                   'pCut::CTRL_Position_controller::Actual_speed',
                   'pSvolFilm::CTRL_Position_controller::Actual_position',
                   'pSvolFilm::CTRL_Position_controller::Actual_speed',
                   'pSvolFilm::CTRL_Position_controller::Lag_error', 'pSpintor::VAX_speed',
                   'Month']

def value_gen(data):
    for i in range(0,len(data),125):
        yield data.iloc[i][useful_features].round(2)
        
value = value_gen(df)
#x = next(value)

#print(df)

model = pickle.load(open('EllipticEnvelope_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


total_time = 0.0

data_row = 0

def generate_next_data_values():
    global data_row
    global value
    data_row = next(value)
    return data_row


@app.route('/_stuff', methods=['GET'])
def stuff():
    try:
        #a = next(value)
        a = generate_next_data_values()
        scaled_df = scaler.transform(np.array(a).reshape(1, -1))
        scaled_df = pd.DataFrame(scaled_df, columns = useful_features)
        anomaly_predict = model.predict(scaled_df)
        anomaly_predict =  pd.Series(anomaly_predict).replace([-1,1],[1,0])
        #print(anomaly_predict[0])
        output = anomaly_predict[0]
        global total_time
        total_time += 0.5
        #print(total_time)
        #print(a)
        #col = a.index[0]
        x = {
                'col1' : a[0],
                'col2' : a[1],
                'col3' : a[2],
                'col4' : a[3],
                'col5' : a[4],
                'col6' : a[5],
                'col7' : a[6],
                'col8' : a[7],
                'col9' : a[8],
                'pred' : int(output),
                'time' : total_time    
            }
        response = make_response(json.dumps(x))
        response.content_type = 'application/json'
        return response
        """
        return jsonify(col1 = a[0],
                       col2 = a[1],
                       col3 = a[2],
                       col4 = a[3],
                       col5 = a[4],
                       col6 = a[5],
                       col7 = a[6],
                       col8 = a[7],
                       col9 = a[8],
                       pred = int(output),
                       time = total_time
                       )
        """
    except Exception as e:
        print("Exception is :",e)
        return jsonify(result='data not found')


@app.route('/')
def index():
    return render_template("dashboard.html")

@app.route('/info')
def info():
    return render_template("info.html")

################################################################################################

X = [deque(maxlen=50) for _ in range(8)]
Y = [deque(maxlen=50) for _ in range(8)]

for i in range(8):
    X[i].append(1)
    Y[i].append(1)

dash_app = dash.Dash(__name__, server=app, routes_pathname_prefix='/dash/')

dash_app.layout = html.Div(
    [
        dcc.Graph(id='pCut::Motor_Torque', animate=True),
        dcc.Interval(
            id='col1-update',
            interval=1000,
            n_intervals = 0
        ),
        dcc.Graph(id='pCut::CTRL_Position_controller::Lag_error', animate=True),
        dcc.Interval(
            id='col2-update',
            interval=1000,
            n_intervals = 0
        ),
        dcc.Graph(id='pCut::CTRL_Position_controller::Actual_position', animate=True),
        dcc.Interval(
            id='col3-update',
            interval=1000,
            n_intervals = 0
        ),
        dcc.Graph(id='pCut::CTRL_Position_controller::Actual_speed', animate=True),
        dcc.Interval(
            id='col4-update',
            interval=1000,
            n_intervals = 0
        ),
        dcc.Graph(id='pSvolFilm::CTRL_Position_controller::Actual_position', animate=True),
        dcc.Interval(
            id='col5-update',
            interval=1000,
            n_intervals = 0
        ),
        dcc.Graph(id='pSvolFilm::CTRL_Position_controller::Actual_speed', animate=True),
        dcc.Interval(
            id='col6-update',
            interval=1000,
            n_intervals = 0
        ),
        dcc.Graph(id='pSvolFilm::CTRL_Position_controller::Lag_error', animate=True),
        dcc.Interval(
            id='col7-update',
            interval=1000,
            n_intervals = 0
        ),
        dcc.Graph(id='pSpintor::VAX_speed', animate=True),
        dcc.Interval(
            id='col8-update',
            interval=1000,
            n_intervals = 0
        )
        
        
    ]
)


@dash_app.callback(Output('pCut::Motor_Torque', 'figure'),
        [Input('col1-update', 'n_intervals')])
def col1_graph_scatter(n):
    global X
    global Y
    data_x = X[0]
    data_x.append(data_x[-1]+1)
    data_y = Y[0]

    x = data_row['pCut::Motor_Torque']
    data_y.append(x)
    data = plotly.graph_objs.Scatter(
            x=list(data_x),
            y=list(data_y),
            name='Scatter',
            mode= 'lines+markers',
            
            )
    return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(data_x),max(data_x)]),
                                                yaxis=dict(range=[min(data_y),max(data_y)]),
                                                xaxis_title="Time in seconds",
                                                yaxis_title="pCut::Motor_Torque",
                                                title="pCut::Motor_Torque")}


@dash_app.callback(Output('pCut::CTRL_Position_controller::Lag_error', 'figure'),
        [Input('col2-update', 'n_intervals')])
def col2_graph_scatter(n):
    global X
    global Y
    data_x = X[1]
    data_x.append(data_x[-1]+1)
    data_y = Y[1]
    

    x = data_row['pCut::CTRL_Position_controller::Lag_error']
    
    data_y.append(x)
    data = plotly.graph_objs.Scatter(
            x=list(data_x),
            y=list(data_y),
            name='Scatter',
            mode= 'lines+markers',
            
            )
    return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(data_x),max(data_x)]),
                                                yaxis=dict(range=[min(data_y),max(data_y)]),
                                                xaxis_title="Time in seconds",
                                                yaxis_title="pCut::CTRL_Position_controller::Lag_error",
                                                title="pCut::CTRL_Position_controller::Lag_error")}

@dash_app.callback(Output('pCut::CTRL_Position_controller::Actual_position', 'figure'),
        [Input('col3-update', 'n_intervals')])
def col3_graph_scatter(n):
    global X
    global Y
    data_x = X[2]
    data_x.append(data_x[-1]+1)
    data_y = Y[2]
    col = 'pCut::CTRL_Position_controller::Actual_position'

    x = data_row[col]
    
    data_y.append(x)
    data = plotly.graph_objs.Scatter(
            x=list(data_x),
            y=list(data_y),
            name='Scatter',
            mode= 'lines+markers',
            
            )
    return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(data_x),max(data_x)]),
                                                yaxis=dict(range=[min(data_y),max(data_y)]),
                                                xaxis_title="Time in seconds",
                                                yaxis_title=col,
                                                title=col)}

@dash_app.callback(Output('pCut::CTRL_Position_controller::Actual_speed', 'figure'),
        [Input('col4-update', 'n_intervals')])
def col4_graph_scatter(n):
    global X
    global Y
    data_x = X[3]
    data_x.append(data_x[-1]+1)
    data_y = Y[3]
    col = 'pCut::CTRL_Position_controller::Actual_speed'

    x = data_row[col]
    
    data_y.append(x)
    data = plotly.graph_objs.Scatter(
            x=list(data_x),
            y=list(data_y),
            name='Scatter',
            mode= 'lines+markers',
            
            )
    return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(data_x),max(data_x)]),
                                                yaxis=dict(range=[min(data_y),max(data_y)]),
                                                xaxis_title="Time in seconds",
                                                yaxis_title=col,
                                                title=col)}

@dash_app.callback(Output('pSvolFilm::CTRL_Position_controller::Actual_position', 'figure'),
        [Input('col5-update', 'n_intervals')])
def col5_graph_scatter(n):
    global X
    global Y
    data_x = X[4]
    data_x.append(data_x[-1]+1)
    data_y = Y[4]
    col = 'pSvolFilm::CTRL_Position_controller::Actual_position'

    x = data_row[col]
    
    data_y.append(x)
    data = plotly.graph_objs.Scatter(
            x=list(data_x),
            y=list(data_y),
            name='Scatter',
            mode= 'lines+markers',
            
            )
    return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(data_x),max(data_x)]),
                                                yaxis=dict(range=[min(data_y),max(data_y)]),
                                                xaxis_title="Time in seconds",
                                                yaxis_title=col,
                                                title=col)}

@dash_app.callback(Output('pSvolFilm::CTRL_Position_controller::Actual_speed', 'figure'),
        [Input('col6-update', 'n_intervals')])
def col6_graph_scatter(n):
    global X
    global Y
    data_x = X[5]
    data_x.append(data_x[-1]+1)
    data_y = Y[5]
    col = 'pSvolFilm::CTRL_Position_controller::Actual_speed'
    x = data_row[col]
    
    data_y.append(x)
    data = plotly.graph_objs.Scatter(
            x=list(data_x),
            y=list(data_y),
            name='Scatter',
            mode= 'lines+markers',
            
            )
    return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(data_x),max(data_x)]),
                                                yaxis=dict(range=[min(data_y),max(data_y)]),
                                                xaxis_title="Time in seconds",
                                                yaxis_title=col,
                                                title=col)}

@dash_app.callback(Output('pSvolFilm::CTRL_Position_controller::Lag_error', 'figure'),
        [Input('col7-update', 'n_intervals')])
def col7_graph_scatter(n):
    global X
    global Y
    data_x = X[6]
    data_x.append(data_x[-1]+1)
    data_y = Y[6]
    col = 'pSvolFilm::CTRL_Position_controller::Lag_error'
    x = data_row[col]
    
    data_y.append(x)
    data = plotly.graph_objs.Scatter(
            x=list(data_x),
            y=list(data_y),
            name='Scatter',
            mode= 'lines+markers',
            
            )
    return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(data_x),max(data_x)]),
                                                yaxis=dict(range=[min(data_y),max(data_y)]),
                                                xaxis_title="Time in seconds",
                                                yaxis_title=col,
                                                title=col)}

@dash_app.callback(Output('pSpintor::VAX_speed', 'figure'),
        [Input('col8-update', 'n_intervals')])
def col8_graph_scatter(n):
    global X
    global Y
    data_x = X[7]
    data_x.append(data_x[-1]+1)
    data_y = Y[7]
    col = 'pSpintor::VAX_speed'
    x = data_row[col]
    
    data_y.append(x)
    data = plotly.graph_objs.Scatter(
            x=list(data_x),
            y=list(data_y),
            name='Scatter',
            mode= 'lines+markers',
            
            )
    return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(data_x),max(data_x)]),
                                                yaxis=dict(range=[min(data_y),max(data_y)]),
                                                xaxis_title="Time in seconds",
                                                yaxis_title=col,
                                                title=col)}





if __name__ == '__main__':
    dash_app.run_server()



# END OF APPLICATION
##################################################################################################
"""
        dcc.Graph(id='pCut::CTRL_Position_controller::Actual_position', animate=True),
        dcc.Interval(
            id='col3-update',
            interval=1000,
            n_intervals = 0
        ),
        dcc.Graph(id='pCut::CTRL_Position_controller::Actual_speed', animate=True),
        dcc.Interval(
            id='col4-update',
            interval=1000,
            n_intervals = 0
        ),
        dcc.Graph(id='pSvolFilm::CTRL_Position_controller::Actual_position', animate=True),
        dcc.Interval(
            id='col5-update',
            interval=1000,
            n_intervals = 0
        ),
        dcc.Graph(id='pSvolFilm::CTRL_Position_controller::Actual_speed', animate=True),
        dcc.Interval(
            id='col6-update',
            interval=1000,
            n_intervals = 0
        ),
        dcc.Graph(id='pSvolFilm::CTRL_Position_controller::Lag_error', animate=True),
        dcc.Interval(
            id='col7-update',
            interval=1000,
            n_intervals = 0
        ),
        dcc.Graph(id='pSpintor::VAX_speed', animate=True),
        dcc.Interval(
            id='col8-update',
            interval=1000,
            n_intervals = 0
        )

"""



