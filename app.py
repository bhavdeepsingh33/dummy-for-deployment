# -*- coding: utf-8 -*-
"""
Created on Tue May 19 02:42:29 2020

@author: bhavdeep singh
"""

from flask import Flask, render_template, stream_with_context, request, Response, jsonify, make_response

app = Flask(__name__)



@app.route('/')
def index():
    return render_template("dashboard.html")

@app.route('/info')
def info():
    return render_template("info.html")

################################################################################################

if __name__ == '__main__':
    app.run()











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



