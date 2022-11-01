import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

def dash_app(df):

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    fig = px.line(df)

    app.layout = html.Div(children=[
        html.H1(children='S&P500 Forecast'),

        html.Div(children='''
            Sample
        '''),

        dcc.Graph(
            id='example-graph',
            figure=fig

    app.run_server(debug=True)
