#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import networkx as nx
import plotly.graph_objs as go

import pandas as pd
from colour import Color
from textwrap import dedent as d
import json
import decimal

from Corpus import Corpus

ctx = decimal.Context()
ctx.prec = 20

def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

def color_to_str(c):
    return "("+", ".join(map(float_to_str, c))+")"

# import the css template, and pass the css template into dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Transaction Network"

ACCOUNT="data;"
NSUBMIT=0
FIG=None

nb_docs = 3
corpus = Corpus("data")
corpus.download_collection(nb_docs)
A = corpus.get_adjacency_matrix()

##############################################################################################################################################################
def network_graph(WordsToSearch):
    WordsToSearch = set(WordsToSearch.split(";"))
    if "" in WordsToSearch:
        WordsToSearch.remove("")
    
    #edge1 = pd.read_csv('edge1.csv')
    #node1 = pd.read_csv('node1.csv')
    
    #filtre par rapport à l'input de filtre
    words = WordsToSearch.copy()
    if len(words) > 0:
        words.update(A.loc[words].loc[:,(A.loc[words]!=0).any(axis=0)].columns)
    else:
        words.update(A.columns)
    Ap = A.loc[words]
    
    #filtre sur les 30 mots les plus fréquents
    # words = corpus.most_frequent_words(30)
    
    #filtre sur les mots plus fréquents que la moyenne
    moyennes = Ap.mean()
    moyenne = moyennes.mean()
    words = set(moyennes[moyennes>=moyenne].index)
    
    edge1 = Ap.loc[words,words].stack()
    edge1 = edge1.reset_index()
    edge1 = edge1[edge1[0] != 0]
    edge1["from"] = edge1.apply(lambda x : min(x[["level_0","level_1"]]), axis=1)
    edge1["to"] = edge1.apply(lambda x : max(x[["level_0","level_1"]]), axis=1)
    edge1["qt"] = edge1[0]
    edge1 = edge1.drop(columns=["level_0","level_1",0])
    edge1 = edge1.drop_duplicates().reset_index(drop=True)
    node1 = pd.DataFrame(A.index)
    node1.columns = ("name",)

    # filter the record by datetime, to enable interactive control through the input box
    # edge1['Datetime'] = "" # add empty Datetime column to edge1 dataframe
    accountSet=set() # contain unique account
    for index in range(0,len(edge1)):
        # edge1['Datetime'][index] = datetime.strptime(edge1['Date'][index], '%d/%m/%Y')
        # if edge1['Datetime'][index].year<yearRange[0] or edge1['Datetime'][index].year>yearRange[1]:
        #     edge1.drop(axis=0, index=index, inplace=True)
        #     continue
        accountSet.add(edge1['from'][index])
        accountSet.add(edge1['to'][index])

    # to define the centric point of the networkx layout
    shells=[]
    shell1=[]
    shell1.extend(WordsToSearch)
    shells.append(shell1)
    shell2=[]
    for ele in accountSet:
        if ele!=WordsToSearch:
            shell2.append(ele)
    shells.append(shell2)


    G = nx.from_pandas_edgelist(edge1, 'from', 'to', ['from', 'to', 'qt'], create_using=nx.MultiDiGraph())
    # nx.set_node_attributes(G, node1.set_index('Account')['CustomerName'].to_dict(), 'CustomerName')
    # nx.set_node_attributes(G, node1.set_index('Account')['Type'].to_dict(), 'Type')
    # pos = nx.layout.spring_layout(G)
    # pos = nx.layout.circular_layout(G)
    # nx.layout.shell_layout only works for more than 3 nodes
    if len(shell2)>1:
        pos = nx.drawing.layout.shell_layout(G, shells)
    else:
        pos = nx.drawing.layout.spring_layout(G)
    for node in G.nodes:
        G.nodes[node]['pos'] = list(pos[node])


    # if len(shell2)==0:
    #     traceRecode = []  # contains edge_trace, node_trace, middle_node_trace

    #     node_trace = go.Scatter(x=tuple([1]), y=tuple([1]), text=tuple([str(AccountToSearch)]), textposition="bottom center",
    #                             mode='markers+text',
    #                             marker={'size': 50, 'color': 'LightSkyBlue'})
    #     traceRecode.append(node_trace)

    #     node_trace1 = go.Scatter(x=tuple([1]), y=tuple([1]),
    #                             mode='markers',
    #                             marker={'size': 50, 'color': 'LightSkyBlue'},
    #                             opacity=0)
    #     traceRecode.append(node_trace1)

    #     figure = {
    #         "data": traceRecode,
    #         "layout": go.Layout(title='Interactive Transaction Visualization', showlegend=False,
    #                             margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
    #                             xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
    #                             yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
    #                             height=600
    #                             )}
    #     return figure


    traceRecode = []  # contains edge_trace, node_trace, middle_node_trace
    ############################################################################################################################################################
    colors = list(Color('lightyellow').range_to(Color('darkred'), max(edge1['qt'])-min(edge1['qt'])+1))
    colors = ['rgb' + color_to_str(x.rgb) for x in colors]

    # index = 0
    for edge in G.edges:
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        #weight = float(G.edges[edge]['qt']) / max(edge1['qt']) * 10
        trace = go.Scatter(x=tuple([x0, x1, None]), y=tuple([y0, y1, None]),
                            mode='lines',
                            line={'width': 1},
                            marker=dict(color=colors[G.edges[edge]['qt']-min(edge1['qt'])]),
                            line_shape='spline',
                            opacity=1)
        traceRecode.append(trace)
        # index = index + 1
    ###############################################################################################################################################################
    node_trace = go.Scatter(x=[], y=[], hovertext=[], text=[], mode='markers+text', textposition="bottom center",
                            #textfont=dict(color="Orange"),
                            hoverinfo="text",
                            marker={'size': 10, 'color': 'LightSkyBlue'})

    index = 0
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        # hovertext = "name: " + node1['name'][index]
        text = node1['name'][index]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        # node_trace['hovertext'] += tuple([hovertext])
        node_trace['text'] += tuple([text])
        index = index + 1

    traceRecode.append(node_trace)
    ################################################################################################################################################################
    # middle_hover_trace = go.Scatter(x=[], y=[], hovertext=[], mode='markers', hoverinfo="text",
    #                                 marker={'size': 20, 'color': 'LightSkyBlue'},
    #                                 opacity=0)

    # index = 0
    # for edge in G.edges:
    #     x0, y0 = G.nodes[edge[0]]['pos']
    #     x1, y1 = G.nodes[edge[1]]['pos']
    #     hovertext = "From: " + str(G.edges[edge]['from']) + "<br>" + "To: " + str(
    #         G.edges[edge]['to']) + "<br>" + "qt: " + str(G.edges[edge]['qt'])
    #     middle_hover_trace['x'] += tuple([(x0 + x1) / 2])
    #     middle_hover_trace['y'] += tuple([(y0 + y1) / 2])
    #     middle_hover_trace['hovertext'] += tuple([hovertext])
    #     index = index + 1

    # traceRecode.append(middle_hover_trace)
    #################################################################################################################################################################
    figure = {
        "data": traceRecode,
        "layout": go.Layout(title='Co-occurrences des mots '+str(WordsToSearch)+"\nNombre de noeuds : "+str(len(words)), showlegend=False, hovermode='closest',
                            margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            height=600,
                            clickmode='event+select',
                            # annotations=[
                            #     dict(
                            #         # ax=(G.nodes[edge[0]]['pos'][0] + G.nodes[edge[1]]['pos'][0]) / 2,
                            #         # ay=(G.nodes[edge[0]]['pos'][1] + G.nodes[edge[1]]['pos'][1]) / 2, axref='x', ayref='y',
                            #         # x=(G.nodes[edge[1]]['pos'][0] * 3 + G.nodes[edge[0]]['pos'][0]) / 4,
                            #         # y=(G.nodes[edge[1]]['pos'][1] * 3 + G.nodes[edge[0]]['pos'][1]) / 4, xref='x', yref='y',
                            #         ax = G.nodes[edge[0]]['pos'][0],
                            #         ay = G.nodes[edge[0]]['pos'][1], axref='x', ayref='y',
                            #         x = G.nodes[edge[1]]['pos'][0],
                            #         y = G.nodes[edge[1]]['pos'][1], xref='x', yref='y',
                            #         showarrow=True,
                            #         arrowhead=0,
                            #         # arrowsize=4,
                            #         # arrowwidth=1,
                            #         opacity=1
                            #     ) for edge in G.edges]
                            )}
    return figure
######################################################################################################################################################################
# styles: for right side hover/click component
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = html.Div([
    #########################Title
    html.Div([html.H1("Co-occurrences des mots du Corpus \"Data\"")],
             className="row",
             style={'textAlign': "center"}),
    #############################################################################################define the row
    html.Div(
        className="row",
        children=[
            ##############################################left side two input components
            html.Div(
                className="two columns",
                children=[
                    # dcc.Markdown(d("""
                    #         **Time Range To Visualize**
                    #         Slide the bar to define year range.
                    #         """)),
                    # html.Div(
                    #     className="twelve columns",
                    #     children=[
                    #         dcc.RangeSlider(
                    #             id='my-range-slider',
                    #             min=2010,
                    #             max=2019,
                    #             step=1,
                    #             value=[2010, 2019],
                    #             marks={
                    #                 2010: {'label': '2010'},
                    #                 2011: {'label': '2011'},
                    #                 2012: {'label': '2012'},
                    #                 2013: {'label': '2013'},
                    #                 2014: {'label': '2014'},
                    #                 2015: {'label': '2015'},
                    #                 2016: {'label': '2016'},
                    #                 2017: {'label': '2017'},
                    #                 2018: {'label': '2018'},
                    #                 2019: {'label': '2019'}
                    #             }
                    #         ),
                    #         html.Br(),
                    #         html.Div(id='output-container-range-slider')
                    #     ],
                    #     style={'height': '300px'}
                    # ),
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.Markdown(d("""
                            **Words To Search**
                            Input the words to visualize.
                            """)),
                            dcc.Input(id="words", type="text", placeholder="Words", value="data;", n_submit=1),
                            html.Div(id="output")
                        ],
                        style={'height': '300px'}
                    )
                ]
            ),

            ############################################middle graph component
            html.Div(
                className="eight columns",
                children=[dcc.Graph(id="my-graph",
                                    figure=network_graph(ACCOUNT))],
            ),

            #########################################right side two output component
            html.Div(
                className="two columns",
                children=[
                    html.Div(
                        className='twelve columns',
                        children=[
                            dcc.Markdown(d("""
                            **Hover Data**
                            Mouse over values in the graph.
                            """)),
                            html.Pre(id='hover-data', style=styles['pre'])
                        ],
                        style={'height': '400px'}),

                    html.Div(
                        className='twelve columns',
                        children=[
                            dcc.Markdown(d("""
                            **Click Data**
                            Click on points in the graph.
                            """)),
                            html.Pre(id='click-data', style=styles['pre'])
                        ],
                        style={'height': '400px'})
                ]
            )
        ]
    )
])

###################################callback for left side components
@app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    [dash.dependencies.Input('words', 'value'),
     dash.dependencies.Input('words', 'n_submit')
     ])
def update_output(words, n):
    global ACCOUNT
    global NSUBMIT
    global FIG
    if FIG == None or n > NSUBMIT:
        if n > NSUBMIT:
            NSUBMIT = n
            ACCOUNT = words
        FIG = network_graph(ACCOUNT)
        print("finished")
    return FIG
    # to update the global variable of YEAR and ACCOUNT
################################callback for right side components
@app.callback(
    dash.dependencies.Output('hover-data', 'children'),
    [dash.dependencies.Input('my-graph', 'hoverData')])
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)


@app.callback(
    dash.dependencies.Output('click-data', 'children'),
    [dash.dependencies.Input('my-graph', 'clickData')])
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)



if __name__ == '__main__':
    app.run_server()