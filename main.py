#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import networkx as nx
import plotly.graph_objs as go

import pandas as pd
import numpy as np
from colour import Color
from textwrap import dedent as d
import json
import decimal
from math import ceil, log

from Corpus import Corpus

from sklearn.neighbors import KernelDensity

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


class App:
    def __init__(self, theme="data", nb_docs=20):
        self.NSUBMIT_words = 0
        self.NSUBMIT_corpus = 0
        self.FIG = None
        
        self.setup_corpus(theme, nb_docs)
        self.setup_dash()
        #self.setup_dash("Co-occurrences des mots du Corpus \""+theme+"\"")
        self.callbacks()
        self.launch()

    def setup_corpus(self, theme, nb_docs):
        self.WORDS = theme+";"
        self.DEGCEN = {}
        self.CLOCEN = {}
        self.BETCEN = {}
        self.THEME = theme
        self.NB_DOCS = nb_docs
        self.corpus = Corpus(theme)
        self.corpus.download_collection(nb_docs, keyword=theme)
        self.A = self.corpus.get_adjacency_matrix()

    def setup_dash(self):
        # create the dash app,
        # import the css template,
        # and pass the css template into dash
        self.app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
        self.app.title = "Projet Prog"
        ######################################################################################################################################################################
        # styles: for right side hover/click component
        styles = {
            'pre': {
                'border': 'thin lightgrey solid',
                'overflowX': 'scroll'
            }
        }
        
        self.FIG = self.network_graph()
        
        self.app.layout = html.Div([
            #########################Title
            html.Div([html.H1("Co-occurrences of words in the corpus - Number of words : "+str(self.nb_words), id="title")],
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
                            html.Div(
                                className="twelve columns",
                                children=[
                                    dcc.Markdown(d("**Corpus theme**")),
                                    dcc.Input(id="theme", type="text", placeholder="Theme", value=self.THEME),
                                    html.Div(id="output1"),
                                    
                                    dcc.Markdown(d("**Number of documents to download**")),
                                    dcc.Input(id="nbdocs", type="number", value=self.NB_DOCS, min=1),
                                    html.Div(id="output2"),
                                    
                                    html.Button('Reset the Corpus', id='reset')
                                ],
                                style={'height': '300px'}
                            ),
                            html.Div(
                                className="twelve columns",
                                children=[
                                    dcc.Markdown(d("**Words To Search**")),
                                    dcc.Input(id="words", type="text", placeholder="Words", value=self.WORDS, n_submit=1),
                                    html.Div(id="output3")
                                ],
                                style={'height': '200px'}
                            )
                        ]
                    ),
        
                    ############################################middle graph component
                    html.Div(
                        className="eight columns",
                        children=[dcc.Graph(id="my-graph",
                                            figure=self.FIG)]
                    ),
        
                    #########################################right side two output component
                    html.Div(
                        className="two columns",
                        children=[
                            html.Div(
                                className='twelve columns',
                                children=[
                                    dcc.Markdown(d("**Words metrics**")),
                                    html.Pre(id='hover-data', style=styles['pre'])
                                ],
                                style={'height': '400px'}),
        
                            # html.Div(
                            #     className='twelve columns',
                            #     children=[
                            #         dcc.Markdown(d("**Click Data**")),
                            #         html.Pre(id='click-data', style=styles['pre'])
                            #     ],
                            #     style={'height': '400px'})
                        ]
                    )
                ]
            )
        ])
    
    def launch(self):
        # launch dash
        self.app.run_server()

    def network_graph(self):
        WordsToSearch = list(set(self.WORDS.split(";")))
        if "" in WordsToSearch:
            WordsToSearch.remove("")
        
        #edge1 = pd.read_csv('edge1.csv')
        #node1 = pd.read_csv('node1.csv')
        
        #filtre par rapport à l'input de filtre
        words = WordsToSearch.copy()
        if len(words) > 0:
            words.extend(self.A.loc[words].loc[:,(self.A.loc[words]!=0).any(axis=0)].columns)
        else:
            words.extend(self.A.columns)
        words = list(set(words))
        Ap = self.A.loc[words,words]
        
        #filtre sur les 30 mots les plus fréquents
        # words = corpus.most_frequent_words(30)
        
        #filtre sur les stopwords
        to_delete = []
        for word in words:
            if self.corpus.is_stop_words(word):
                to_delete.append(word)
        for word in to_delete:
            words.remove(word)
            # Ap = Ap.drop(word, axis=0)
            # Ap = Ap.drop(word, axis=1)
        Ap = Ap.loc[words,words]
        
        #filtre sur les mots plus fréquents que la moyenne
        freq = self.corpus.frequencies
        freq = freq.loc[freq.index.isin(Ap.index)]
        term_freq_mean = freq["term frequency"].mean()
        words = list(freq[freq["term frequency"] >= term_freq_mean].index)
        Ap = Ap.loc[words,words]
        
        #filtre sur les mots plus co-occurrents que la moyenne
        moyennes = Ap.mean()
        moyenne = moyennes.mean()
        words = list(moyennes[moyennes>=moyenne].index)
        Ap = Ap.loc[words,words]
        
        #test de calcul des collocats
        # for wordx in Ap.columns:
        #     for wordy in Ap.index:
        #         print(self.pmi_func(Ap, wordx, wordy))
        
        self.nb_words = len(words)
        
        #calcul du graphe
        edge1 = Ap.stack()
        edge1 = edge1.reset_index()
        edge1 = edge1[edge1[0] != 0]
        edge1["from"] = edge1.apply(lambda x : min(x[["level_0","level_1"]]), axis=1)
        edge1["to"] = edge1.apply(lambda x : max(x[["level_0","level_1"]]), axis=1)
        edge1["qt"] = edge1[0]
        edge1 = edge1.drop(columns=["level_0","level_1",0])
        edge1 = edge1.drop_duplicates().reset_index(drop=True)
        node1 = pd.DataFrame(Ap.index)
        node1.columns = ("name",)
    
        self.G = nx.from_pandas_edgelist(edge1, 'from', 'to', ['from', 'to', 'qt'], create_using=nx.Graph())
        
        base = 3
        nwords = len(words)
        initinA = list(Ap.index[Ap.index.isin(WordsToSearch)])
        ninitinA = len(initinA)
        n = nwords-ninitinA
        if ninitinA > 0:
            nb_shells = ceil(log(n,base)-log(ninitinA,base))
        else:
            nb_shells = ceil(log(n,base))
        s = (base-1)*n/(base**nb_shells-1)
        if ninitinA > 0:
            shells = [WordsToSearch]
            flat_shells = initinA.copy()
        else:
            shells = []
            flat_shells = []
        for i in range(int(nb_shells+0.5)-1):
            x = ceil(s*base**i)
            if len(flat_shells) > 0:
                shell = list(Ap[Ap.loc[flat_shells] != 0].loc[:,~Ap.index.isin(flat_shells)].max().sort_values(ascending=False).head(x).index)
            else:
                shell = list(Ap.max().sort_values(ascending=False).head(x).index)
            flat_shells.extend(shell)
            shells.append(shell)
        shell = list(Ap.index[~Ap.index.isin(flat_shells)])
        shells.append(shell)
        
        pos = nx.drawing.layout.shell_layout(self.G, shells, dim=2)
        # pos = nx.drawing.layout.random_layout(self.G, dim=2, center=None)
        # pos = nx.drawing.layout.spring_layout(self.G)
        # pos = nx.drawing.layout.kamada_kawai_layout(self.G)
        # pos = nx.drawing.layout.spectral_layout(self.G)
        # pos = nx.drawing.layout.spiral_layout(self.G)
        
        for node in self.G.nodes:
            self.G.nodes[node]['pos'] = list(pos[node])
        
        
    
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
        alphas = [((i-min(edge1['qt']))/(max(edge1['qt'])-min(edge1['qt']))/10)+0.05 for i in range(min(edge1['qt']),max(edge1['qt'])+1)]
    
        # index = 0
        for edge in self.G.edges:
            x0, y0 = self.G.nodes[edge[0]]['pos']
            x1, y1 = self.G.nodes[edge[1]]['pos']
            #weight = float(self.G.edges[edge]['qt']) / max(edge1['qt']) * 10
            trace = go.Scatter(x=tuple([x0, x1, None]), y=tuple([y0, y1, None]),
                                mode='lines',
                                line={'width': 1},
                                marker=dict(color=colors[self.G.edges[edge]['qt']-min(edge1['qt'])]),
                                line_shape='spline',
                                opacity=alphas[self.G.edges[edge]['qt']-min(edge1['qt'])])
            traceRecode.append(trace)
            # index = index + 1
        ###############################################################################################################################################################
        sizes = Ap.sum().values*5/len(words)+5
        node_trace = go.Scatter(x=[], y=[], hovertext=[], text=[], mode='markers+text', textposition="bottom center",
                                #textfont=dict(color="Orange"),
                                hoverinfo="text",
                                marker={'size': sizes, 'color': 'LightSkyBlue'})
    
        index = 0
        for node in self.G.nodes():
            x, y = self.G.nodes[node]['pos']
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
        # for edge in self.G.edges:
        #     x0, y0 = self.G.nodes[edge[0]]['pos']
        #     x1, y1 = self.G.nodes[edge[1]]['pos']
        #     hovertext = "From: " + str(self.G.edges[edge]['from']) + "<br>" + "To: " + str(
        #         self.G.edges[edge]['to']) + "<br>" + "qt: " + strself.G.edges[edge]['qt'])
        #     middle_hover_trace['x'] += tuple([(x0 + x1) / 2])
        #     middle_hover_trace['y'] += tuple([(y0 + y1) / 2])
        #     middle_hover_trace['hovertext'] += tuple([hovertext])
        #     index = index + 1
    
        # traceRecode.append(middle_hover_trace)
        #################################################################################################################################################################
        figure = {
            "data": traceRecode,
            "layout": go.Layout(#title='Co-occurrences des mots '+str(WordsToSearch)+"\nNombre de noeuds : "+str(len(words)),
                                showlegend=False, hovermode='closest',
                                margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                                xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                height=600,
                                clickmode='event+select',
                                # annotations=[
                                #     dict(
                                #         # ax=(self.G.nodes[edge[0]]['pos'][0] + self.G.nodes[edge[1]]['pos'][0]) / 2,
                                #         # ay=(self.G.nodes[edge[0]]['pos'][1] + self.G.nodes[edge[1]]['pos'][1]) / 2, axref='x', ayref='y',
                                #         # x=(self.G.nodes[edge[1]]['pos'][0] * 3 + self.G.nodes[edge[0]]['pos'][0]) / 4,
                                #         # y=(self.G.nodes[edge[1]]['pos'][1] * 3 + self.G.nodes[edge[0]]['pos'][1]) / 4, xref='x', yref='y',
                                #         ax = self.G.nodes[edge[0]]['pos'][0],
                                #         ay = self.G.nodes[edge[0]]['pos'][1], axref='x', ayref='y',
                                #         x = self.G.nodes[edge[1]]['pos'][0],
                                #         y = self.G.nodes[edge[1]]['pos'][1], xref='x', yref='y',
                                #         showarrow=True,
                                #         arrowhead=0,
                                #         # arrowsize=4,
                                #         # arrowwidth=1,
                                #         opacity=1
                                #     ) for edge in self.G.edges]
                                )}
        return figure

    #PMI Methode
    # pmi function 
    def pmi_func(self, A, x, y): 
        freq_x = A.groupby(x).transform('count') 
        freq_y = A.groupby(y).transform('count') 
        freq_x_y = A.groupby([x, y]).transform('count') 
        return np.log(len(A.index) * (freq_x_y/(freq_x * freq_y))) 
    
    # pmi with kernel density estimation  
    def kernel_pmi_func(self, A, x, y): 
        # reshape data 
        x = np.array(A[x]) 
        y = np.array(A[y]) 
        x_y = np.stack((x, y), axis=-1) 
        
        # kernel density estimation 
        kde_x = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(x[:, np.newaxis]) 
        kde_y = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(y[:, np.newaxis]) 
        kde_x_y = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(x_y) 
        
        # score 
        p_x = pd.Series(np.exp(kde_x.score_samples(x[:, np.newaxis]))) 
        p_y = pd.Series(np.exp(kde_y.score_samples(y[:, np.newaxis]))) 
        p_x_y = pd.Series(np.exp(kde_x_y.score_samples(x_y))) 
        
        return np.log(p_x_y/(p_x * p_y)) 

    def callbacks(self):
        ###################################callback for the filter and the corpus settings components
        @self.app.callback(
            [dash.dependencies.Output('my-graph', 'figure'),
             dash.dependencies.Output('title', 'children')
             ],
            [dash.dependencies.Input('words', 'value'),
             dash.dependencies.Input('words', 'n_submit'),
             dash.dependencies.Input("reset", "n_clicks"),
             dash.dependencies.Input('theme', 'value'),
             dash.dependencies.Input('nbdocs', 'value')
             ])
        def update_output(words, ns, nc, theme, nb_docs):
            if nc != None and nc > self.NSUBMIT_corpus:
                self.NSUBMIT_corpus = nc
                if self.FIG == None or self.THEME != theme or self.NB_DOCS != nb_docs:
                    self.setup_corpus(theme, nb_docs)
                    self.FIG = self.network_graph()
            elif ns != None and ns > self.NSUBMIT_words:
                self.NSUBMIT_words = ns
                self.WORDS = words
                self.FIG = self.network_graph()
            return self.FIG, "Co-occurrences of words in the corpus - Number of words : "+str(self.nb_words)
        
        ################################callback for the nodes hovering
        @self.app.callback(
            dash.dependencies.Output('hover-data', 'children'),
            dash.dependencies.Input('my-graph', 'hoverData'))
        def display_hover_data(hoverData):
            if hoverData != None and "text" in hoverData["points"][0]:
                word = hoverData["points"][0]["text"]
            
                if word not in self.DEGCEN:
                    self.DEGCEN[word] = nx.degree_centrality(self.G)[word]
                if word not in self.CLOCEN:
                    self.CLOCEN[word] = nx.closeness_centrality(self.G, word)
                if word not in self.BETCEN:
                    self.BETCEN[word] = nx.betweenness_centrality(self.G)[word]
                
                frequencies = self.corpus.get_frequencies(word)
                
                datas = {"Word": word,
                         "Term frequency": int(frequencies["term frequency"].values[0]),
                         "Document frequency": int(frequencies["document frequency"].values[0]),
                         "Degree centrality": self.DEGCEN[word],
                         "Closeness centrality": self.CLOCEN[word],
                         "Betweenness centrality": self.BETCEN[word]}
                return json.dumps(datas, indent=2)
        
        ################################callback for the nodes clicking
        # @self.app.callback(
        #     dash.dependencies.Output('click-data', 'children'),
        #     [dash.dependencies.Input('my-graph', 'clickData')])
        # def display_click_data(clickData):
        #     word = clickData["points"][0]["text"]
        #     return word

if __name__ == '__main__':
    app = App(theme="data", nb_docs=20)