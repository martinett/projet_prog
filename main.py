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
    #converti un float en str sans avoir de représentation exponentielle (écriture scientifique)
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

def color_to_str(c):
    #converti une couleur en rgb sous forme de texte (sans écriture scientifique)
    return "("+", ".join(map(float_to_str, c))+")"


class App:
    def __init__(self, theme="data", nb_docs=20):
        #nombre de fois qu'on a appuyé sur entrer dans l'input de filtre
        self.NSUBMIT_words = 0
        #nombre de fois qu'on a appuyé sur le bouton de reset du corpus
        self.NSUBMIT_corpus = 0
        #figure contenant le graphe
        self.FIG = None
        
        self.setup_corpus(theme, nb_docs)
        self.setup_dash()
        self.callbacks()
        self.launch()

    def setup_corpus(self, theme, nb_docs):
        #contient les mots filtrés
        self.WORDS = theme+";"
        #les 3 métriques de centralités (pour chaque mot : dictionnaire)
        self.DEGCEN = {}
        self.CLOCEN = {}
        self.BETCEN = {}
        #le thème du corpus
        self.THEME = theme
        #nombre de documents du corpus
        self.NB_DOCS = nb_docs
        #le corpus
        self.corpus = Corpus(theme)
        self.corpus.download_collection(nb_docs, keyword=theme)
        self.A = self.corpus.get_adjacency_matrix()

    def setup_dash(self):
        # crée l'application dash,
        # paramètrise le css,
        # et l'envoie dans l'application
        self.app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
        self.app.title = "Projet Prog"
        ######################################################################################################################################################################
        # styles: pour les composants de droite : click et hover
        styles = {
            'pre': {
                'border': 'thin lightgrey solid',
                'overflowX': 'scroll'
            }
        }
        
        self.FIG = self.network_graph()
        
        #composants de l'interface graphique
        self.app.layout = html.Div([
            #########################Title
            html.Div([html.H1("Co-occurrences of words in the corpus '"+self.corpus.name+"' - Number of words : "+str(self.nb_words), id="title")],
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
                                style={'height': '250px'}),
        
                            html.Div(
                                className='twelve columns',
                                children=[
                                    dcc.Markdown(d("**Click Data**")),
                                    html.Pre(id='click-data', style=styles['pre'])
                                ],
                                style={'height': '250px'})
                        ]
                    )
                ]
            )
        ])
    
    def launch(self):
        # lance l'appli
        self.app.run_server()

    def network_graph(self):
        #génère le graphe
        
        #récupère et sépare les mots filtrés
        WordsToSearch = list(set(self.WORDS.split(";")))
        if "" in WordsToSearch:
            WordsToSearch.remove("")
        
        #filtre par rapport à l'input de filtre
        words = WordsToSearch.copy()
        #si il y a des mots dans le filtre
        if len(words) > 0:
            #on filtre la matrice
            words.extend(self.A.loc[words].loc[:,(self.A.loc[words]!=0).any(axis=0)].columns)
        else:
            #sinon on ne filtre rien et on prend tout
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
        
        #calcul des différentes couches (shell) du graphe (mots organisés en cercles : centre = plus fréquent)
        #on détermine le nombre de couches (nb_shells)
        base = 3
        nwords = len(words)
        initinA = list(Ap.index[Ap.index.isin(WordsToSearch)])
        ninitinA = len(initinA)
        n = nwords-ninitinA
        if ninitinA > 0:
            nb_shells = ceil(log(n,base)-log(ninitinA,base))
        else:
            nb_shells = ceil(log(n,base))
        #puis la taille de base d'une couche
        s = (base-1)*n/(base**nb_shells-1)
        if ninitinA > 0:
            shells = [WordsToSearch]
            flat_shells = initinA.copy()
        else:
            shells = []
            flat_shells = []
        #puis on crée chaque couche
        for i in range(int(nb_shells+0.5)-1):
            #x est le nombre de mot pour la couche i
            x = ceil(s*base**i)
            if len(flat_shells) > 0:
                shell = list(Ap[Ap.loc[flat_shells] != 0].loc[:,~Ap.index.isin(flat_shells)].max().sort_values(ascending=False).head(x).index)
            else:
                shell = list(Ap.max().sort_values(ascending=False).head(x).index)
            flat_shells.extend(shell)
            shells.append(shell)
        #la dernière couche étant tout ce qui n'est pas dans les autres couches
        shell = list(Ap.index[~Ap.index.isin(flat_shells)])
        shells.append(shell)
        
        #on envoie ensuite les couches au moteur de networkx
        pos = nx.drawing.layout.shell_layout(self.G, shells, dim=2)
        
        #autres tests de rendu du graphe : peu concluant
        # pos = nx.drawing.layout.random_layout(self.G, dim=2, center=None)
        # pos = nx.drawing.layout.spring_layout(self.G)
        # pos = nx.drawing.layout.kamada_kawai_layout(self.G)
        # pos = nx.drawing.layout.spectral_layout(self.G)
        # pos = nx.drawing.layout.spiral_layout(self.G)
        
        #on récupère alors les positions des noeuds de networkx et on les stocke quelque part où on peut revenir les chercher
        for node in self.G.nodes:
            self.G.nodes[node]['pos'] = list(pos[node])
        
        
        #traceRecode va contenir tous les éléments du graphe
        traceRecode = []
        ############################################################################################################################################################
        #gestion des couleurs des arrêtes
        colors = list(Color('lightyellow').range_to(Color('darkred'), max(edge1['qt'])-min(edge1['qt'])+1))
        colors = ['rgb' + color_to_str(x.rgb) for x in colors]
        #et de leur transparance
        alphas = [((i-min(edge1['qt']))/(max(edge1['qt'])-min(edge1['qt']))/10)+0.05 for i in range(min(edge1['qt']),max(edge1['qt'])+1)]
    
        #on crée ensuite un trait pour chaque arrête
        for edge in self.G.edges:
            x0, y0 = self.G.nodes[edge[0]]['pos']
            x1, y1 = self.G.nodes[edge[1]]['pos']
            trace = go.Scatter(x=tuple([x0, x1, None]), y=tuple([y0, y1, None]),
                                mode='lines',
                                line={'width': 1},
                                marker=dict(color=colors[self.G.edges[edge]['qt']-min(edge1['qt'])]),
                                line_shape='spline',
                                opacity=alphas[self.G.edges[edge]['qt']-min(edge1['qt'])])
            traceRecode.append(trace)
        ###############################################################################################################################################################
        #on crée ensuite les marker pour les noeuds
        sizes = Ap.sum().values*5/len(words)+5
        node_trace = go.Scatter(x=[], y=[], hovertext=[], text=[], mode='markers+text', textposition="bottom center",
                                #textfont=dict(color="Orange"),
                                hoverinfo="text",
                                marker={'size': sizes, 'color': 'LightSkyBlue'})
    
        #puis on stocke les informations des noeuds dans traceRecode
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
        #################################################################################################################################################################
        #enfin on crée une "figure" qui pourra être utilisée par dash et plotly
        figure = {
            "data": traceRecode,
            "layout": go.Layout(showlegend=False, hovermode='closest',
                                margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                                xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                height=600,
                                clickmode='event'
                                )}
        return figure

    #méthode PMI
    def pmi_func(self, A, x, y):
        freq_x = A.groupby(x).transform('count')
        freq_y = A.groupby(y).transform('count')
        freq_x_y = A.groupby([x, y]).transform('count')
        return np.log(len(A.index) * (freq_x_y/(freq_x * freq_y)))
    
    #approximation de la méthode PMI avec KernelDensity
    def kernel_pmi_func(self, A, x, y):
        x = np.array(A[x])
        y = np.array(A[y])
        x_y = np.stack((x, y), axis=-1)
        
        kde_x = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(x[:, np.newaxis])
        kde_y = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(y[:, np.newaxis])
        kde_x_y = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(x_y)
        
        p_x = pd.Series(np.exp(kde_x.score_samples(x[:, np.newaxis])))
        p_y = pd.Series(np.exp(kde_y.score_samples(y[:, np.newaxis])))
        p_x_y = pd.Series(np.exp(kde_x_y.score_samples(x_y)))
        
        return np.log(p_x_y/(p_x * p_y))

    def callbacks(self):
        ###################################événements pour le reset de corpus et le filtre de mots
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
            #cette fonction sera appelée si on appuie sur le bouton de reset ou si on appuie sur entrée dans le filtre
            #si on a appuyé sur le bouton
            if nc != None and nc > self.NSUBMIT_corpus:
                self.NSUBMIT_corpus = nc
                #si le theme ou le nombre de docs à changé
                if self.FIG == None or self.THEME != theme or self.NB_DOCS != nb_docs:
                    #on re setup le corpus
                    self.setup_corpus(theme, nb_docs)
                    #et on recalcule le graphe
                    self.FIG = self.network_graph()
            #sinon si on a appuyé sur entrée
            elif ns != None and ns > self.NSUBMIT_words:
                #on recalcule le graphe
                self.NSUBMIT_words = ns
                self.WORDS = words
                self.FIG = self.network_graph()
            #on renvoie le graphe et le titre
            return self.FIG, "Co-occurrences of words in the corpus '"+self.corpus.name+"' - Number of words : "+str(self.nb_words)
        
        ################################événement pour le hover (survol) des noeuds
        @self.app.callback(
            dash.dependencies.Output('hover-data', 'children'),
            dash.dependencies.Input('my-graph', 'hoverData'))
        def display_hover_data(hoverData):
            #cette fonction sera appelée si on survole un noeud
            #si on survole bien un noeud
            if hoverData != None and "text" in hoverData["points"][0]:
                #on récupère le mot du noeud survolé
                word = hoverData["points"][0]["text"]
                
                #si on n'a jamais calculé une des métriques, on la calcule
                if word not in self.DEGCEN:
                    self.DEGCEN[word] = nx.degree_centrality(self.G)[word]
                if word not in self.CLOCEN:
                    self.CLOCEN[word] = nx.closeness_centrality(self.G, word)
                if word not in self.BETCEN:
                    self.BETCEN[word] = nx.betweenness_centrality(self.G)[word]
                
                #on récupère les fréquences du mot dans le corpus
                frequencies = self.corpus.get_frequencies(word)
                
                #et on renvoie tout ça en json
                datas = {"Word": word,
                         "Term frequency": int(frequencies["term frequency"].values[0]),
                         "Document frequency": int(frequencies["document frequency"].values[0]),
                         "Degree centrality": self.DEGCEN[word],
                         "Closeness centrality": self.CLOCEN[word],
                         "Betweenness centrality": self.BETCEN[word]}
                return json.dumps(datas, indent=2)
        
        ###############################événement pour le clic des noeuds
        @self.app.callback(
            dash.dependencies.Output('click-data', 'children'),
            [dash.dependencies.Input('my-graph', 'clickData')])
        def display_click_data(clickData):
            #cette fonction sera appelée si on clique sur un noeud
            #si on clique bien sur un noeud
            if clickData != None and "text" in clickData["points"][0]:
                #on récupère le mot du noeud cliqué
                word = clickData["points"][0]["text"]
            
                #si on n'a jamais calculé une des métriques, on la calcule
                if word not in self.DEGCEN:
                    self.DEGCEN[word] = nx.degree_centrality(self.G)[word]
                if word not in self.CLOCEN:
                    self.CLOCEN[word] = nx.closeness_centrality(self.G, word)
                if word not in self.BETCEN:
                    self.BETCEN[word] = nx.betweenness_centrality(self.G)[word]
                
                #on récupère les fréquences du mot dans le corpus
                frequencies = self.corpus.get_frequencies(word)
                
                #et on renvoie tout ça en json
                datas = {"Word": word,
                         "Term frequency": int(frequencies["term frequency"].values[0]),
                         "Document frequency": int(frequencies["document frequency"].values[0]),
                         "Degree centrality": self.DEGCEN[word],
                         "Closeness centrality": self.CLOCEN[word],
                         "Betweenness centrality": self.BETCEN[word]}
                return json.dumps(datas, indent=2)
        
if __name__ == '__main__':
    #si on exécute ce fichier, l'appli se lancera avec de base un corpus à 20 documents sur le thème de la data
    app = App(theme="data", nb_docs=20)