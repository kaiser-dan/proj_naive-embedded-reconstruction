# =========== SETUP ==========
import os
import shelve
import random

import numpy as np
import networkx as nx
from cdlib import algorithms
from sklearn import metrics
from tqdm import tqdm

from EMB.mplxio.readers import from_edgelist

jn = os.path.join
ROOT = jn("..", "..", "")
MPLX = jn(ROOT, "data", "edgelists", "")
OUT = jn(ROOT, "data", "")

import warnings
warnings.filterwarnings("ignore")

# =========== FUNCTIONS ==========
def duplex_network (G, l1, l2):
    G1 = G[l1].copy()
    G2 = G[l2].copy()

    ##delete common edges
    list_of_common_edges = []

    for e in G[l1].edges():
        if G[l2].has_edge(e[0], e[1]):
            list_of_common_edges.append([e[0], e[1]])

    # print (len(list_of_common_edges))

    for e in list_of_common_edges:
        G1.remove_edge(e[0], e[1])
        G2.remove_edge(e[0], e[1])

    ##delete nodes with zero degree
    list_of_nodes = []
    for n in G1.nodes():
        if G1.degree(n)==0:
            list_of_nodes.append(n)
    for n in list_of_nodes:
        G1.remove_node(n)

    list_of_nodes = []
    for n in G2.nodes():
        if G2.degree(n)==0:
            list_of_nodes.append(n)
    for n in list_of_nodes:
        G2.remove_node(n)

    ##create union of nodes
    list_of_nodes = []
    for n in G1.nodes():
        list_of_nodes.append(n)
    for n in G2.nodes():
        list_of_nodes.append(n)
    for n in list_of_nodes:
        G1.add_node(n)
        G2.add_node(n)

    return G1, G2

def partial_information (G1, G2, frac):
    ##training/test sets
    Etest = {}
    Etrain = {}

    for e in G1.edges():
        if random.random() < frac:
            Etrain[e] = 1
        else:
            Etest[e] = 1

    for e in G2.edges():
        if random.random() < frac:
            Etrain[e] = 0
        else:
            Etest[e] = 0

    ##remnants
    rem_G1 = nx.Graph()
    rem_G2 = nx.Graph()
    for n in G1:
        rem_G1.add_node(n)
        rem_G2.add_node(n)
    for n in G2:
        rem_G1.add_node(n)
        rem_G2.add_node(n)

    for e in Etest:
        rem_G1.add_edge(e[0], e[1])
        rem_G2.add_edge(e[0], e[1])

    for e in Etrain:
        if Etrain[e] == 1:
            rem_G1.add_edge(e[0], e[1])
        if Etrain[e] == 0:
            rem_G2.add_edge(e[0], e[1])

    return rem_G1, rem_G2, Etest

def community_finder (G):
    C = algorithms.louvain(G)
    sigma = {}
    c = 0
    for r in C.communities:
        for q in r:
            sigma[q] = c
        c = c + 1

    mu = tot = 0.0
    for n in G.nodes():
        for m in G.neighbors(n):
            tot += 1.0
            if sigma[n] == sigma[m]:
                mu += 1.0

    return sigma, mu, tot

def classifier (rem_G1, rem_G2, Etest, TT = 0, show_log = False):
    ## degree/community
    if TT == 0:
        sigma1, mu1, tot1 = community_finder(rem_G1)
        sigma2, mu2, tot2 = community_finder(rem_G2)


        mu = 0.5
        if tot1 + tot2 > 0.0:
            mu = (mu1 + mu2) / (tot1 + tot2)



        classification, scores, ground_truth = [], [], []

        for e in Etest:


            n = e[0]
            m = e[1]

            s1 = rem_G1.degree(n)*rem_G1.degree(m)
            if sigma1[n] == sigma1[m]:
                s1 = s1 * mu
            else:
                s1 = s1 * (1.0-mu)



            s2 = rem_G2.degree(n)*rem_G2.degree(m)
            if sigma2[n] == sigma2[m]:
                s2 = s2 * mu
            else:
                s2 = s2 * (1.0-mu)


            t1 = t2 = 0.5
            if s1 + s2 > 0.0:
                t1 = s1 / (s1 + s2)
                t2 = s2 / (s1 + s2)


            s = random.randint(0,1)
            if t1 > t2:
                s = 1
            if t2 > t1:
                s = 0

            if show_log == True:
                print (mu)
                print (rem_G1.degree(n), rem_G1.degree(m), t1)
                print (rem_G2.degree(n), rem_G2.degree(m), t2)
                print (Etest[e], '\n')



            scores.append(t1)
            classification.append(s)
            ground_truth.append(Etest[e])



        return classification, scores, ground_truth

    ## degree
    if TT == 1:

        classification, scores, ground_truth = [], [], []

        for e in Etest:


            n = e[0]
            m = e[1]

            s1 = rem_G1.degree(n)*rem_G1.degree(m)
            s2 = rem_G2.degree(n)*rem_G2.degree(m)


            t1 = t2 = 0.5
            if s1 + s2 > 0.0:
                t1 = s1 / (s1 + s2)
                t2 = s2 / (s1 + s2)


            s = random.randint(0,1)
            if t1 > t2:
                s = 1
            if t2 > t1:
                s = 0

            scores.append(t1)
            classification.append(s)
            ground_truth.append(Etest[e])


        return classification, scores, ground_truth

    ## community
    if TT == 2:
        sigma1, mu1, tot1 = community_finder(rem_G1)
        sigma2, mu2, tot2 = community_finder(rem_G2)

        mu = 0.5
        if tot1 + tot2 > 0.0:
            mu = (mu1 + mu2) / (tot1 + tot2)


        classification, scores, ground_truth = [], [], []

        for e in Etest:


            n = e[0]
            m = e[1]

            s1 = 1.0
            if sigma1[n] == sigma1[m]:
                s1 = s1 * mu
            else:
                s1 = s1 * (1.0-mu)



            s2 = 1.0
            if sigma2[n] == sigma2[m]:
                s2 = s2 * mu
            else:
                s2 = s2 * (1.0-mu)


            t1 = t2 = 0.5
            if s1 + s2 > 0.0:
                t1 = s1 / (s1 + s2)
                t2 = s2 / (s1 + s2)


            s = random.randint(0,1)
            if t1 > t2:
                s = 1
            if t2 > t1:
                s = 0



            scores.append(t1)
            classification.append(s)
            ground_truth.append(Etest[e])



        return classification, scores, ground_truth

def perform_analysis (G1, G2, step, TT = 0):
    x , y, z  = [], [], []

    for frac in tqdm(np.linspace(0.0, 1-step, num=int((1)/step)), desc="PFIs", position=2, leave=False, colour="red"):
        rem_G1, rem_G2, Etest  = partial_information(G1, G2, frac)
        classification, scores, ground_truth = classifier (rem_G1, rem_G2, Etest, TT)
        acc = metrics.accuracy_score(ground_truth, classification)
        auc = metrics.roc_auc_score(ground_truth, scores)
        prec, rec, _ = metrics.precision_recall_curve(ground_truth, scores)
        pr = metrics.auc(rec, prec)

        x.append(frac)
        y.append(pr)
        z.append(auc)

    results = [x, y, z]
    return results


# ========== MAIN ===========
def main(filepath, l1, l2):
    D = from_edgelist(filepath)
    G, H = D[l1], D[l2]
    results = perform_analysis(G, H, 0.05)

    for (pfi, pr, auc) in zip(*results):
        print(f"{filepath},{l1}-{l2},{pfi},{auc},{pr}", file=open("dc-extreal.csv", "a"))

if __name__ == "__main__":
    FPS = [
        filepath for filepath in os.listdir(MPLX)
        if (("rattus" in filepath) \
            or ("euair" in filepath) \
            or ("homo" in filepath) \
            or ("pomb" in filepath)) \
            and ('l1-1_l2-2' in filepath)
    ]
    for filepath in tqdm(FPS, desc="Extended Real Corpus", position=0, leave=True, colour="blue"):
        filepath = jn(MPLX, filepath)
        for _ in tqdm(range(5), desc="Reps", colour="green", position=1, leave=False):
            try:
                main(filepath, 1, 2)
            except KeyError as err:
                print(">>>", filepath)
                raise err
