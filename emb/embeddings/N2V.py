"""Node2Vec graph embedding utility.
"""
# ============= SET-UP =================
__all__ = ["embed_N2V", "embed_multiplex_N2V"]

# --- Imports ---
import random

import numpy as np
import networkx as nx
from gensim.models import Word2Vec

from loguru import logger as LOGGER

# --- Globals ---
ACCEPTED_PARAMS_N2V = {"dimensions", "walk_length", "num_walks"}
PANIC = False


# ========== CLASSES ==========
class Node2Vec:
    """Adapted from https://github.com/yijiaozhang/hypercompare"""

    def __init__(
        self,
        dimensions=128,
        walk_length=80,
        num_walks=10,
        window_size=10,
        workers=1,
        iteration=1,
        p=1,
        q=1,
    ):
        self.dimension = dimensions
        self.walk_length = walk_length
        self.walk_num = num_walks
        self.window_size = window_size
        self.worker = workers
        self.iteration = iteration
        self.p = p
        self.q = q

    def train(self, G):
        self.G = G
        is_directed = nx.is_directed(self.G)
        for i, j in G.edges():
            G[i][j]["weight"] = G[i][j].get("weight", 1.0)
            if not is_directed:
                G[j][i]["weight"] = G[j][i].get("weight", 1.0)
        self._preprocess_transition_probs()
        walks = self._simulate_walks(self.walk_num, self.walk_length)
        walks = [[str(node) for node in walk] for walk in walks]
        model = Word2Vec(
            walks,
            vector_size=self.dimension,
            window=self.window_size,
            min_count=0,
            sg=1,
            workers=self.worker,
        )
        self.id2node = dict([(vid, node) for vid, node in enumerate(G.nodes())])
        self.embeddings = {
            int(self.id2node[i]): model.wv[str(self.id2node[i])]
            for i in range(len(self.id2node))
        }
        return self.embeddings

    def _node2vec_walk(self, walk_length, start_node):
        # Simulate a random walk starting from start node.
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]
                    )
                else:
                    prev = walk[-2]
                    next = cur_nbrs[
                        alias_draw(
                            alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1]
                        )
                    ]
                    walk.append(next)
            else:
                break

        return walk

    def _simulate_walks(self, num_walks, walk_length):
        # Repeatedly simulate random walks from each node.
        G = self.G
        walks = []
        nodes = list(G.nodes())
        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(
                    self._node2vec_walk(walk_length=walk_length, start_node=node)
                )

        return walks

    def _get_alias_edge(self, src, dst):
        # Get the alias edge setup lists for a given edge.
        G = self.G
        unnormalized_probs = []
        for dst_nbr in G.neighbors(dst):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"] / self.p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]["weight"])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"] / self.q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def _preprocess_transition_probs(self):
        # Preprocessing of transition probabilities for guiding the random walks.
        G = self.G
        is_directed = nx.is_directed(self.G)

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]["weight"] for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob) / norm_const for u_prob in unnormalized_probs
            ]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self._get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self._get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self._get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.

    Input:
        J::numpy.ndarray: alias lookup table
        q::numpy.ndarray: threshold table

    Return:
        ::Int: an index randomly selected according to the probability list
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


# Function implemenentation for node2vec etc
def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to http://cgi.cs.mcgill.ca/~enewel3/posts/alias-method/index.html
    for details

    Input:
        probs::list: a list of probabilites
                     ensure sum(probs) == 1

    Return:
        J::numpy.ndarray: alias lookup table
        q::numpy.ndarray: threshold table
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


# =================== FUNCTIONS ===================
def embed_N2V(graph, **kwargs):
    vectors = Node2Vec(**kwargs).train(graph)

    return vectors


def embed_multiplex_N2V(multiplex, **kwargs):
    vectors = {label: embed_N2V(graph, **kwargs) for label, graph in multiplex.items()}

    return vectors


# --- Helpers ---
def _check_kwargs(_panic=PANIC, **kwargs):
    keys = [key for key in kwargs.keys()]
    kwargs = kwargs.copy()

    for key in keys:
        if key not in ACCEPTED_PARAMS_N2V:
            if _panic:
                raise ValueError(
                    f"'{key}' is not a N2V parameter or not designated as mutable!"
                )
            else:
                LOGGER.warning(
                    f"'{key}' is not a N2V parameter or not designated as mutable! Panic disabled - dropping '{key}' from kwargs."
                )
                del kwargs[key]

    return kwargs
