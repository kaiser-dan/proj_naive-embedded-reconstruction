"""Project source code for applying Node2Vec embedding.

Wrapper for node2vec package with
additional post-processing for non-consecutive node ids.
"""
# ============= SET-UP =================
# --- Standard library ---
import random

# --- Network science ---
import networkx as nx
from gensim.models import Word2Vec

# --- Miscellaneous ---
from embmplxrec.embed.helpers import get_contiguous_vectors
from embmplxrec.embed.embedding import Embedding
import embmplxrec.hypercomparison.utils
import embmplxrec.hypercomparison.networks

# ============= CLASSES =================
class Node2Vec:
    def __init__(
        self,
        dimensions=128,
        walk_length=80,
        num_walks=10,
        window_size=10,
        workers=1,
        iteration=1,
        p=1,
        q=1
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
            # iter=self.iteration,
        )
        self.id2node = dict([(vid, node) for vid, node in enumerate(G.nodes())])
        #temp_embeddings = np.array([list(model.wv[str(self.id2node[i])]) for i in range(len(self.id2node))]) # centering the coordinates
        #center_point = temp_embeddings.mean(axis=0) #
        #temp_embeddings -= center_point #
        #self.embeddings = {str(self.id2node[i]): temp_embeddings[i] for i in range(len(self.id2node))}
        self.embeddings = {
            str(self.id2node[i]): model.wv[str(self.id2node[i])] for i in range(len(self.id2node))
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
                        cur_nbrs[embmplxrec.hypercomparison.utils.alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]
                    )
                else:
                    prev = walk[-2]
                    next = cur_nbrs[
                        embmplxrec.hypercomparison.utils.alias_draw(
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
        # logger.info("Walk iteration:")
        for walk_iter in range(num_walks):
            # if walk_iter % 10 == 0:
                # logger.info(str(walk_iter + 1) + "/" + str(num_walks))
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

        return embmplxrec.hypercomparison.utils.alias_setup(normalized_probs)

    def _preprocess_transition_probs(self):
        # Preprocessing of transition probabilities for guiding the random walks.
        G = self.G
        is_directed = nx.is_directed(self.G)

        # logger.info(len(list(G.nodes())))
        # logger.info(len(list(G.edges())))

        # s = time.time()
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]["weight"] for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob) / norm_const for u_prob in unnormalized_probs
            ]
            alias_nodes[node] = embmplxrec.hypercomparison.utils.alias_setup(normalized_probs)

        # t = time.time()
        # logger.info("alias_nodes {}".format(t - s))

        alias_edges = {}
        # s = time.time()

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self._get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self._get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self._get_alias_edge(edge[1], edge[0])

        # t = time.time()
        # logger.info("alias_edges {}".format(t - s))

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return

    # def to_data_frame(self):
    #     embedding_df = pd.DataFrame(self.embeddings)
    #     node_id_df = pd.DataFrame(list(self.id2node.items()), columns=['index', 'node_id'])
    #     self.embedding_df = node_id_df.merge(embedding_df, left_on='index', right_index=True)


# ============= FUNCTIONS =================
def N2V(
        graph: nx.Graph,
        parameters: dict, hyperparameters: dict,
        per_component: bool = False):
    """Embed `graph` using node2vec.

    Parameters
    ----------
    graph : nx.Graph
        Graph to embed. Node and edge attributes are ignored.
    parameters : dict
        Keyword arguments for node2vec walk generation.
    hyperparameters : dict
        Keyword arguments for word2vec fitting on node2vec-generated walks.
    per_component: bool, optional
        Embed each graph component separately, by default False.

    Returns
    -------
    Embedding
        Embedding class instance.
    """
    # # >>> Dispatch >>>
    # if per_component:
    #     return _N2V_per_component(graph, parameters, hyperparameters)

    # # <<< Dispatch <<<

    # # Sample random walks
    # embedding_model = n2v(graph, **parameters)

    # # Embed walks with word2vec and retrieve model
    # embedding_model = embedding_model.fit(**hyperparameters)
    # embedding_model = embedding_model.wv

    vectors = Node2Vec(**{k: v for k, v in parameters.items() if k in ["dimensions", "num_walks", "walk_length"]}).train(graph)

    # Retrieve resultant vectors
    # vectors = get_contiguous_vectors(embedding_model)

    embedding = Embedding(vectors, "N2V")

    return embedding


def _N2V_per_component(graph: nx.Graph, parameters: dict, hyperparameters: dict):
    # >>> Book-keeping >>>
    vectors_per_component = []  # list of vector embeddings, canonical ordering
    vectors = {}  # amalgamated mapping of nodes to their embedded vectors (by component)
    # <<< Book-keeping <<<

    # >>> Embedding >>>
    # Retrieve each component as a graph
    component_subgraphs = sorted(
        [
            graph.subgraph(component).copy()
            for component in nx.connected_components(graph)
        ],
        key=len,reverse=True
    )

    # Embed each component by themselves
    for component_subgraph in component_subgraphs:
        vectors_per_component.append(N2V(component_subgraph, parameters, hyperparameters).vectors)

    # Amalgamate results
    for component_vectors in vectors_per_component:
        for node, vector in component_vectors.items():
            vectors[node] = vector
    # <<< Embedding <<<

    return Embedding(vectors, "N2V")