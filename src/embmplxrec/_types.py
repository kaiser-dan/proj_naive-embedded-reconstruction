from typing import Union, Tuple, Set, List, Dict
import networkx.classes.reportviews
from networkx import Graph

# Edge
AbstractEdge = Tuple[int, int]

# Collection of edges
AbstractEdgeset = Set[AbstractEdge]
AbstractEdgesetLabeled = Dict[AbstractEdge, int]
AbstractEdgesetView = networkx.classes.reportviews.EdgeView
AbstractEdges = Union[
    AbstractEdgeset,
    AbstractEdgesetLabeled,
    AbstractEdgesetView
]

# Collection of collection of edges
AbstractEdgesList = List[AbstractEdges]

# Maybe container for collection of previously observed edges
MaybePrevs = Union[None, List[AbstractEdges]]

# nx.Graph container
Graphs = List[Graph]

