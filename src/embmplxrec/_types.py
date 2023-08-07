from typing import Union, Tuple, FrozenSet, List
import networkx.classes.reportviews

# Edge
AbstractEdge = Tuple[int, int]

# Collection of edges
AbstractEdges = Union[
    FrozenSet[AbstractEdge],
    networkx.classes.reportviews.EdgeView]

# Maybe container for collection of previously observed edges
MaybePrevs = Union[None, List[AbstractEdges]]