# standard
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from functools import wraps
from types import MappingProxyType
from typing import Any, Self


@dataclass(slots=True, frozen=True)
class Vertex[K: Hashable]:
    id: K
    attributes: MappingProxyType[str, Any]

@dataclass(slots=True, frozen=True)
class Edge[K]:
    start: Vertex[K]
    end: Vertex[K]
    attributes: MappingProxyType[str, Any]

class WeightedDigraph[K: Hashable]:
    _DEBUG = False

    __slots__ = ("default_vertex_attributes", "default_edge_attributes", "_vertices", "_out_edges", "_in_edges")

    def __init__(
        self,
        default_vertex_attributes: dict[str, Any] | None = None,
        default_edge_attributes: dict[str, Any] | None = None
    ):
        self.default_vertex_attributes = MappingProxyType(default_vertex_attributes.copy() if default_vertex_attributes else {})
        self.default_edge_attributes = MappingProxyType(default_edge_attributes.copy() if default_edge_attributes else {})

        # vertex -> vertex attributes
        self._vertices: dict[K, dict[str, Any]] = {}

        # start -> end -> edge attributes
        self._out_edges: dict[K, dict[K, dict[str, Any]]] = {}

        # end -> start -> edge attributes
        self._in_edges: dict[K, dict[K, dict[str, Any]]] = {}
        
        self._conditional_invariant_check()

    def _check_invariant(f):
        @wraps(f)
        def wrapper(self, *args, **kwargs):
            self._conditional_invariant_check()
            res = f(self, *args, **kwargs)
            self._conditional_invariant_check()
            return res
        return wrapper

    #===================================
    # public methods
    #===================================

    def contains_vertex(self, vertex: K) -> bool:
        return vertex in self._vertices

    # ! using 'allow_exists' as a keyword parameter shadows possible attribute name
    @_check_invariant
    def add_vertex(self, vertex: K, /, allow_exists: bool = False, **kwargs) -> bool:
        vertex_exists = self.contains_vertex(vertex)
        if vertex_exists and not allow_exists:
            raise ValueError(f"vertex {vertex} already exists")
        elif not vertex_exists:
            self._add_vertex(vertex, kwargs)
            added = True
        else:
            added = False
        return added
    
    @_check_invariant
    def set_vertex_attributes(self, vertex: K, /, **kwargs):
        attributes = self._vertices.get(vertex)
        if attributes is None:
            raise ValueError(f"no vertex {vertex}")
        attributes.update(kwargs)

    @_check_invariant
    def remove_vertex(self, vertex: K, allow_not_exists: bool = False) -> bool:
        vertex_exists = self.contains_vertex(vertex)
        if not vertex_exists and not allow_not_exists:
            raise ValueError(f"vertex {vertex} does not exist")
        elif vertex_exists:
            del self._vertices[vertex]

            # remove outgoing edges
            for end in self._out_edges[vertex]:
                del self._in_edges[end][vertex]
            del self._out_edges[vertex]

            # remove incoming edges
            for start in self._in_edges[vertex]:
                del self._out_edges[start][vertex]
            del self._in_edges[vertex]

            removed = True
        else:
            removed = False
        return removed
    
    def contains_edge(self, start: K, end: K) -> bool:
        return end in self._out_edges.get(start, ())

    @_check_invariant
    def add_edge(self, start: K, end: K, /, allow_create_vertices: bool = False, allow_update: bool = False, **kwargs) -> bool:
        if allow_create_vertices:
            # create vertices if they don't already exist
            if not self.contains_vertex(start):
                self._add_vertex(start)
            if not self.contains_vertex(end):
                self._add_vertex(end)
            start_exists = True
            end_exists = True
        else:
            start_exists = self.contains_vertex(start)
            end_exists = self.contains_vertex(end)
        
        if not start_exists:
            raise ValueError(f"start vertex {start} does not exist")
        if not end_exists:
            raise ValueError(f"end vertex {end} does not exist")
        
        edge_attributes = self._out_edges.get(start, {}).get(end)
        edge_exists = edge_attributes is not None
        if not allow_update and edge_exists:
            raise ValueError(f"edge from {start} to {end} already exists")
        if not edge_exists:
            edge_attributes = {
                **self.default_edge_attributes,
                **kwargs
            }
            self._out_edges[start][end] = edge_attributes  # shared reference => only need to modify one
            self._in_edges[end][start] = edge_attributes
        else:  # update
            edge_attributes.update(kwargs)

        return not edge_exists

    @_check_invariant
    def set_edge_attributes(self, start: K, end: K, /, **kwargs):
        attributes = self._out_edges.get(start, {}).get(end)
        if attributes is None:
            raise ValueError(f"no edge from {start} to {end}")
        attributes.update(kwargs)

    @_check_invariant
    def remove_edge(self, start: K, end: K, allow_not_exists: bool = False) -> bool:
        if not self.contains_vertex(start):
            raise ValueError(f"start vertex {start} does not exist")
        if not self.contains_vertex(end):
            raise ValueError(f"end vertex {end} does not exist")
        
        from_start = self._out_edges[start]
        edge_exists = end in from_start
        if not edge_exists and not allow_not_exists:
            raise ValueError(f"no edge from {start} to {end}")
        elif edge_exists:
            del from_start[end]
            del self._in_edges[end][start]
            removed = True
        else:
            removed = False
        return removed

    # ! live view of attributes
    def get_vertex(self, vertex: K) -> Vertex[K] | None:
        attributes = self._vertices.get(vertex)
        return Vertex(vertex, MappingProxyType(attributes)) if attributes is not None else None

    def vertices(self) -> Iterable[Vertex[K]]:
        for vertex, attributes in self._vertices.items():
            yield Vertex(vertex, MappingProxyType(attributes))
    
    # ! live view of attributes
    def get_edge(self, start: K, end: K) -> Edge[K] | None:
        from_start = self._out_edges.get(start)
        if from_start is None:
            raise ValueError(f"start vertex {start} does not exist")
        
        edge_attributes = from_start.get(end)
        if edge_attributes is None and not self.contains_vertex(end):
            # logically equivalent to just checking existence of end vertex (existence of start->end edge
            # implies existence of end vertex); short-circuits lookup in case where edge exists
            raise ValueError(f"end vertex {end} does not exist")
        
        return (
            Edge(
                self.get_vertex(start),
                self.get_vertex(end),
                MappingProxyType(edge_attributes)
            ) if edge_attributes is not None
            else None
        )

    def edges(self) -> Iterable[Edge[K]]:
        for start, end_attributes in self._out_edges.items():
            for end, attributes in end_attributes.items():
                yield Edge(self.get_vertex(start), self.get_vertex(end), MappingProxyType(attributes))

    def out_edges(self, start: K) -> Iterable[Edge[K]]:
        from_start = self._out_edges.get(start)
        if from_start is None:
            raise ValueError(f"start vertex {start} does not exist")
        for end, attributes in from_start.items():
            yield Edge(self.get_vertex(start), self.get_vertex(end), MappingProxyType(attributes))
    
    def in_edges(self, end: K) -> Iterable[Edge[K]]:
        from_end = self._in_edges.get(end)
        if from_end is None:
            raise ValueError(f"end vertex {end} does not exist")
        for start, attributes in from_end.items():
            yield Edge(self.get_vertex(start), self.get_vertex(end), MappingProxyType(attributes))

    def out_degree(self, start: K) -> int:
        from_start = self._out_edges.get(start)
        if from_start is None:
            raise ValueError(f"start vertex {start} does not exist")
        return len(from_start)

    def in_degree(self, end: K) -> int:
        from_end = self._in_edges.get(end)
        if from_end is None:
            raise ValueError(f"end vertex {end} does not exist")
        return len(from_end)

    def copy(self) -> Self:
        G = WeightedDigraph()

        G.default_vertex_attributes = self.default_vertex_attributes  # immutable, so shared reference is fine
        G.default_edge_attributes = self.default_edge_attributes

        G._vertices = {vertex: attributes.copy() for vertex, attributes in self._vertices.items()}
        G._out_edges = {
            start: {end: attributes.copy() for end, attributes in end_attributes.items()}
            for start, end_attributes in self._out_edges.items()
        }
        G._in_edges = {  # G._in_edges[end][start] = G._out_edges[start][end]
            # shared edge attribute reference between _out_edges and _in_edges
            end: {start: G._out_edges[start][end] for start in start_attributes.keys()}
            for end, start_attributes in self._in_edges.items()
        }

        return G

    def reverse(self, inplace: bool = False) -> Self:
        if inplace:
            self._out_edges, self._in_edges = self._in_edges, self._out_edges
            G = self
        else:
            G = WeightedDigraph()

            G.default_vertex_attributes = self.default_vertex_attributes
            G.default_edge_attributes = self.default_edge_attributes

            G._vertices = {vertex: attributes.copy() for vertex, attributes in self._vertices.items()}
            G._out_edges = {  # G._out_edges[u][v] = self._out_edges[v][u] = self._in_edges[u][v]
                u: {v: attributes.copy() for v, attributes in v_attributes.items()}
                for u, v_attributes in self._in_edges.items()
            }
            G._in_edges = {  # G._in_edges[u][v] = G._out_edges[v][u] = self._out_edges[u][v]
                u: {v: G._out_edges[v][u] for v in v_attributes.keys()}
                for u, v_attributes in self._out_edges.items()
            }

        return G

    def freeze(self) -> _FrozenWeightedDigraph[K]:
        return _FrozenWeightedDigraph(self)

    @classmethod
    def from_dict[K: Hashable](
        cls,
        edges: dict[K, list[K] | dict[K, dict[str, Any]]],
        vertex_attributes: dict[K, dict[str, Any]] | None = None,
        default_vertex_attributes: dict[str, Any] | None = None,
        default_edge_attributes: dict[str, Any] | None = None
    ) -> Self:
        vertex_attributes = vertex_attributes if vertex_attributes is not None else {}

        G = cls(default_vertex_attributes, default_edge_attributes)

        # create vertices (even those without edges)
        for vertex, attributes in vertex_attributes.items():
            G.add_vertex(vertex, allow_exists=False, **attributes)

        # ensure vertices in edges, create edges
        for start, end_list_or_dict in edges.items():
            G.add_vertex(start, allow_exists=True, **vertex_attributes.get(start, {}))

            if isinstance(end_list_or_dict, list):
                for end in end_list_or_dict:
                    G.add_vertex(end, allow_exists=True)
                    G.add_edge(start, end, allow_create_vertices=False)
            else:
                assert isinstance(end_list_or_dict, dict)
                for end, attributes in end_list_or_dict.items():
                    G.add_vertex(end, allow_exists=True, **vertex_attributes.get(end, {}))
                    G.add_edge(start, end, allow_create_vertices=False, **attributes)
        
        return G


    #===================================
    # private implementations/helpers
    #===================================

    def _conditional_invariant_check(self):
        if self._DEBUG:
            # consistent vertex set
            assert self._vertices.keys() == self._out_edges.keys() and self._vertices.keys() == self._in_edges.keys()

            # consistent edge set: (start,end,attr) in self._out_edges iff (end,start,attr) in self._in_edges
            for start, end_attr in self._out_edges.items():
                for end, attributes in end_attr.items():
                    # we check identity rather than equality due to an optimization (if both are identical, we only have to update once)
                    assert id(attributes) == id(self._in_edges.get(end, {}).get(start, {}))
            
            for end, start_attr in self._in_edges.items():
                for start, attributes in start_attr.items():
                    assert id(attributes) == id(self._out_edges.get(start, {}).get(end, {}))

    def _add_vertex(self, vertex: K, attributes: dict[str, Any] | None = None) -> bool:
        self._vertices[vertex] = {
            **self.default_vertex_attributes,
            **(attributes or {})
        }
        self._out_edges[vertex] = {}
        self._in_edges[vertex] = {}


class _FrozenWeightedDigraph[K: Hashable]:
    __slots__ = ("_vertices", "_out_edges", "_in_edges")

    def __init__(self, G: WeightedDigraph[K]):
        self._vertices: dict[K, Vertex[K]] = {
            vertex.id: Vertex(vertex.id, MappingProxyType(dict(vertex.attributes)))
            for vertex in G.vertices()
        }

        self._out_edges: dict[K, dict[K, Edge[K]]] = {vertex_id: {} for vertex_id in self._vertices.keys()}
        self._in_edges: dict[K, dict[K, Edge[K]]] = {vertex_id: {} for vertex_id in self._vertices.keys()}
        for edge in G.edges():
            edge_copy = Edge(self._vertices[edge.start.id], self._vertices[edge.end.id], MappingProxyType(dict(edge.attributes)))
            self._out_edges[edge.start.id][edge.end.id] = edge_copy
            self._in_edges[edge.end.id][edge.start.id] = edge_copy


    def contains_vertex(self, vertex: K) -> bool:
        return vertex in self._vertices
    
    def contains_edge(self, start: K, end: K) -> bool:
        return end in self._out_edges.get(start, ())

    def get_vertex(self, vertex: K) -> Vertex[K] | None:
        return self._vertices.get(vertex)

    def vertices(self) -> Iterable[Vertex[K]]:
        return (vertex for vertex in self._vertices.values())
    
    def get_edge(self, start: K, end: K) -> Edge[K] | None:
        from_start = self._out_edges.get(start)
        if from_start is None:
            raise ValueError(f"start vertex {start} does not exist")
        
        edge = from_start.get(end)
        if edge is None and not self.contains_vertex(end):
            raise ValueError(f"end vertex {end} does not exist")
        
        return edge

    def edges(self) -> Iterable[Edge[K]]:
        for end_edge in self._out_edges.values():
            for edge in end_edge.values():
                yield edge

    def out_edges(self, start: K) -> Iterable[Edge[K]]:
        from_start = self._out_edges.get(start)
        if from_start is None:
            raise ValueError(f"start vertex {start} does not exist")
        for edge in from_start.values():
            yield edge
    
    def in_edges(self, end: K) -> Iterable[Edge[K]]:
        from_end = self._in_edges.get(end)
        if from_end is None:
            raise ValueError(f"end vertex {end} does not exist")
        for edge in from_end.values():
            yield edge

    def out_degree(self, start: K) -> int:
        from_start = self._out_edges.get(start)
        if from_start is None:
            raise ValueError(f"start vertex {start} does not exist")
        return len(from_start)

    def in_degree(self, end: K) -> int:
        from_end = self._in_edges.get(end)
        if from_end is None:
            raise ValueError(f"end vertex {end} does not exist")
        return len(from_end)
