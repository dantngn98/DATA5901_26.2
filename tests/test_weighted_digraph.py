# standard
import pytest
from types import MappingProxyType

# local
from src.cost_model.weighted_digraph import WeightedDigraph, Vertex, Edge


# =========================================================
# construction
# =========================================================

def test_empty_graph():
    g = WeightedDigraph()

    assert list(g.vertices()) == []
    assert list(g.edges()) == []


def test_default_attributes():
    g = WeightedDigraph(
        default_vertex_attributes={"color": "red"},
        default_edge_attributes={"weight": 1},
    )

    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B")

    v = g.get_vertex("A")
    e = g.get_edge("A", "B")

    assert v.attributes["color"] == "red"
    assert e.attributes["weight"] == 1


# =========================================================
# vertex operations
# =========================================================

def test_add_vertex():
    g = WeightedDigraph()

    added = g.add_vertex("A", color="blue")

    assert added is True
    assert g.contains_vertex("A")

    v = g.get_vertex("A")

    assert isinstance(v, Vertex)
    assert v.id == "A"
    assert v.attributes["color"] == "blue"


def test_add_existing_vertex_raises():
    g = WeightedDigraph()

    g.add_vertex("A")

    with pytest.raises(ValueError):
        g.add_vertex("A")


def test_add_existing_vertex_allowed():
    g = WeightedDigraph()

    g.add_vertex("A")

    added = g.add_vertex("A", allow_exists=True)

    assert added is False


def test_remove_vertex():
    g = WeightedDigraph()

    g.add_vertex("A")

    removed = g.remove_vertex("A")

    assert removed is True
    assert not g.contains_vertex("A")


def test_remove_missing_vertex_raises():
    g = WeightedDigraph()

    with pytest.raises(ValueError):
        g.remove_vertex("A")


def test_remove_missing_vertex_allowed():
    g = WeightedDigraph()

    removed = g.remove_vertex("A", allow_not_exists=True)

    assert removed is False


def test_set_vertex_attributes():
    g = WeightedDigraph()

    g.add_vertex("A", color="red")

    g.set_vertex_attributes("A", size=10)

    v = g.get_vertex("A")

    assert v.attributes["color"] == "red"
    assert v.attributes["size"] == 10


def test_set_missing_vertex_attributes_raises():
    g = WeightedDigraph()

    with pytest.raises(ValueError):
        g.set_vertex_attributes("A", x=1)


# =========================================================
# edge operations
# =========================================================

def test_add_edge():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")

    added = g.add_edge("A", "B", weight=7)

    assert added is True
    assert g.contains_edge("A", "B")

    edge = g.get_edge("A", "B")

    assert isinstance(edge, Edge)

    assert edge.start.id == "A"
    assert edge.end.id == "B"

    assert edge.attributes["weight"] == 7


def test_add_edge_requires_vertices():
    g = WeightedDigraph()

    with pytest.raises(ValueError):
        g.add_edge("A", "B")


def test_add_edge_auto_creates_vertices():
    g = WeightedDigraph()

    g.add_edge(
        "A",
        "B",
        allow_create_vertices=True,
        weight=5,
    )

    assert g.contains_vertex("A")
    assert g.contains_vertex("B")
    assert g.contains_edge("A", "B")


def test_add_existing_edge_raises():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")

    g.add_edge("A", "B")

    with pytest.raises(ValueError):
        g.add_edge("A", "B")


def test_add_existing_edge_overwrites():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")

    g.add_edge("A", "B", weight=1)

    added = g.add_edge(
        "A",
        "B",
        allow_overwrite=True,
        cost=99,
    )

    assert added is False

    edge = g.get_edge("A", "B")

    # overwrite replaces attribute dict entirely
    assert dict(edge.attributes) == {"cost": 99}


def test_set_edge_attributes():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")

    g.add_edge("A", "B", weight=1)

    g.set_edge_attributes("A", "B", cost=10)

    edge = g.get_edge("A", "B")

    assert edge.attributes["weight"] == 1
    assert edge.attributes["cost"] == 10


def test_set_missing_edge_attributes_raises():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")

    with pytest.raises(ValueError):
        g.set_edge_attributes("A", "B", x=1)


def test_remove_edge():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")

    g.add_edge("A", "B")

    removed = g.remove_edge("A", "B")

    assert removed is True
    assert not g.contains_edge("A", "B")


def test_remove_missing_edge_raises():
    g = WeightedDigraph()

    with pytest.raises(ValueError):
        g.remove_edge("A", "B")


def test_remove_missing_edge_allowed():
    g = WeightedDigraph()

    removed = g.remove_edge("A", "B", allow_not_exists=True)

    assert removed is False


# =========================================================
# graph consistency
# =========================================================

def test_removing_vertex_removes_incident_edges():
    g = WeightedDigraph()

    for v in ["A", "B", "C"]:
        g.add_vertex(v)

    g.add_edge("A", "B")
    g.add_edge("C", "B")

    g.remove_vertex("B")

    assert not g.contains_edge("A", "B")
    assert not g.contains_edge("C", "B")


def test_out_degree():
    g = WeightedDigraph()

    for v in ["A", "B", "C"]:
        g.add_vertex(v)

    g.add_edge("A", "B")
    g.add_edge("A", "C")

    assert g.out_degree("A") == 2
    assert g.out_degree("B") == 0
    assert g.out_degree("C") == 0


def test_in_degree():
    g = WeightedDigraph()

    for v in ["A", "B", "C"]:
        g.add_vertex(v)

    g.add_edge("A", "C")
    g.add_edge("B", "C")

    assert g.in_degree("A") == 0
    assert g.in_degree("B") == 0
    assert g.in_degree("C") == 2


# =========================================================
# iterators
# =========================================================

def test_vertices_iterator():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")

    vertices = list(g.vertices())

    ids = {v.id for v in vertices}

    assert ids == {"A", "B"}


def test_edges_iterator():
    g = WeightedDigraph()

    for v in ["A", "B", "C"]:
        g.add_vertex(v)

    g.add_edge("A", "B")
    g.add_edge("B", "C")

    edges = {
        (e.start.id, e.end.id)
        for e in g.edges()
    }

    assert edges == {
        ("A", "B"),
        ("B", "C"),
    }


def test_out_edges_iterator():
    g = WeightedDigraph()

    for v in ["A", "B", "C"]:
        g.add_vertex(v)

    g.add_edge("A", "B")
    g.add_edge("A", "C")

    edges = {
        (e.start.id, e.end.id)
        for e in g.out_edges("A")
    }

    assert edges == {
        ("A", "B"),
        ("A", "C"),
    }


def test_in_edges_iterator():
    g = WeightedDigraph()

    for v in ["A", "B", "C"]:
        g.add_vertex(v)

    g.add_edge("A", "C")
    g.add_edge("B", "C")

    edges = {
        (e.start.id, e.end.id)
        for e in g.in_edges("C")
    }

    assert edges == {
        ("A", "C"),
        ("B", "C"),
    }


# =========================================================
# immutability
# =========================================================

def test_vertex_attributes_are_immutable():
    g = WeightedDigraph()

    g.add_vertex("A", value=1)

    vertex = g.get_vertex("A")

    assert isinstance(vertex.attributes, MappingProxyType)

    with pytest.raises(TypeError):
        vertex.attributes["value"] = 2


def test_edge_attributes_are_immutable():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")

    g.add_edge("A", "B", weight=1)

    edge = g.get_edge("A", "B")

    assert isinstance(edge.attributes, MappingProxyType)

    with pytest.raises(TypeError):
        edge.attributes["weight"] = 2


# =========================================================
# from_dict
# =========================================================

def test_from_dict():
    g = WeightedDigraph.from_dict(
        vertex_attributes={
            "A": {"color": "red"},
            "B": {"color": "blue"},
            "C": {},
        },
        start_end_attributes={
            "A": {
                "B": {"weight": 5},
            },
            "B": {
                "C": {"weight": 7},
            },
        },
    )

    assert g.contains_vertex("A")
    assert g.contains_vertex("B")
    assert g.contains_vertex("C")

    assert g.contains_edge("A", "B")
    assert g.contains_edge("B", "C")

    assert g.get_vertex("A").attributes["color"] == "red"
    assert g.get_edge("A", "B").attributes["weight"] == 5


def test_from_dict_supports_isolated_vertices():
    g = WeightedDigraph.from_dict(
        vertex_attributes={
            "A": {},
            "B": {},
        },
        start_end_attributes={},
    )

    assert g.contains_vertex("A")
    assert g.contains_vertex("B")

    assert list(g.edges()) == []


# =========================================================
# main
# =========================================================

if __name__ == "__main__":
    test_empty_graph()
    test_default_attributes()

    test_add_vertex()
    test_add_existing_vertex_raises()
    test_add_existing_vertex_allowed()
    test_remove_vertex()
    test_remove_missing_vertex_raises()
    test_remove_missing_vertex_allowed()
    test_set_vertex_attributes()
    test_set_missing_vertex_attributes_raises()

    test_add_edge()
    test_add_edge_requires_vertices()
    test_add_edge_auto_creates_vertices()
    test_add_existing_edge_raises()
    test_add_existing_edge_overwrites()
    test_set_edge_attributes()
    test_set_missing_edge_attributes_raises()
    test_remove_edge()
    test_remove_missing_edge_raises()
    test_remove_missing_edge_allowed()

    test_removing_vertex_removes_incident_edges()
    test_out_degree()
    test_in_degree()

    test_vertices_iterator()
    test_edges_iterator()
    test_out_edges_iterator()
    test_in_edges_iterator()

    test_vertex_attributes_are_immutable()
    test_edge_attributes_are_immutable()

    test_from_dict()
    test_from_dict_supports_isolated_vertices()

    print("All tests passed.")