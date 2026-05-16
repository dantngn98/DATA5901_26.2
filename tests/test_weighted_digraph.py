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


def test_add_existing_edge_updates():
    g = WeightedDigraph(
        default_edge_attributes={
            "weight": 1,
            "capacity": 10,
        }
    )

    g.add_vertex("A")
    g.add_vertex("B")

    g.add_edge("A", "B", cost=5)

    added = g.add_edge(
        "A",
        "B",
        allow_update=True,
        cost=99,
    )

    assert added is False

    edge = g.get_edge("A", "B")

    # existing attributes preserved unless explicitly updated
    assert dict(edge.attributes) == {
        "weight": 1,
        "capacity": 10,
        "cost": 99,
    }


def test_add_edge_update_preserves_default_attributes():
    g = WeightedDigraph(
        default_edge_attributes={
            "weight": 1,
            "status": "active",
        }
    )

    g.add_vertex("A")
    g.add_vertex("B")

    g.add_edge("A", "B")

    g.add_edge(
        "A",
        "B",
        allow_update=True,
        cost=5,
    )

    edge = g.get_edge("A", "B")

    assert edge.attributes["weight"] == 1
    assert edge.attributes["status"] == "active"
    assert edge.attributes["cost"] == 5


def test_add_edge_update_modifies_existing_attributes():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")

    g.add_edge("A", "B", weight=1, cost=5)

    g.add_edge(
        "A",
        "B",
        allow_update=True,
        weight=99,
    )

    edge = g.get_edge("A", "B")

    assert edge.attributes["weight"] == 99
    assert edge.attributes["cost"] == 5


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

    g.add_vertex("A")
    g.add_vertex("B")

    with pytest.raises(ValueError):
        g.remove_edge("A", "B")


def test_remove_missing_edge_allowed():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")

    removed = g.remove_edge("A", "B", allow_not_exists=True)

    assert removed is False


def test_remove_edge_missing_vertices_raises():
    g = WeightedDigraph()

    with pytest.raises(ValueError):
        g.remove_edge("A", "B", allow_not_exists=True)


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
# copy
# =========================================================

def test_copy_creates_distinct_graph():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B", weight=5)

    c = g.copy()

    assert c is not g
    assert isinstance(c, WeightedDigraph)

    assert c.contains_vertex("A")
    assert c.contains_edge("A", "B")


def test_copy_preserves_attributes():
    g = WeightedDigraph(
        default_vertex_attributes={"color": "red"},
        default_edge_attributes={"weight": 1},
    )

    g.add_vertex("A", size=10)
    g.add_vertex("B")

    g.add_edge("A", "B", cost=5)

    c = g.copy()

    v = c.get_vertex("A")
    e = c.get_edge("A", "B")

    assert v.attributes["color"] == "red"
    assert v.attributes["size"] == 10

    assert e.attributes["weight"] == 1
    assert e.attributes["cost"] == 5


def test_copy_is_independent_vertex_attributes():
    g = WeightedDigraph()

    g.add_vertex("A", color="red")

    c = g.copy()

    g.set_vertex_attributes("A", color="blue")

    assert c.get_vertex("A").attributes["color"] == "red"
    assert g.get_vertex("A").attributes["color"] == "blue"


def test_copy_is_independent_edge_attributes():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")

    g.add_edge("A", "B", weight=1)

    c = g.copy()

    g.set_edge_attributes("A", "B", weight=99)

    assert c.get_edge("A", "B").attributes["weight"] == 1
    assert g.get_edge("A", "B").attributes["weight"] == 99


def test_copy_is_independent_structure():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")

    c = g.copy()

    g.add_vertex("C")
    g.add_edge("A", "B")

    assert not c.contains_vertex("C")
    assert not c.contains_edge("A", "B")

# =========================================================
# reverse
# =========================================================

def test_reverse_non_inplace_structure():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")

    g.add_edge("A", "B", weight=1)
    g.add_edge("B", "A", weight=2)

    r = g.reverse(inplace=False)

    # original unchanged
    assert g.contains_edge("A", "B")
    assert g.contains_edge("B", "A")

    # reversed edges
    assert r.contains_edge("B", "A")
    assert r.contains_edge("A", "B")

    # attributes preserved but direction flipped
    e1 = r.get_edge("B", "A")
    e2 = r.get_edge("A", "B")

    assert e1.attributes["weight"] == 1
    assert e2.attributes["weight"] == 2

def test_reverse_creates_new_graph_object():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B")

    r = g.reverse(inplace=False)

    assert r is not g
    assert isinstance(r, WeightedDigraph)

def test_reverse_inplace_mutates_graph():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B", weight=10)

    original_id = id(g)

    r = g.reverse(inplace=True)

    # same object returned
    assert r is g
    assert id(g) == original_id

    # direction flipped
    assert g.contains_edge("B", "A")
    assert not g.contains_edge("A", "B")

    e = g.get_edge("B", "A")
    assert e.attributes["weight"] == 10

def test_reverse_preserves_vertices():
    g = WeightedDigraph()

    g.add_vertex("A", color="red")
    g.add_vertex("B", color="blue")

    r = g.reverse()

    assert r.contains_vertex("A")
    assert r.contains_vertex("B")

    assert r.get_vertex("A").attributes["color"] == "red"
    assert r.get_vertex("B").attributes["color"] == "blue"

def test_reverse_degree_swap_property():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")
    g.add_vertex("C")

    g.add_edge("A", "B")
    g.add_edge("A", "C")
    g.add_edge("B", "C")

    r = g.reverse()

    # out-degree becomes in-degree
    assert r.out_degree("A") == g.in_degree("A")
    assert r.in_degree("A") == g.out_degree("A")

    assert r.out_degree("C") == g.in_degree("C")
    assert r.in_degree("C") == g.out_degree("C")

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
        edges={
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
        edges={},
    )

    assert g.contains_vertex("A")
    assert g.contains_vertex("B")

    assert list(g.edges()) == []


# =========================================================
# freeze / frozen graph
# =========================================================

def test_freeze_returns_frozen_graph():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B", weight=5)

    frozen = g.freeze()

    assert frozen is not g

    assert frozen.contains_vertex("A")
    assert frozen.contains_vertex("B")

    assert frozen.contains_edge("A", "B")


def test_freeze_preserves_vertex_attributes():
    g = WeightedDigraph()

    g.add_vertex("A", color="red", size=10)

    frozen = g.freeze()

    v = frozen.get_vertex("A")

    assert v.attributes["color"] == "red"
    assert v.attributes["size"] == 10


def test_freeze_preserves_edge_attributes():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")

    g.add_edge("A", "B", weight=7, cost=99)

    frozen = g.freeze()

    e = frozen.get_edge("A", "B")

    assert e.attributes["weight"] == 7
    assert e.attributes["cost"] == 99


def test_freeze_vertex_attributes_are_immutable():
    g = WeightedDigraph()

    g.add_vertex("A", value=1)

    frozen = g.freeze()

    v = frozen.get_vertex("A")

    with pytest.raises(TypeError):
        v.attributes["value"] = 2


def test_freeze_edge_attributes_are_immutable():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")

    g.add_edge("A", "B", weight=1)

    frozen = g.freeze()

    e = frozen.get_edge("A", "B")

    with pytest.raises(TypeError):
        e.attributes["weight"] = 2


def test_freeze_is_snapshot_not_live_view_vertex_attributes():
    g = WeightedDigraph()

    g.add_vertex("A", color="red")

    frozen = g.freeze()

    # mutate original graph after freezing
    g.set_vertex_attributes("A", color="blue")

    frozen_vertex = frozen.get_vertex("A")
    live_vertex = g.get_vertex("A")

    assert frozen_vertex.attributes["color"] == "red"
    assert live_vertex.attributes["color"] == "blue"


def test_freeze_is_snapshot_not_live_view_edge_attributes():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")

    g.add_edge("A", "B", weight=1)

    frozen = g.freeze()

    # mutate original graph after freezing
    g.set_edge_attributes("A", "B", weight=99)

    frozen_edge = frozen.get_edge("A", "B")
    live_edge = g.get_edge("A", "B")

    assert frozen_edge.attributes["weight"] == 1
    assert live_edge.attributes["weight"] == 99


def test_freeze_not_affected_by_new_vertices():
    g = WeightedDigraph()

    g.add_vertex("A")

    frozen = g.freeze()

    g.add_vertex("B")

    assert frozen.contains_vertex("A")
    assert not frozen.contains_vertex("B")


def test_freeze_not_affected_by_new_edges():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")
    g.add_vertex("C")

    g.add_edge("A", "B")

    frozen = g.freeze()

    g.add_edge("B", "C")

    assert frozen.contains_edge("A", "B")
    assert not frozen.contains_edge("B", "C")


def test_freeze_not_affected_by_removals():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B")

    frozen = g.freeze()

    g.remove_edge("A", "B")
    g.remove_vertex("B")

    assert frozen.contains_vertex("B")
    assert frozen.contains_edge("A", "B")


def test_frozen_vertices_iterator():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")

    frozen = g.freeze()

    ids = {v.id for v in frozen.vertices()}

    assert ids == {"A", "B"}


def test_frozen_edges_iterator():
    g = WeightedDigraph()

    for v in ["A", "B", "C"]:
        g.add_vertex(v)

    g.add_edge("A", "B")
    g.add_edge("B", "C")

    frozen = g.freeze()

    edges = {
        (e.start.id, e.end.id)
        for e in frozen.edges()
    }

    assert edges == {
        ("A", "B"),
        ("B", "C"),
    }


def test_frozen_out_degree():
    g = WeightedDigraph()

    for v in ["A", "B", "C"]:
        g.add_vertex(v)

    g.add_edge("A", "B")
    g.add_edge("A", "C")

    frozen = g.freeze()

    assert frozen.out_degree("A") == 2
    assert frozen.out_degree("B") == 0


def test_frozen_in_degree():
    g = WeightedDigraph()

    for v in ["A", "B", "C"]:
        g.add_vertex(v)

    g.add_edge("A", "C")
    g.add_edge("B", "C")

    frozen = g.freeze()

    assert frozen.in_degree("C") == 2
    assert frozen.in_degree("A") == 0


def test_frozen_edge_objects_shared_between_in_and_out_maps():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")

    g.add_edge("A", "B", weight=5)

    frozen = g.freeze()

    edge1 = frozen._out_edges["A"]["B"]
    edge2 = frozen._in_edges["B"]["A"]

    # same exact Edge object
    assert edge1 is edge2


def test_frozen_vertex_objects_are_reused_by_edges():
    g = WeightedDigraph()

    g.add_vertex("A")
    g.add_vertex("B")

    g.add_edge("A", "B")

    frozen = g.freeze()

    edge = frozen.get_edge("A", "B")

    assert edge.start is frozen.get_vertex("A")
    assert edge.end is frozen.get_vertex("B")


# =========================================================
# main
# =========================================================

def run_all_tests():
    # construction
    test_empty_graph()
    test_default_attributes()

    # vertex operations
    test_add_vertex()
    test_add_existing_vertex_raises()
    test_add_existing_vertex_allowed()
    test_remove_vertex()
    test_remove_missing_vertex_raises()
    test_remove_missing_vertex_allowed()
    test_set_vertex_attributes()
    test_set_missing_vertex_attributes_raises()

    # edge operations
    test_add_edge()
    test_add_edge_requires_vertices()
    test_add_edge_auto_creates_vertices()
    test_add_existing_edge_raises()
    test_add_existing_edge_updates()
    test_add_edge_update_preserves_default_attributes()
    test_add_edge_update_modifies_existing_attributes()
    test_set_edge_attributes()
    test_set_missing_edge_attributes_raises()
    test_remove_edge()
    test_remove_missing_edge_raises()
    test_remove_edge_missing_vertices_raises()
    test_remove_missing_edge_allowed()

    # graph consistency
    test_removing_vertex_removes_incident_edges()
    test_out_degree()
    test_in_degree()

    # iterators
    test_vertices_iterator()
    test_edges_iterator()
    test_out_edges_iterator()
    test_in_edges_iterator()

    # immutability
    test_vertex_attributes_are_immutable()
    test_edge_attributes_are_immutable()

    # copy
    test_copy_creates_distinct_graph()
    test_copy_preserves_attributes()
    test_copy_is_independent_vertex_attributes()
    test_copy_is_independent_edge_attributes()
    test_copy_is_independent_structure()

    # reverse
    test_reverse_non_inplace_structure()
    test_reverse_creates_new_graph_object()
    test_reverse_inplace_mutates_graph()
    test_reverse_preserves_vertices()
    test_reverse_degree_swap_property()

    # from_dict
    test_from_dict()
    test_from_dict_supports_isolated_vertices()

    # freeze / frozen graph
    test_freeze_returns_frozen_graph()
    test_freeze_preserves_vertex_attributes()
    test_freeze_preserves_edge_attributes()
    test_freeze_vertex_attributes_are_immutable()
    test_freeze_edge_attributes_are_immutable()
    test_freeze_is_snapshot_not_live_view_vertex_attributes()
    test_freeze_is_snapshot_not_live_view_edge_attributes()
    test_freeze_not_affected_by_new_vertices()
    test_freeze_not_affected_by_new_edges()
    test_freeze_not_affected_by_removals()
    test_frozen_vertices_iterator()
    test_frozen_edges_iterator()
    test_frozen_out_degree()
    test_frozen_in_degree()
    test_frozen_edge_objects_shared_between_in_and_out_maps()
    test_frozen_vertex_objects_are_reused_by_edges()


if __name__ == "__main__":
    run_all_tests()
    print("All tests passed.")