from typing import List
from collections import namedtuple

from tqdm import tqdm


Relationship = namedtuple("Relationship", "parent,child")
Node = namedtuple("Node", "name,children,parents")


def create_hierarchy(scores, min_similarity):
    nodes_by_name = {}
    relationships = []

    for parent, child, score in tqdm(scores, desc="Grouping nodes"):
        if parent == child:
            continue
        if score >= min_similarity:
            relationships.append(Relationship(parent, child))

    for ship in relationships:
        parent = nodes_by_name.setdefault(
            ship.parent, Node(ship.parent, [], [])
        )
        child = nodes_by_name.setdefault(ship.child, Node(ship.child, [], []))

        parent.children.append(child)
        child.parents.append(parent)

    return list(nodes_by_name.values())


def carry_transitive_relations(hierarchy: List[Node]):
    # Go from least parents to most parents and carry children all the way
    # to the top of the hierarchy
    by_parent_count_asc = sorted(hierarchy, key=lambda n: len(n.parents))
    for node in tqdm(
        by_parent_count_asc, desc="Carrying transitive relations"
    ):
        for parent_node in node.parents:
            parent_node.children.extend(
                c
                for c in node.children
                if c.name != parent_node.name
                and not any(c.name == n.name for n in parent_node.children)
            )
    return by_parent_count_asc


def flatten_to_top(hierarchy: List[Node]):
    top_level = []
    lower_levels = []
    with_transitive_relations = carry_transitive_relations(hierarchy)
    by_child_count_desc = sorted(
        with_transitive_relations, key=lambda n: len(n.children), reverse=True
    )

    for node in tqdm(by_child_count_desc, desc="Getting top level"):
        if node.name in lower_levels:
            continue
        top_level.append(node)
        lower_levels.extend(n.name for n in node.children)
    return top_level
