# -*- coding: utf-8 -*-
"""
@author: Mitchell Edmunds
@title: Prices Classification Tree

TODO: 
    * Write aggregate up hierarchy function, choosing the function
    * Fix all the default args where there is an empty list or dict.
"""

from ooh.utils import precon
import pandas as pd

# print(__name__)


def levels_from_paths(paths, separator, root_level_same_as_children):
    """Pass a Series of hierarchical paths and return the levels."""
    levels = paths.str.split(separator, expand=True)
    #levels.columns = ['L'+str(level) for level in levels.columns]

    if isinstance(levels.index, pd.core.indexes.range.RangeIndex):
        levels.index = paths

    if root_level_same_as_children:
        levels.columns += 1

    return levels


def depths_from_levels(levels, root_level_same_as_children):
    "Get depths of each node by number of non-null values in the levels."
    if root_level_same_as_children:
        return levels.notnull().sum(axis=1)
    else:
        return levels.notnull().sum(axis=1) - 1


def tree_from_labels(
        labels, names, ids, paths, sep,
        root_level_same_as_children
        ):
    """Creates a tree structure from a labels DataFrame."""
    # Separate the root and tree
    root = labels.iloc[0]
    tree_df = labels.iloc[1:]

    # Initialise the tree at the root
    ooh_tree = OohTree(root, name_col=names, id_col=ids)
    # Then build tree
    ooh_tree.build_tree(tree_df, paths, sep, root_level_same_as_children)

    return ooh_tree


class OohTree:
    """Creates a tree class using the ooh classification system."""

    def __init__(self, root, name_col, id_col, level=0, parent=None):
        self.name = root[name_col]
        self.id_ = root[id_col]
        self.level = level  # change to depth
        self.children = []
        self.parent = parent

        self.name_col = name_col
        self.id_col = id_col

    def identify_children(self, df, levels, depths):
        """Returns the DataFrame sliced by the children of the current node.

           The children are those with a matching path to the parent and
           are just one level down in the depth of the tree.
        """
        if self.level != depths.max():
            # Which paths in the levels df match with the current node path
            if self.level == 0:
                same_paths = True
            else:
                same_paths = ((levels.loc[:, :self.level] ==
                               levels.loc[self.id_, :self.level]).all(axis=1))

            # Which nodes are a level down from the current node
            one_level_down = (depths == self.level + 1)

            # The children are those with a matching path to the parent and
            # are just one level down in the depth of the tree
            children = df[same_paths & one_level_down]

            return children

    def create_children(self, df, levels, depths):
        """Identify the children, iterate through each child and initialise
           a new tree with the child's attributes.
        """
        children = self.identify_children(df, levels, depths)

        for id_, row in children.iterrows():
            self.children.append(OohTree(root=row,
                                         name_col=self.name_col,
                                         id_col=self.id_col,
                                         level=self.level + 1,
                                         parent=self))

    def render_tree(self, tree_strings=None, mapper=None):
        """Create a list of strings displaying class structure."""
        if not tree_strings:
            tree_strings = []

        if mapper:
            code_to_print = mapper.get(self.id_)  # Take mapped string
        else:
            code_to_print = self.id_  # Use node id

        s = ('\t' * self.level) + code_to_print + '  ' + self.name
        tree_strings.append(s)

        for child in self.children:
            tree_strings = child.render_tree(tree_strings, mapper=mapper)

        return tree_strings

    def print_tree(self, mapper=None):
        """Print the class structure."""
        print('\n')
        tree_strings = self.render_tree(mapper=mapper)
        for s in tree_strings:
            print(s)

    def write_tree(self, filepath, mapper=None):
        """Writes the tree to the given file."""
        tree_strings = self.render_tree(mapper=mapper)
        with open(filepath, 'w') as f:
            for s in tree_strings:
                f.write(s + '\n')

    def recursive_build(self, df, levels, depths):
        if self.level < depths.max():
            self.create_children(df, levels, depths)

            for child in self.children:
                child.recursive_build(df, levels, depths)

    def build_tree(self, df, paths, separator,
                   root_level_same_as_children=False):
        """Build the tree from the classification levels."""
        levels = levels_from_paths(df[paths],
                                   separator,
                                   root_level_same_as_children)
        depths = depths_from_levels(levels, root_level_same_as_children)
        self.recursive_build(df, levels, depths)

    def get_leaves(self, leaves=None):
        """Return the id_ attribute of the nodes without children."""
        if not leaves:
            leaves = []

        if not self.children:
            leaves.append(self.id_)

        for child in self.children:
            leaves = child.get_leaves(leaves)

        return leaves

    def get_child_ids(self, child_ids={}):
        """Return a dictionary of parent-children key-value pairs. """
        child_ids[self.id_] = []

        for child in self.children:
            child.get_child_ids(child_ids)
            child_ids[self.id_].append(child.id_)

        return child_ids

    def aggregate_sum(self, indices, child_ids={}):
        """Each parent column is the sum of its childrens' values."""
        if self.level == 0:
            indices = indices.copy()
            child_ids = self.get_child_ids()

        for child in self.children:
            child.aggregate_sum(indices, child_ids)

        if self.children:
            indices.loc[:, self.id_] = indices.loc[:,
                                                   child_ids[self.id_]].sum(axis=1)

        return indices

    def child_shares(self, weights):
        """Each child column is the percentage share of its parents value."""
        if self.level == 0:
            weights = weights.copy()

        for child in self.children:
            child.child_shares(weights)

        if self.parent:
            self_weights = weights.loc[:, self.id_]
            parent_weights = weights.loc[:, self.parent.id_]
            weights.loc[:, self.id_] = self_weights / parent_weights
        else:
            # The top of the tree is divided by itself.
            weights.loc[:, self.id_] /= weights.loc[:, self.id_]

        return weights

    def weighted_aggregate(self, indices, shares, child_ids=None):
        """Given weight shares and indices, calculate the weighted aggregate."""
        if not child_ids:
            child_ids = {}

        if self.level == 0:
            indices = indices.copy()
            child_ids = self.get_child_ids()

        for child in self.children:
            indices = child.weighted_aggregate(indices, shares, child_ids)

        if self.children:
            shares_sub = shares.loc[:, child_ids[self.id_]]
            indices_sub = indices.loc[:, child_ids[self.id_]]
            aggregated = precon.aggregate(indices_sub, shares_sub)
            indices.loc[:, self.id_] = aggregated

        return indices

    def detach_nodes(self, nodes, detach=None):
        """Remove the nodes in the passed argument from the tree."""
        if self.level == 0:
            detach = []

        for child in self.children:
            if child.id_ in nodes:
                detach.append(child)

            child.detach_nodes(nodes, detach)

        for node in detach:
            if node in self.children:
                self.children.remove(node)
                print("\n" + "Deleted node: " + node.id_)

        return detach
        # Write code to readjust the levels in the new tree

    def attach_nodes(self, nodes, parent_id):
        """Add the nodes at the parent_id specified."""
        if self.children:
            for child in self.children:
                child.attach_nodes(nodes, parent_id)

        if self.id_ == parent_id:
            for node in nodes:
                self.children.append(node)
                print("\n" + "Added node: " + node.id_ + " at " + parent_id)

    def get_nodes_at_level(self, level, nodes_at_level=None):
        """ """
        if not nodes_at_level:
            nodes_at_level = []

        if self.level == level:
            nodes_at_level.append(self.id_)

        for child in self.children:
            nodes_at_level = child.get_nodes_at_level(level, nodes_at_level)

        return nodes_at_level

    def chain_ooh(self, indices):
        """If the indices exist, then chain them according to one of two 
           methods:
               Direct (fixed-base) with annual double-chainlink if the node
               is a parent.
               Indirect if the node is a leaf (childless).
        """
        if self.level == 0:
            indices = indices.copy()

        for child in self.children:
            child.chain_ooh(indices)

        if indices.loc[:, self.id_].sum() != 0:
            if self.children:
                indices.loc[:, self.id_] = precon.chain(indices.loc[:, self.id_],
                                                        double_link=True)
            elif not self.children:
                indices.loc[:, self.id_] = precon.chain(
                    indices.loc[:, self.id_])

        return indices

    def jan_adjustment(self, indices):
        """ """
        if self.level == 0:
            indices = indices.copy()

        for child in self.children:
            child.jan_adjustment(indices)

        if indices.loc[:, self.id_].sum() != 0:
            if not self.children:
                indices.loc[:, self.id_] = precon.jan_adjustment(
                    indices.loc[:, self.id_])

        return indices

# class OohTree(ClassTree):
#    def __init__(self, root, name_col, id_col, level=0, parent=None):
#        super().__init__(root, name_col, id_col, level=0, parent=None)
#        self.ooh = root['agg_num']
