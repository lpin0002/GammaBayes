import warnings
import numpy as np
from gammabayes.priors.core import DiscreteLogPrior

class MTreeNode:
    def __init__(self, value, id=None, parent_node=None, prior=None):
        self.value = value
        self.children = []
        self.id = id  # Unique identifier for each node
        self.parent_node = parent_node
        self.prior = prior

        if id is None:
            if prior is None:
                self.id = None
            else:
                self.id = prior.name
            
        


    def add_child(self, child):
        self.children.append(child)
        return child

    def __repr__(self):
        return f"TreeNode(value={self.value}, id={self.id})"
    

    def deep_copy(self):
        copied_node = MTreeNode(value    = self.value, 
                                id  = self.id, 
                                parent_node = self.parent_node, 
                                prior = self.prior)
        copied_node.children = [child.deep_copy() for child in self.children]
        return copied_node




class MTree:
    def __init__(self, root=None):
        if root is None:
            self.root = MTreeNode(1.0, id="root")  # Initialize the root node
            self.nodes = {self.root.id: self.root}
        else:
            self.root = root
            self.nodes = {node.id: node for node in self.collect_nodes(root)}
        self.refresh_leaves()

    def collect_nodes(self, node):
        yield node
        for child in node.children:
            yield from self.collect_nodes(child)

    def refresh_leaves(self):
        self.leaves = []
        self.leaf_values = {}
        self.collect_leaves(self.root)

    def collect_leaves(self, node):
        if not node.children:
            self.leaves.append(node)
            self.leaf_values[node.id] = self.compute_weight(node)
        for child in node.children:
            self.collect_leaves(child)

    def generate_id(self, parent=None, prior=None):
        if parent is not None:
            idprefix = f"{parent.id}_" 
        elif prior is not None:
            idprefix = f"{prior.name}_"
        else:
            idprefix = f""

        id = 1
        while f"{idprefix}auto_id{id}" in self.nodes:
            id += 1
        return f"{idprefix}auto_id{id}"

    def add(self, value, parent, prior:DiscreteLogPrior = None, id=None):
        if id and id in self.nodes:
            raise ValueError(f"Node ID '{id}' already exists.")
        new_node_id = id if id is not None else self.generate_id(parent)
        new_node = MTreeNode(value, id=new_node_id, parent_node=parent, prior=prior)
        parent.add_child(new_node)
        self.nodes[new_node_id] = new_node
        self.refresh_leaves()
        return new_node

    def compute_weight(self, node):
        product = node.value
        while node.parent_node:
            node = node.parent_node
            product *= node.value
        return product

    def create_tree(self, layout, values, parent=None, index=0):
        if isinstance(layout, MTree):
            # If the layout is another Tree, deep copy it
            self.root = layout.root.deep_copy()
            self.refresh_leaves()
            return
        if parent is None:
            parent = self.root

        for item in layout:
            if isinstance(item, list):
                # Recursive case: item is a sublist representing children of the current node
                new_node = self.add(values[index], parent)
                index += 1
                index = self.create_tree(item, values, new_node, index)
            elif isinstance(item, DiscreteLogPrior):
                # Base case: item is an ID
                new_node = self.add(values[index], parent, prior=item, id=item.name)
                index += 1

            else:
                # Base case: item is an ID
                new_node = self.add(values[index], parent, id=item)
                index += 1
            

        return index

    def _tree_str(self, node=None, level=0, prefix="", precision='3g', prev_str="", print_ids=False):
        if node is None:
            node = self.root
        connector = "|__ " if level > 0 else ""
        output_string = prev_str

        if print_ids:
            print_val = str(node.id)
        else:
            print_val = f"{node.value:.{precision}}"

        if level==0:
            output_string += f"{prefix}{connector}{print_val} __\n"
        else:
            output_string += f"{prefix}{connector}{print_val}\n"

        if node.children:
            child_prefix = prefix + ("|   " if level > 0 else "    ")
            for child in node.children:
                output_string = self._tree_str(child, level + 1, child_prefix, precision, prev_str=output_string, print_ids=print_ids)

        return output_string
        

    def print_tree(self, node=None, level=0, prefix="", precision='3g', print_ids=False):
        print(self._tree_str(node=node, level=level, prefix=prefix, precision=precision, print_ids=print_ids))


    def __repr__(self):
        output_string = "___ID Structure___:\n"
        output_string += self._tree_str(print_ids=True)
        output_string +='\n'
        output_string +='\n'
        output_string += "\n___Leaf Values___:\n"
        for leaf in self.leaf_values.items():
            output_string += str(leaf)+'\n'

        output_string += "\n__Nodes__:\n"
        for node in self.nodes.values():
            output_string += str(node)+'\n'


        return output_string

    def copy(self):
        return MTree(self.root.deep_copy())


    # Recursive function to delete all children nodes
    def _remove_children(self, node):

        # iterate over a copy of the list, as otherwise when iterating
            # the list itself could be changing
        for child in node.children[:]:  
            self._remove_children(child)  # recursively remove children
            del self.nodes[child.id]  # remove child from nodes dictionary

        # Clear the children list after removal
        node.children.clear()  


    def delete_node(self, node_id):
        if node_id not in self.nodes:
            warnings.warn(f"No node with ID '{node_id}' exists.")
            return False

        # Get the node to be deleted from the nodes dictionary
        node_to_delete = self.nodes[node_id]

        # If the node is not the root, remove it from its parent's children list
        if node_to_delete.parent_node:
            node_to_delete.parent_node.children = [
                child for child in node_to_delete.parent_node.children if child.id != node_id
            ]

        # Recursively remove this node and all its children from the tree
        self._remove_children(node_to_delete)

        # Finally, remove the node itself from the nodes dictionary
        del self.nodes[node_id]

        # Refresh the leaves and leaf values since the structure of the tree has changed
        self.refresh_leaves()
        return True


    def overwrite(self, values):
        # Presumes nicely behaving inputs

        
        # Use a queue to handle the breadth-first traversal
        queue = list(self.root.children)  # Start with children of the root
        value_index = 0  # Start from the first provided value

        while queue and value_index < len(values):
            current_node = queue.pop(0)
            # Update current node's value
            current_node.value = values[value_index]
            # Calculate the complementary value and update it
            
            value_index += 1  # Move to the next value

            # Add children of the current nodes to the queue
            queue.extend(current_node.children)

        # After updating, recompute leaf values
        self.refresh_leaves()


