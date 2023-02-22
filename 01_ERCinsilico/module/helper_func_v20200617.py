####
#### My helper functions (Python)
#### 2020.6.17
####

# Flatten function ----------------------------------------------------------------- #
# From https://www.lifewithpython.com/2014/01/python-flatten-nested-lists.html ----- #
def flatten_with_any_depth(nested_list):
    flat_list = []
    fringe = [nested_list]
    while len(fringe) > 0:
        node = fringe.pop(0)
        if isinstance(node, list):
            fringe = node + fringe
        else:
            flat_list.append(node)
    return flat_list
# ---------------------------------------------------------------------------------- #
