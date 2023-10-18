import numpy as np


# ======================================================================
def dict2list(dictionary):
    array = []
    relabeling = dict()

    key_adj = 0
    for key, value in dictionary.items():
        array.append(value)
        relabeling[key] = key_adj
        key_adj += 1

    array = array

    return array, relabeling


def dict2arr(dictionary):
    array, relabeling = dict2list(dictionary)
    array = np.array(array)

    return array, relabeling


def list2dict(list_, index_mapping):
    dictionary = dict()
    for list_index, dict_index in index_mapping.items():
        dictionary[dict_index] = list_[list_index]

    return dictionary


def unpack_mappings(*mappings):
    raise NotImplementedError("Currently not implemented!")
    # return [dict2list(mapping) for mapping in mappings]


def cutkey(dictionary, *keys):
    dictionary_adj = dictionary.copy()

    for key in keys:
        if dictionary.get(key, None) is not None:
            del dictionary_adj[key]

    return dictionary_adj


def inverse_map(dictionary):
    inverse_mapping = dict()
    for key, value in dictionary.items():
        assert value not in inverse_mapping
        inverse_mapping[value] = key

    return inverse_mapping
