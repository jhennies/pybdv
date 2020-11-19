
import numpy as np


def load_with_zero_padding(dataset, starts, ends, shape, verbose=False):

    starts_target = np.zeros((3,), dtype=int)
    starts_target[starts < 0] = -starts[starts < 0]
    starts_source = np.zeros((3,), dtype=int)
    starts_source[starts > 0] = starts[starts > 0]

    too_large = (np.array(dataset.shape) - ends) < 0
    ends_target = shape.copy()
    ends_target[too_large] = (np.array(dataset.shape) - starts)[too_large]
    ends_source = ends.copy()
    ends_source[too_large] = np.array(dataset.shape)[too_large]

    if verbose:
        print(f'shape = {shape}')
        print(f'dataset.shape = {dataset.shape}')
        print(f'starts = {starts}')
        print(f'ends = {ends}')
        print(f'starts_source = {starts_source}')
        print(f'starts_target = {starts_target}')
        print(f'ends_source = {ends_source}')
        print(f'ends_target = {ends_target}')

    slicing_source = np.s_[
        starts_source[0]: ends_source[0],
        starts_source[1]: ends_source[1],
        starts_source[2]: ends_source[2]
    ]
    slicing_target = np.s_[
        starts_target[0]: ends_target[0],
        starts_target[1]: ends_target[1],
        starts_target[2]: ends_target[2]
    ]

    vol = np.zeros(shape, dtype=dataset.dtype)
    vol[slicing_target] = dataset[slicing_source]

    return vol


def get_non_fully_contained_ids(volume):

    vol = volume.copy()
    vol[1:-1, 1:-1, 1:-1] = 0

    return np.unique(vol)[1:]


def relabel_with_skip_ids(map, skip_ids):
    """
    Relabels a volume and skips a list of ids
    The zero label is kept as it is and treated as background
    """

    labels = np.unique(map)
    labels = labels[np.nonzero(labels)].tolist()
    new_labels = []
    c = 0
    for _ in labels:
        c += 1
        while c in skip_ids:
            c += 1
        new_labels.append(c)

    assert len(labels) == len(new_labels)

    relabel_dict = dict(zip(labels, new_labels))
    relabel_dict[0] = 0

    # Reshape the map
    shp = map.shape
    map = np.reshape(map, np.product(shp))

    # Do the mapping
    map = np.array([relabel_dict[label] for label in map])

    # Shape back
    map = np.reshape(map, shp)

    return map
