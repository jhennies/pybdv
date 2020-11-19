
import os
import numpy as np
import json
from .util import open_file, get_scale_factors, get_key, HDF5_EXTENSIONS
from .converter import make_bdv
from .stitching_utils import load_with_zero_padding, get_non_fully_contained_ids, relabel_with_skip_ids
from shutil import rmtree
from warnings import warn


def _check_for_out_of_bounds(position, volume, full_shape, verbose=False):

    position = np.array(position)
    full_shape = np.array(full_shape)

    vol_shape = np.array(volume.shape)
    if position.min() < 0 or (position + vol_shape - full_shape).max() > 0:

        print(f'position = {position}')

        too_large = (position + vol_shape - full_shape) > 0
        source_ends = vol_shape
        source_ends[too_large] = (full_shape - position)[too_large]

        source_starts = np.zeros((3,), dtype=int)
        source_starts[position < 0] = -position[position < 0]
        position[position < 0] = 0

        if verbose:
            print(f'source_starts = {source_starts}')
            print(f'source_ends = {source_ends}')
            print(f'position = {position}')

        volume = volume[
            source_starts[0]: source_ends[0],
            source_starts[1]: source_ends[1],
            source_starts[2]: source_ends[2]
        ]

    return position, volume


def _check_shape_and_position_scaling(max_scale, position, volume,
                                      data_path, key,
                                      verbose=False):
    vol_shape = np.array(volume.shape)
    if ((position / max_scale) - (position / max_scale).astype(int)).max()\
            or ((vol_shape / max_scale) - (vol_shape / max_scale).astype(int)).max():

        # They don't scale properly:
        # So, we have to read a volume from the target data (largest scale) that does and covers the area of where the
        # volume belongs, which we here call target_vol with respective properties target_pos and target_shape

        if verbose:
            print('----------------------')
            print(f'max_scale = {max_scale}')
            print(f'position = {position}')
            print(f'vol_shape = {vol_shape}')
            print('----------------------')
        target_pos = max_scale * (position / max_scale).astype(int)
        target_shape = max_scale * np.ceil((position + vol_shape) / max_scale).astype(int) - target_pos
        if verbose:
            print(f'target_pos = {target_pos}')
            print(f'target_shape = {target_shape}')
        with open_file(data_path, mode='r') as f:
            target_vol = f[key][
                target_pos[0]: target_pos[0] + target_shape[0],
                target_pos[1]: target_pos[1] + target_shape[1],
                target_pos[2]: target_pos[2] + target_shape[2]
            ]
        if verbose:
            print(f'target_vol.shape = {target_vol.shape}')

        # Now we have to put the volume to this target_vol at the proper position
        in_target_pos = position - target_pos
        target_vol[
            in_target_pos[0]: in_target_pos[0] + volume.shape[0],
            in_target_pos[1]: in_target_pos[1] + volume.shape[1],
            in_target_pos[2]: in_target_pos[2] + volume.shape[2]
        ] = volume

    else:

        # Everything scales nicely, so we just have to take care that the proper variables exist
        target_vol = volume
        target_pos = position
        target_shape = vol_shape

    return target_vol, target_pos, target_shape


def _scale_and_add_to_dataset(
        data_path, setup_id, timepoint,
        target_pos, target_vol, target_shape,
        scales, downscale_mode, n_threads):

    is_h5 = os.path.splitext(data_path)[1] in HDF5_EXTENSIONS

    # Perform scaling of volume, by generating a temporary bdv file
    # FIXME it is probably sufficient to keep it in memory, since I assume that the volume is in memory anyways, but the
    # FIXME     make_bdv function is so very convenient right now...
    scales = np.array(scales).astype(int)
    scale_factors = scales[1: 2].tolist()
    tmp_name = f'./tmp_{np.random.randint(0, 2 ** 16, dtype="uint16")}'
    if is_h5:
        tmp_bdv = tmp_name + '.h5'
    else:
        tmp_bdv = tmp_name + '.n5'
    for scale in scales[2:]:
        scale_factors.append((scale / np.product(scale_factors, axis=0)).astype(int).tolist())
    make_bdv(
        data=target_vol,
        output_path=tmp_bdv,
        downscale_factors=scale_factors,
        downscale_mode=downscale_mode,
        n_threads=n_threads
    )

    # Now, we just need to fetch the scaled data and put it to the proper positions
    for scale_id, scale in enumerate(scales):

        # Position in the current scale
        pos_in_scale = (target_pos / scale).astype(int)
        shp_in_scale = (target_shape / scale).astype(int)

        # Write the data to each scale
        with open_file(tmp_bdv, mode='r') as f:
            key = get_key(is_h5, timepoint=0, setup_id=0, scale=scale_id)
            scaled_vol = f[key][:]
            # assert list(scaled_vol.shape) == shp_in_scale.tolist()

        with open_file(data_path, mode='a') as f:
            key = get_key(is_h5, timepoint=timepoint, setup_id=setup_id, scale=scale_id)
            f[key][
                pos_in_scale[0]: pos_in_scale[0] + shp_in_scale[0],
                pos_in_scale[1]: pos_in_scale[1] + shp_in_scale[1],
                pos_in_scale[2]: pos_in_scale[2] + shp_in_scale[2]
            ] = scaled_vol

    # Delete the temporary bdv file
    if is_h5:
        os.remove(tmp_bdv)
    else:
        rmtree(tmp_bdv)
    os.remove(tmp_name + '.xml')


class BdvDataset:
    """
    The basic BDV dataset to which volumes can be written using numpy nomenclature.

    The data is included into each of the down-sampling layers accordingly
    The full resolution area is padded, if necessary, to avoid sub-pixel locations in the down-sampling layers
    """

    def __init__(self, path, timepoint, setup_id, downscale_mode='mean', n_threads=1, verbose=False):

        self._path = path
        self._timepoint = timepoint
        self._setup_id = setup_id
        self._downscale_mode = downscale_mode
        self._n_threads = n_threads
        self._verbose = verbose

        # Check if it is h5 or n5
        self._is_h5 = os.path.splitext(path)[1] in HDF5_EXTENSIONS

        # Get the scales
        self._scales = np.array(get_scale_factors(self._path, self._setup_id)).astype(int)

        # Determine full dataset shape
        with open_file(self._path, mode='r') as f:
            self._key = get_key(self._is_h5, timepoint=timepoint, setup_id=setup_id, scale=0)
            self._full_shape = f[self._key].shape

    def _add_to_volume(self, position, volume):

        if self._verbose:
            print(f'scales = {self._scales}')
            print(f'full_shape = {self._full_shape}')

        # Check for out of bounds (and fix it if not)
        position, volume = _check_for_out_of_bounds(position, volume, self._full_shape, verbose=self._verbose)

        # Check if volume and position properly scale to the final scale level (and fix it if not)
        max_scale = self._scales[-1]
        target_vol, target_pos, target_shape = _check_shape_and_position_scaling(
            max_scale, position, volume,
            self._path, self._key,
            verbose=self._verbose)

        # Scale volume and write to target dataset
        _scale_and_add_to_dataset(self._path, self._setup_id, self._timepoint,
                                  target_pos, target_vol, target_shape,
                                  self._scales, self._downscale_mode,
                                  self._n_threads)

    def __setitem__(self, key, value):

        # We are assuming the index to be relative to scale 0 (full resolution)

        position = [k.start for k in key]
        shp = [k.stop - k.start for k in key]
        assert list(value.shape) == shp, f'Shape of array = {value.shape} does not match target shape = {shp}'

        self._add_to_volume(position, value)


# TODO Implement this one that includes stitching operations
class BdvDatasetWithStitching(BdvDataset):

    # Use 'crop' for normal images, 'flow' for supervoxels, and 'iou' for segmentations
    STITCHING_METHODS = ['crop', 'flow', 'iou']

    def __init__(
            self,
            path,
            timepoint,
            setup_id,
            downscale_mode='mean',
            halo=None,
            stitch_method='crop',
            stitch_kwargs={},
            unique=False,
            update_max_id=False,
            n_threads=1,
            verbose=False
    ):

        assert stitch_method in self.STITCHING_METHODS
        assert downscale_mode == 'nearest' or stitch_method == 'crop', 'Downscale mode does not match the stitching ' \
                                                                       'method'
        assert downscale_mode == 'nearest' or not unique, \
            'Unique label stitching methods only work with nearest downsampling mode'

        self._halo = halo
        self._stitch_method = stitch_method
        self._stitch_kwargs = stitch_kwargs
        self._update_max_id = update_max_id
        self._unique = unique

        if unique:
            warn('Using unique stitching: update_max_id set to True')
            self._update_max_id = True
        if stitch_method in ['flow', 'iou']:
            warn(f'Using {stitch_method}: update_max_id set to True')
            self._update_max_id = True

        if stitch_method == 'crop':
            self._stitch_func = self._crop
        elif stitch_method == 'flow':
            self._stitch_func = self._flow
        elif stitch_method == 'iou':
            raise NotImplementedError
        else:
            raise ValueError

        super().__init__(path, timepoint, setup_id, downscale_mode=downscale_mode, n_threads=n_threads, verbose=verbose)

    def set_halo(self, halo):
        """
        Adjust the halo any time you want
        """
        self._halo = halo

    def set_max_id(self, idx, compare_with_present=False):
        """
        Use this to update the largest present id in the dataset if you employ a stitching method with unique == True.
        The id is automatically updated if new data is written.
        """
        data_path = self._path
        with open_file(data_path, 'a') as f:
            key = get_key(self._is_h5, self._timepoint, self._setup_id, 0)
            if not compare_with_present or f[key].attrs['maxId'] < idx:
                f[key].attrs['maxId'] = idx

    def get_max_id(self):
        data_path = self._path
        with open_file(data_path, 'r') as f:
            key = get_key(self._is_h5, self._timepoint, self._setup_id, 0)
            max_id = f[key].attrs['maxId']
        return max_id

    def _crop(self, dd, volume, unique):
        """
        Technically not a stitching method.
        Just crops the relevant data, while completely ignoring what is in the halo.
        """

        if unique:
            raise NotImplementedError

        position = [d.start for d in dd]
        shp = [d.stop - d.start for d in dd]
        assert list(volume.shape) == shp

        halo = self._halo

        volume = volume[
                 halo[0]: -halo[0],
                 halo[1]: -halo[1],
                 halo[2]: -halo[2]
                 ]

        dd = np.s_[
            position[0] + halo[0]: position[0] + shp[0] - halo[0],
            position[1] + halo[1]: position[1] + shp[1] - halo[1],
            position[2] + halo[2]: position[2] + shp[2] - halo[2]
        ]

        return dd, volume

    def _flow(self, dd, volume, unique):

        assert 0 not in volume, 'The zero label is reserved for the background!'

        if unique:
            raise NotImplementedError

        position = [d.start for d in dd]
        shp = [d.stop - d.start for d in dd]
        assert list(volume.shape) == shp

        halo = self._halo
        data_path = self._path

        path_in_file = get_key(self._is_h5, self._timepoint, self._setup_id, 0)

        # Extract target
        target_pos = np.array(position - halo)
        target_shape = np.array(volume.shape)
        with open_file(data_path, mode='r') as f:
            target_vol = load_with_zero_padding(
                f[path_in_file],
                target_pos, target_pos + target_shape, target_shape, verbose=self._verbose
            )

        if len(np.unique(target_vol)) == 1 and np.unique(target_vol)[0] == 0:
            # If nothing is there, just write the volume
            result = volume

        else:

            # Get all objects that are not fully contained in the volume, i.e. touch the volume edges
            nfc_ids = get_non_fully_contained_ids(volume)

            # Write the fully contained objects to a result volume
            result = np.zeros(volume.shape, dtype=volume.dtype)
            result[np.isin(volume, nfc_ids, invert=True)] = volume[np.isin(volume, nfc_ids, invert=True)]
            # Add all objects that are also not present in the target
            result[target_vol == 0] = volume[target_vol == 0]

            # Remove all these objects from the target volume
            target_vol[result != 0] = 0

            # Get all object ids that are left in the target volume
            target_ids = np.unique(target_vol)[1:]

            # Relabel the result volume and omit the target_ids
            result = relabel_with_skip_ids(result, target_ids)

            # Add the objects from the target volume
            result += target_vol

        return dd, result

    def _apply_stitching(self, dd, volume):
        """
        Modifies the target location and source volume according to the stitching method
        """
        assert self._stitch_method == 'crop' or self._halo is not None, \
            'Only crop can be performed without a halo'

        dd, volume = self._stitch_func(dd, volume, self._unique, **self._stitch_kwargs)

        # Update the maximum label
        if self._update_max_id:
            self.set_max_id(int(volume.max()))

        return dd, volume

    def __setitem__(self, key, value):

        # Do the stitching and stuff here
        key, value = self._apply_stitching(key, value)

        # Now call the super with the properly stitched volume
        super().__setitem__(key, value)
