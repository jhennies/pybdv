
import os
import numpy as np
from .util import open_file, get_scale_factors, get_key, HDF5_EXTENSIONS
from .downsample import downsample_in_memory
from .stitching_utils import load_with_zero_padding, get_non_fully_contained_ids, relabel_with_skip_ids
from .stitching_utils import get_block_faces, match_ids_at_block_faces
from .stitching_utils import relabel_from_mapping, iou, largest_non_zero_overlap
from warnings import warn
import json
import pickle


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

    scales = np.array(scales).astype(int)
    scale_factors = scales[1: 2].tolist()

    for scale in scales[2:]:
        scale_factors.append((scale / np.product(scale_factors, axis=0)).astype(int).tolist())

    # Scale the data
    downscaled_vols = [target_vol]
    downscaled_vols.extend(
        downsample_in_memory(
            target_vol,
            downscale_factors=scale_factors,
            downscale_mode=downscale_mode,
            block_shape=(64, 64, 64),
            n_threads=n_threads
        )
    )

    # Now, we just need to put it to the proper positions in the bdv file
    for scale_id, scale in enumerate(scales):

        # Position in the current scale
        pos_in_scale = (target_pos / scale).astype(int)
        shp_in_scale = (target_shape / scale).astype(int)

        scaled_vol = downscaled_vols[scale_id]

        with open_file(data_path, mode='a') as f:
            key = get_key(is_h5, timepoint=timepoint, setup_id=setup_id, scale=scale_id)
            f[key][
                pos_in_scale[0]: pos_in_scale[0] + shp_in_scale[0],
                pos_in_scale[1]: pos_in_scale[1] + shp_in_scale[1],
                pos_in_scale[2]: pos_in_scale[2] + shp_in_scale[2]
            ] = scaled_vol


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
    STITCHING_METHODS = ['crop', 'flow', 'iou', 'make_mapping']

    def __init__(
            self,
            path,
            timepoint,
            setup_id,
            downscale_mode='mean',
            halo=None,
            stitch_method='crop',
            stitch_kwargs={},
            background_value=None,
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
        self._background_value = background_value

        if unique:
            warn('Using unique stitching: update_max_id set to True')
            self._update_max_id = True
        if stitch_method in ['flow', 'iou', 'make_mapping']:
            warn(f'Using {stitch_method}: update_max_id set to True')
            self._update_max_id = True

        if stitch_method == 'crop':
            self._stitch_func = self._crop
        elif stitch_method == 'flow':
            self._stitch_func = self._flow
        elif stitch_method == 'iou':
            raise NotImplementedError
            # self._stitch_func = self._iou
        elif stitch_method == 'make_mapping':
            if not unique:
                warn(f'Using make_mapping: unique set to True')
            self._stitch_func = self._make_mapping
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
            max_id = self.get_max_id()
        else:
            max_id = 0

        position = [d.start for d in dd]
        shp = [d.stop - d.start for d in dd]
        assert list(volume.shape) == shp

        halo = self._halo

        volume = volume[
                 halo[0]: -halo[0],
                 halo[1]: -halo[1],
                 halo[2]: -halo[2]
                 ]
        if unique:
            if self._background_value is not None:
                assert self._background_value == 0, 'Only implemented for background value == 0'
                volume[volume != 0] = volume[volume != 0] + max_id
            else:
                volume += max_id

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

        position = np.array([d.start for d in dd])
        shp = [d.stop - d.start for d in dd]
        assert list(volume.shape) == shp

        halo = np.array(self._halo)
        data_path = self._path

        position = position + halo

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

    def _get_halo_blocks(self, shp, pos):

        pos = np.array(pos)
        halo = np.array(self._halo)

        # Defining the halo as rectangular blocks
        block_xy0 = np.array([[0, 0, 0], [shp[0], shp[1], halo[2]]])
        block_xy1 = np.array([[0, 0, shp[2] - halo[2]], [shp[0], shp[1], shp[2]]])
        block_xz0 = np.array([[0, 0, halo[2]], [shp[0], halo[1], shp[2] - halo[2]]])
        block_xz1 = np.array([[0, shp[1] - halo[1], halo[2]], [shp[0], shp[1], shp[2] - halo[2]]])
        block_yz0 = np.array([[0, halo[1], halo[2]], [halo[0], shp[1] - halo[1], shp[2] - halo[2]]])
        block_yz1 = np.array([[shp[0] - halo[0], halo[1], halo[2]], [shp[0], shp[1] - halo[1], shp[2] - halo[2]]])

        blocks = [block_xy0, block_xy1, block_xz0, block_xz1, block_yz0, block_yz1]
        blocks_pos = [block + pos for block in blocks]

        return blocks, blocks_pos

    def _load_data_from_blocks(self, blocks):

        data_path = self._path
        path_in_file = get_key(self._is_h5, self._timepoint, self._setup_id, 0)

        f = open_file(data_path, mode='r')
        dataset = f[path_in_file]
        data = [load_with_zero_padding(dataset, block[0], block[1], block[1] - block[0], verbose=self._verbose)
                for block in blocks]
        f.close()

        return data

    # @staticmethod
    # def _flatten_data_blocks(blocks):
    #
    #     flat_blocks = [block.flatten() for block in blocks]
    #     flat_blocks = np.concatenate(flat_blocks, axis=0)
    #     assert flat_blocks.ndim == 1
    #
    #     return flat_blocks
    #
    # @staticmethod
    # def _match_by_iou(src, tgt, labels_to_check=None, min_label_for_non_matched=None):
    #
    #     if min_label_for_non_matched is not None:
    #         assert min_label_for_non_matched > np.unique(tgt)[-1]
    #
    #     if labels_to_check is None:
    #         labels_to_check = np.unique(src)
    #         labels_to_keep = None
    #     else:
    #         # Label has to be in src and labels_to_check
    #         # print(f'labels_to_check = {labels_to_check}')
    #         # print(f'np.unique(src) = {np.unique(src)}')
    #         # print(f'labels_to_check - src = {sorted(list(set(labels_to_check) - set(np.unique(src))))}')
    #         labels_to_keep = sorted(list(set(labels_to_check) - set(np.unique(src))))
    #         labels_to_check = list(set(labels_to_check) & set(np.unique(src)))
    #         # print(f'labels_to_check = {labels_to_check}')
    #
    #     print(f'src.shape = {src.shape}')
    #     print(f'tgt.shape = {tgt.shape}')
    #
    #     matches = {}
    #     c = min_label_for_non_matched
    #     for lbl in labels_to_check:
    #
    #         if lbl > 0:
    #             # Only the larges component is a suspect (considering we want a IOU of > 0.5 or more)
    #             print(f'tgt[src == lbl] = {tgt[src == lbl]}')
    #             tgt_lbl, overlap_ratio = largest_non_zero_overlap(src, tgt, lbl)
    #             # tgt_lbl = np.unique(tgt[src == lbl])[-1]
    #             print(f'lbl = {lbl}')
    #             print(f'tgt_lbl = {tgt_lbl}')
    #             print(f'overlap_ratio = {overlap_ratio}')
    #             if tgt_lbl is not None and overlap_ratio > 0.8:
    #                 print(f'Match found with ratio = {overlap_ratio}')
    #                 matches[lbl] = tgt_lbl
    #             else:
    #                 print('No match found')
    #                 matches[lbl] = c
    #                 c += 1
    #             # print(f'iou = {iou((src == lbl).astype("uint8"), (tgt == tgt_lbl).astype("uint8"))}')
    #             # if tgt_lbl > 0 and iou((src == lbl).astype('uint8'), (tgt == tgt_lbl).astype('uint8')) > 0.8:
    #             #     print(f'Match found with iou = {iou(src == lbl, tgt == tgt_lbl)}')
    #             #     matches[lbl] = tgt_lbl
    #             # else:
    #             #     print('No match found')
    #             #     matches[lbl] = c
    #             #     c += 1
    #         matches[0] = 0
    #
    #     if labels_to_keep is not None:
    #         for lidx, lbl in enumerate(labels_to_keep):
    #             print(f'{lbl}: {c + lidx}')
    #             matches[lbl] = c + lidx
    #
    #     return matches
    #
    # def _iou(self, dd, volume, unique=True):
    #
    #     assert unique
    #
    #     position = np.array([d.start for d in dd])
    #     halo = np.array(self._halo)
    #     data_path = self._path
    #     path_in_file = get_key(self._is_h5, self._timepoint, self._setup_id, 0)
    #     shp = np.array(volume.shape)
    #
    #     # Define the halo blocks
    #     halo_blocks, target_halo_blocks = self._get_halo_blocks(shp, position)
    #
    #     # Get data in halo
    #     target_halo = self._load_data_from_blocks(target_halo_blocks)
    #     vol_halo = [load_with_zero_padding(volume, block[0], block[1], block[1] - block[0], verbose=False)
    #                 for block in halo_blocks]
    #
    #     # Flatten the data blocks (pixel locations don't matter here)
    #     target_halo_flat = self._flatten_data_blocks(target_halo)
    #     vol_halo_flat = self._flatten_data_blocks(vol_halo)
    #
    #     # Crop away the halo from the input volume
    #     main_vol = volume[
    #         halo[0]: -halo[0],
    #         halo[1]: -halo[1],
    #         halo[2]: -halo[2],
    #     ]
    #
    #     # Make matches using IOU, thus creating the mapping dictionary
    #     with open_file(data_path, 'r') as f:
    #         max_id = f[get_key(False, 0, 0, 0)].attrs['maxId']
    #     mapping = self._match_by_iou(
    #         vol_halo_flat, target_halo_flat, labels_to_check=np.unique(main_vol), min_label_for_non_matched=max_id + 1)
    #
    #     # Do the mapping
    #     result = relabel_from_mapping(main_vol, mapping)
    #
    #     # Update the max id
    #     with open_file(data_path, 'a') as f:
    #         if np.max(list(mapping.values())) > max_id:
    #             f[get_key(False, 0, 0, 0)].attrs['maxId'] = int(np.max(list(mapping.values())))
    #
    #     # Adapt the slicing
    #     dd = np.s_[
    #         position[0] + halo[0]: position[0] + shp[0] - halo[0],
    #         position[1] + halo[1]: position[1] + shp[1] - halo[1],
    #         position[2] + halo[2]: position[2] + shp[2] - halo[2]
    #     ]
    #     return dd, result

    def _make_mapping(self, dd, volume, unique, mapping_method='match_block_faces', save_filepath=None):
        """
        mapping_method: ['match_block_faces']
            TODO: implement 'iou'
        """

        assert unique, 'Stitching method "make_mapping" only works with unique labels'
        assert self._background_value == 0, 'Only tested using a background value of 0'
        assert mapping_method in ['match_block_faces'], f'Invalid mapping method: {mapping_method}'
        assert save_filepath is not None, (
            'Supply a json (*.json) or pickle (*.pkl) file to save the mapping: '
            'stitch_kwargs={"save_filepath": "path/to/mapping.json"}'
        )

        # To make sure that we don't get an overflow we need to convert the volume to uint64
        volume = volume.astype('uint64')

        dd, volume = self._crop(dd, volume, unique)

        # The data from the target block has to be one pixel larger in all dimensions
        tgt_shp = np.array(volume.shape) + 2
        tgt_pos = np.array([d.start for d in dd]) - 1

        data_path = self._path
        path_in_file = get_key(self._is_h5, self._timepoint, self._setup_id, 0)
        # FIXME It is not necessary to load the full volume:
        # TODO Move the data loading into get_block_faces and load only the block faces
        with open_file(data_path, mode='r') as f:
            target_vol = load_with_zero_padding(
                f[path_in_file],
                tgt_pos, tgt_pos + tgt_shp, tgt_shp, verbose=self._verbose
            )
        block_faces_target = get_block_faces(target_vol)
        block_faces_vol = get_block_faces(volume)

        mapping = match_ids_at_block_faces(block_faces_vol, block_faces_target, crop=True)
        print(f'mapping = {mapping}')
        save_type = os.path.splitext(save_filepath)[1]
        if save_type == '.json':
            with open(save_filepath, 'w') as f:
                json.dump(mapping, f)
        elif save_type == '.pkl':
            with open(save_filepath, 'wb') as f:
                pickle.dump(mapping, f)
        else:
            raise ValueError(f'Invalid file type requested: {save_type}')

        return dd, volume

    def _apply_stitching(self, dd, volume):
        """
        Modifies the target location and source volume according to the stitching method
        """
        assert self._stitch_method == 'crop' or self._halo is not None, \
            'Only crop can be performed without a halo'

        dd, volume = self._stitch_func(dd, volume, self._unique, **self._stitch_kwargs)

        # Update the maximum label
        if self._update_max_id:
            vol_max = int(volume.max())
            print(f'Updating max_id to: {vol_max}')
            self.set_max_id(vol_max, compare_with_present=True)

        return dd, volume

    def __setitem__(self, key, value):

        # Do the stitching and stuff here
        key, value = self._apply_stitching(key, value)

        # Now call the super with the properly stitched volume
        super().__setitem__(key, value)
