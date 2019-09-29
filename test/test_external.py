import os
import unittest
from shutil import rmtree

import imageio
import h5py
import numpy as np
import skimage.data


# dummy test data, needs to be converted to bdv/xml
# with FIJI externally
def make_test_data(write=False):
    d = skimage.data.astronaut()
    d = d.transpose((2, 0, 1))
    d = np.concatenate(2 * [d], axis=0)
    d = d.astype('float32')
    d *= (np.iinfo('uint16').max / np.iinfo('uint8').max)
    d = d.astype('uint16')
    if write:
        imageio.volwrite('../data/example.tif', d)
    return d


class TestExternal(unittest.TestCase):
    tmp_folder = 'tmp'

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def test_external(self):
        from pybdv import make_bdv
        d = make_test_data()
        res_path = os.path.join(self.tmp_folder, 'data.h5')
        exp_path = '../data/example.h5'
        make_bdv(d, res_path, convert_dtype=True)

        with h5py.File(exp_path, 'r') as f:
            ds = f['t00000/s00/0/cells']
            exp = ds[:]
        with h5py.File(res_path, 'r') as f:
            ds = f['t00000/s00/0/cells']
            res = ds[:]

        self.assertEqual(res.dtype, exp.dtype)
        self.assertEqual(res.shape, exp.shape)
        self.assertTrue(np.array_equal(res, exp))


if __name__ == '__main__':
    # make_test_data(True)
    unittest.main()