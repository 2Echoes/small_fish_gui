import imageio.v3 as iio

path = '/home/flo/Documents/IGH projects/SohaQuantif/SCC/input/230723 n1 b-cat bac APC IF fitc ires neo smfish cy3 without puromycin-01.tif'

props = iio.improps(path)
meta = iio.immeta(path)
print(props)
print(meta['channels'], meta['slices'], meta['unit'], meta['hyperstack'], meta['spacing'])