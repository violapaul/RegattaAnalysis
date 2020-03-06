import gdal

FILE = "/Users/viola/Downloads/MBTILES_06.mbtiles"

# open dataset
ds = gdal.Open(FILE)
ds.GetMetadata()

# close dataset
ds = None

print gtif.GetMetadata()
