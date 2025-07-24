# geos_utils - geospatial utils

A collection of various python processing tools for data handling and geospatial processing tasks.

Github repositories depending on GEOS_utils:
  * [occlusion mapping](https://github.com/MGEOS/CANOPy/tree/main/occlusion_mapping).


## Tools
  * [closest points between two trajectories](./algorithms_tb/closest_points)
  * [voxel traversal algorithm](./algorithms_tb/voxel_traversal)
  * [data management](./data_management)
  * [pointcloud data processing toolbox](./geodata_tb)
  * [raster data processing toolbox](./geodata_tb)
  * [vector data processing toolbox](./geodata_tb)
  * [numba helper functions](./numba_tb)
  * [plotting helper functions](./plotting_tb)


## Installation
Recommended to use python=3.10 or higher.

```bash
conda create -n geos_utils python=3.10
conda activate geos_utils
conda install numpy numba laszip laspy lazrs-python fiona shapely rasterio pyproj pandas geopandas
```


## License
Licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).


## Changelog

### [0.1.0] - 2025-07-22
_First release_