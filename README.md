# geos_utils - geospatial utils

A collection of various python processing tools for data handling and geospatial processing tasks.

Github repositories depending on GEOS_utils:
  * [CANOPy](https://github.com/MGEOS/CANOPy)



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



## Citation
If you find this useful for your research, please consider citing our paper:

```bibtex
@article{gassilloud2025occlusion,
  title={Occlusion mapping reveals the impact of flight and sensing parameters on vertical forest structure exploration with cost-effective UAV based laser scanning},
  author={Gassilloud, Matthias and Koch, Barbara and Goeritz, Anna},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={139},
  pages={104493},
  year={2025},
  publisher={Elsevier}
}
```



## License
Licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).



## Changelog


### [0.1.2] - 2025-10-22

#### Changed
  * function voxelize_pointcloud(): can take directly a numpy array containing point coordinates  [x, y, z] with shape (n,3) as first argument

#### Fixed
  * relative module import
  * function normalize_vox_array(): output data type same as input array
  * function vox_aoi(): correct calculation of cell_size decimal places



### [0.1.1] - 2025-08-04

#### Added
  * function delete_file(): delete a file
  * function find_filenames(): find files in a folder following a regex pattern
  * function print_array_size_gb(): print size of a numpy array
  * function get_file_size_gb(): return size of file
  * function df_instances_to_dict(): convert instances of a dataframe to a dictionary



### [0.1.0] - 2025-07-22
_First release_