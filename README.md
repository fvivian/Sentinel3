# Sentinel3

Jupyter Notebook created to quickly visualize a Sentinel 3 product in natural colors on a interactive widget (web map). The following modules have to be installed (easiest via conda-forge):

- nedCDF4
- basemap
- ipyleaflet
- pyproj
- holoviews

![alt text](https://github.com/fvivian/Sentinel3/blob/master/Rome.PNG)

Basemap is a mpl_toolkit; in addition to installing the basemap itself, install basemap-data-hires ("conda install -c conda-forge basemap-data-hires"). If you would like to omit this installation, you can change the Basemap resolution argument in the S3processing.py module to "l" (low) or "c" (crude).
The modul loads the data from the NetCDF and creates a RGBA matrix. Basic processing of this RGBA matrix is possible. Furthermore, the processed image will be saved as .png file. If the Mercator projection is used, the image can be shown on a Web Map. Projecting data from far north (i.e. > 75 degrees N) onto the Web Mercator is not suggested.


![alt text](https://github.com/fvivian/Sentinel3/blob/master/Sentinel3_Leaflet.PNG)
