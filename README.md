# Sentinel3

Jupyter Notebook created to quickly visualize a Sentinel 3 product in natural colors on a Web Mercator, using netCDF4, Leaflet, Holoviews and mpl_toolkit's Basemap libraries.

The modul loads the data from the NetCDF and creates a RGBA matrix. Basic processing of this RGBA matrix is possible. Furthermore, the processed image will be saved as .png file. If the Mercator projection is used, the image can be shown on a Web Map. Projecting data from far north (i.e. > 75 degrees N) onto the Web Mercator is not suggested.
