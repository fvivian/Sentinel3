
import numpy as np
from netCDF4 import Dataset
import time
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
#from __future__ import print_function
from ipyleaflet import Map, ImageOverlay, Marker
import os

class asdf:
    
    def __init__(self, ProductName, OutputName=True):
	'''ProductName;		contains the path to the Sentinel-3 product that is being processed/of interest.
		OutputName;	will be the name of the later on generated png file. If set to True, this function will set this variable to 'S3A_' followed by the date and time of the products retrieval.
	'''
        
        self.prodName = ProductName

        if OutputName == True:
            self.out = self.prodName[44:47] + '_' + self.prodName[60:75]
        else:
            self.out = OutputName
	

    def ImportNetCDF(self, NumBand=21):
        '''
        This function imports the variables (radiance and coordinates) of a Sentinel-3 OLCI EFR Product
	NumBand;	The number of bands that want to be imported. Currently just increasing band numbers are supported, i.e. if NumBand is set to 13, the band Oa01 - Oa13 are imported. Note: CalcRGB method requires the first 10 bands.
        '''
    
        bandNumber = np.asarray(range(1,NumBand+1))
        bands = ["%02d" % (bandNumber[i],) for i in range(NumBand)]

        rad = [self.prodName+'Oa'+str(i)+'_radiance.nc' for i in bands]
    
    
        # open the 21 radiance NetCDF:
        start = time.time()
    
        self.data = {}
        for i in range(NumBand):
            Radiance = Dataset(rad[i], mode='r')
            self.data["Oa{0}".format(i+1)] = Radiance.variables['Oa'+bands[i]+'_radiance'][:]
            Radiance.close()
        
        end = time.time()
        print('Importing %i bands took %f seconds.' %(NumBand, end-start))

        # open the coordinate NetCDF:
        global longitude, latitude

        coords = self.prodName + 'geo_coordinates.nc'
        f_co = Dataset(coords, mode='r')
        self.lons = f_co.variables['longitude'][:]
        self.lats = f_co.variables['latitude'][:]
        f_co.close()
        
    
    def CalcRGB(self):
        '''merges the bands 1 to 10 according to the method used by the SNAP toolbox.

        red = log(1.0 + 0.01 * Oa01_radiance + 0.09 * Oa02_radiance + 0.35 * Oa03_radiance + 0.04 * Oa04_radiance + 0.01 * Oa05_radiance + 0.59 * Oa06_radiance + 0.85 * Oa07_radiance + 0.12 * Oa08_radiance + 0.07 * Oa09_radiance + 0.04 * Oa10_radiance)

        green = log(1.0 + 0.26 * Oa03_radiance + 0.21 * Oa04_radiance + 0.50 * Oa05_radiance + Oa06_radiance + 0.38 * Oa07_radiance + 0.04 * Oa08_radiance + 0.03 * Oa09_radiance + 0.02 * Oa10_radiance)

        blue = log(1.0 + 0.07 * Oa01_radiance + 0.28 * Oa02_radiance + 1.77 * Oa03_radiance + 0.47 * Oa04_radiance + 0.16 * Oa05_radiance)

        '''
	arr = self.data

        bands = ['Oa1','Oa2','Oa3','Oa4','Oa5','Oa6','Oa7','Oa8','Oa9','Oa10']
        n, m = np.shape(arr[bands[0]])
        self.red = np.ma.array(np.empty((n,m)), mask=arr[bands[0]].mask)
        self.green = np.ma.array(np.empty((n,m)), mask=arr[bands[0]].mask)
        self.blue = np.ma.array(np.empty((n,m)), mask=arr[bands[0]].mask)

        self.red = np.ma.log10(1.0 + 0.01*arr[bands[0]] + 0.09*arr[bands[1]] + 0.35*arr[bands[2]] + 0.04*arr[bands[3]] + 0.01*arr[bands[4]] + 0.59*arr[bands[5]] + 0.85*arr[bands[6]] + 0.12*arr[bands[7]] + 0.07*arr[bands[8]] + 0.04*arr[bands[9]])

        self.green = np.ma.log10(1.0 + 0.26*arr[bands[2]] + 0.21*arr[bands[3]] + 0.50*arr[bands[4]] + arr[bands[5]] + 0.38*arr[bands[6]] + 0.04*arr[bands[7]] + 0.03*arr[bands[8]] + 0.02*arr[bands[9]])

        self.blue = np.ma.log10(1.0 + 0.07*arr[bands[0]] + 0.28*arr[bands[1]] + 1.77*arr[bands[2]] + 0.47*arr[bands[3]] + 0.16*arr[bands[4]])


    def TransformCoords(self, proj='merc'):
	'''Since the herein used Basemap and its pcolormesh method require specifying the corners and not the centers of the quadrilaterals, the x and y arrays have to be converted/transformed to nx+1.
		At the same time, the Basemap is built on the newly created x/y coordinate arrays.
		proj;	Specifies the used projection. Currently only Mercator ('merc') supported.
	'''

	xgrid = self.scaledlats
	ygrid = self.scaledlons
        cornerLats = self.getCorners(xgrid)
        cornerLons = self.getCorners(ygrid)

	self.basemap = Basemap(projection=proj, llcrnrlat=np.min(cornerLats), urcrnrlat=np.max(cornerLats), llcrnrlon=np.min(cornerLons), urcrnrlon=np.max(cornerLons), resolution='c')

        self.xCorners, self.yCorners = self.basemap(cornerLons, cornerLats)

    def Scaling(self, scale=8):

	self.scaledlons = self.lons[::scale, ::scale]	
	self.scaledlats = self.lats[::scale, ::scale]
	redscaled = self.red[::scale, ::scale]
	greenscaled = self.green[::scale, ::scale]
	bluescaled = self.blue[::scale, ::scale]

	nx, ny = redscaled.shape
	mask = [redscaled.mask, greenscaled.mask, bluescaled.mask, np.ones((nx,ny))]

	rgbtemp = np.ma.array(np.zeros((4,nx,ny)), mask=mask)
	rgbtemp[0] = redscaled; rgbtemp[1] = greenscaled; rgbtemp[2] = bluescaled
	rgbtemp.data[-1,rgbtemp.mask[-1]] = 0.0
	rgbtemp.data[-1,~rgbtemp.mask[1]] = 1.0

	self.rgbraw = rgbtemp

    def getCorners(self, centers):
        '''This function is used to convert the coordinate grid, i.e. from pixel center lat/lons to pixel corner lat/lons (N+1)x(M+1). Output is a new coordinate grid.
        '''

        one = centers[:-1,:]
        two = centers[1:,:]
        d1 = (two - one) / 2.
        one = one - d1
        two = two + d1
        stepOne = np.zeros((centers.shape[0] + 1,centers.shape[1]))
        stepOne[:-2,:] = one
        stepOne[-2:,:] = two[-2:,:]
        one = stepOne[:,:-1]
        two = stepOne[:,1:]
        d2 = (two - one) / 2.
        one = one - d2
        two = two + d2
        stepTwo = np.zeros((centers.shape[0] + 1,centers.shape[1] + 1))
        stepTwo[:,:-2] = one
        stepTwo[:,-2:] = two[:,-2:]
        return stepTwo

    def savePNG(self, createNewDir=False, TargetDir=True):
        '''Plots the data in a projected view and saves it as a png figure.
		createNewDir;	if set to True, it creates a new folder within the current directory. Default is False.
		TargetDir;	lets the user decide on how to name the newly created directory. If set to True (i.e. not changed), it will be named like the product output (i.e. self.out). If other name desired, just enter a string. This is only considered, if createNewDir is set to True.
	'''
        
        if createNewDir:
            if TargetDir == True:
                self.targetDir = './'+self.out
            else:
                self.targetDir = TargetDir
            try:
                os.mkdir(self.targetDir)
                print('Create directory with the name %s' %(self.targetDir))
            except:
                print('Directory with the name %s already exists.' %(self.targetDir))

        rgb = self.rgbraw.T

        color_tuple = rgb.transpose((1,0,2)).reshape((rgb.shape[0]*rgb.shape[1],rgb.shape[2]))/np.max(self.rgbraw)

        plt.rcParams['figure.figsize'] = (5, 5)
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        plt.axis('off')
        cd = self.basemap.pcolormesh(self.xCorners, self.yCorners, self.rgbraw[0,:,:], color=color_tuple, linewidth=0)
        cd.set_array(None)

        if createNewDir:
            plt.savefig(os.path.join(self.targetDir, self.out)+'.png', transparent=True, bbox_inches=extent)
        else:
            plt.savefig(self.out+'.png', transparent=True, bbox_inches=extent)
        plt.close('all')

    def MercatorPlot(self):

        ##### LOAD A GeoJSON MAP FOR THE PLOTTING

        center = [np.min(self.lats)+(np.max(self.lats)-np.min(self.lats))/2, np.min(self.lons)+(np.max(self.lons)-np.min(self.lons))/2]
        zoom = 4

        M = Map(center=center, zoom=zoom)

        ##### PLOT THE PRODUCT ON TOP OF THE MAP WITH ImageOverlay

        imgName = self.out+'.png'

        # bounds need to be in format [SW corner, NE corner]:
        img_bounds = [(np.min(self.lats),np.min(self.lons)), (np.max(self.lats),np.max(self.lons))]

        io = ImageOverlay(url=imgName, bounds=img_bounds)
        M.add_layer(io)

        return M
