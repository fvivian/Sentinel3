
import numpy as np
from netCDF4 import Dataset
import time
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from ipyleaflet import Map, ImageOverlay, Marker, TileLayer
import os
from pyproj import Proj, transform
import holoviews as hv
hv.notebook_extension()

class OLCIprocessing:
    
    def __init__(self, ProductName, OutputName=True):
        '''ProductName; contains the path to the Sentinel-3 product that is being processed/of interest.
            OutputName; will be the name of the later on generated png file. If set to True, this function will set this variable to 'S3A_' followed by the date and time of the products retrieval.'''
        
        self.prodName = ProductName

        if OutputName == True:
            self.out = self.prodName[44:47] + '_' + self.prodName[60:75]
        else:
            self.out = OutputName


    def importNetCDF(self, NumBand=21):
        '''
        This function imports the variables (radiance and coordinates) of a Sentinel-3 OLCI EFR Product
        NumBand; The number of bands that want to be imported. Currently just increasing band numbers are supported, i.e. if NumBand is set to 13, the band Oa01 - Oa13 are imported. Note: CalcRGB method requires the first 10 bands.
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

        coords = self.prodName + 'geo_coordinates.nc'
        f_co = Dataset(coords, mode='r')
        self.lons = f_co.variables['longitude'][:]
        self.lats = f_co.variables['latitude'][:]
        f_co.close()
        
        if self.lats.max() > 75:
            print('The image contains regions north of 75 degrees north and therefore it is suggested to use the Lambert Azimuthal Equal-Area (laea) projection for better visibility. For that purpose, use the attribute transformCoords to set up the subsequent procedure.')
        
    def importIMG(self, filetype='png'):
        '''imports the red/green/blue values from the .png/.jpeg file from the Sentinel 3 product.
        '''
        
        from skimage import io
        
        imgName = self.prodName + self.prodName[-100:-5] + filetype
        
        self.im = io.imread(imgName)
        
        
    def calcRGB(self, method='bands357'):
        
        self.calcMethod = method
        
        if method == 'bands357':
            '''uses the bands number 3 (blue), 5 (green), and 7 (red)'''
            n, m = np.shape(self.data['Oa1'])
            self.red = np.ma.array(np.empty((n,m)), mask=self.data['Oa7'].mask)
            self.green = np.ma.array(np.empty((n,m)), mask=self.data['Oa5'].mask)
            self.blue = np.ma.array(np.empty((n,m)), mask=self.data['Oa3'].mask)
            
            self.red   = self.data['Oa7']
            self.green = self.data['Oa5']
            self.blue  = self.data['Oa3']
        
        if method == 'log':
            '''merges the bands 1 to 10 according to the method used by the SNAP toolbox.'''
        
            arr = self.data
    
            bands = ['Oa1','Oa2','Oa3','Oa4','Oa5','Oa6','Oa7','Oa8','Oa9','Oa10']
            n, m = np.shape(arr[bands[0]])
            self.red = np.ma.array(np.empty((n,m)), mask=arr[bands[0]].mask)
            self.green = np.ma.array(np.empty((n,m)), mask=arr[bands[0]].mask)
            self.blue = np.ma.array(np.empty((n,m)), mask=arr[bands[0]].mask)

            self.red = np.ma.log10(1.0 + 
                                   0.01*arr[bands[0]] + 
                                   0.09*arr[bands[1]] + 
                                   0.35*arr[bands[2]] + 
                                   0.04*arr[bands[3]] + 
                                   0.01*arr[bands[4]] + 
                                   0.59*arr[bands[5]] + 
                                   0.85*arr[bands[6]] + 
                                   0.12*arr[bands[7]] + 
                                   0.07*arr[bands[8]] + 
                                   0.04*arr[bands[9]])

            self.green = np.ma.log10(1.0 + 
                                     0.26*arr[bands[2]] + 
                                     0.21*arr[bands[3]] + 
                                     0.50*arr[bands[4]] + 
                                     arr[bands[5]] + 
                                     0.38*arr[bands[6]] + 
                                     0.04*arr[bands[7]] + 
                                     0.03*arr[bands[8]] + 
                                     0.02*arr[bands[9]])

            self.blue = np.ma.log10(1.0 + 
                                    0.07*arr[bands[0]] + 
                                    0.28*arr[bands[1]] + 
                                    1.77*arr[bands[2]] + 
                                    0.47*arr[bands[3]] + 
                                    0.16*arr[bands[4]])
            
        d3mask = [self.data['Oa1'].mask, self.data['Oa1'].mask, self.data['Oa1'].mask]  
        self.rgb = np.ma.array(np.empty((n,m,3)), mask=d3mask)
        self.rgb[:,:,0] = self.red
        self.rgb[:,:,1] = self.green
        self.rgb[:,:,2] = self.blue
        
    def transformCoords(self, proj='laea'):
        ''' transform the coordinates to another projection.'''
        
        self.proj = proj
        if self.proj == 'laea':
            inProj = Proj(init='epsg:4326')
            outProj= Proj(init='epsg:3575')
            self.xgrid, self.ygrid = transform(inProj, outProj, self.lons, self.lats)
        try:
            self.scaledx, self.scaledy = transform(inProj, outProj, self.scaledlons, self.scaledlats)
        except:
            pass

                

    def createBasemap(self):
        '''The Basemap is built on the newly created x/y coordinate arrays. proj; Specifies the used projection. Currently only Mercator ('merc') and Lambert Azimuthal Equal-Area ('laea') supported.
        '''
        try:
            proj = self.proj
        except:
            self.proj = 'merc'
            proj = self.proj
        
        if proj=='merc':
            
            lonCorners = self.getCorners(self.scaledlons)
            latCorners = self.getCorners(self.scaledlats)
            
            self.basemap = Basemap(projection=proj, 
                                   llcrnrlat=np.min(latCorners), 
                                   urcrnrlat=np.max(latCorners), 
                                   llcrnrlon=np.min(lonCorners), 
                                   urcrnrlon=np.max(lonCorners), 
                                   resolution='c')
            
        if proj=='laea':

            lonCorners = self.getCorners(self.scaledlons)
            latCorners = self.getCorners(self.scaledlats)
            
            self.transformCoords()
            self.getBoundaries()
    
            self.basemap = Basemap(projection=proj,
                                   urcrnrlon=self.topright[0],
                                   urcrnrlat=self.topright[1],
                                   llcrnrlon=self.bottomleft[0],
                                   llcrnrlat=self.bottomleft[1],
                                   #lat_ts=80,
                                   lat_0=90,
                                   lon_0=10,
                                   resolution='c')

        self.xCorners, self.yCorners = self.basemap(lonCorners, latCorners)

    def scaling(self, scale=8):

        self.scaledlons = self.lons[::scale, ::scale]
        self.scaledlats = self.lats[::scale, ::scale]
        try:
            self.scaledx = self.xgrid[::scale, ::scale]
            self.scaledy = self.ygrid[::scale, ::scale]
        except:
            pass
        
        redscaled = self.red[::scale, ::scale]
        greenscaled = self.green[::scale, ::scale]
        bluescaled = self.blue[::scale, ::scale]

        nx, ny = redscaled.shape
        mask = [redscaled.mask, greenscaled.mask, bluescaled.mask, np.ones((nx,ny))]

        rgbtemp = np.ma.array(np.zeros((4,nx,ny)), mask=mask)
        rgbtemp[0] = redscaled; rgbtemp[1] = greenscaled; rgbtemp[2] = bluescaled
        rgbtemp.data[-1,rgbtemp.mask[-1]] = 0.0
        rgbtemp.data[-1,~rgbtemp.mask[1]] = 1.0

        self.rgbscaled = rgbtemp

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
    
    def createDynmap(self):
    
        dmap = hv.DynamicMap(self.clip, kdims=['lower', 'upper'])
    
        n = 15
        if self.calcMethod == 'log':
            max09 = 0.9*self.rgbscaled.max()
            max01 = 0.1*self.rgbscaled.max()
            upper_limit = np.linspace(max09, self.rgbscaled.max(), num=n)
            lower_limit = np.linspace(0, max01, num=n)
        else:
            upper_limit = np.linspace(np.percentile(self.rgbscaled.flatten(), 50),
                                      np.percentile(self.rgbscaled.flatten(), 95), num=n)
            lower_limit = np.linspace(np.percentile(self.rgbscaled.flatten(), 0),
                                      np.percentile(self.rgbscaled.flatten(), 50), num=n)
            

        return dmap.redim.values(lower=lower_limit, upper=upper_limit)

    def clip(self, lower_lim, upper_lim):
        '''this functions cuts off the values below/above the lower/upper limit, respectively, and then stretches/squeezes the remaining values on [0, 1].'''
    
        bands_array = np.asarray(self.rgbscaled).astype(float)
        image_array_clip = np.ma.masked_array(np.ones(self.rgbscaled.shape),
                                              mask=self.rgbscaled.mask.copy)
        image_array = np.ma.masked_array(np.empty(self.rgbscaled.shape),
                                         mask=self.rgbscaled.mask.copy)
      
        image_array_clip[:3,:,:] = np.clip(self.rgbscaled.data[:3,:,:], lower_lim, upper_lim)
    
        image_array[:3,:,:] = self.array_normalisation(image_array_clip.data[:3,:,:])
        image_array.data[-1] = self.rgbscaled.data[-1]
        image_array.mask = self.rgbscaled.mask
        arr = np.ma.dstack(image_array)
        
        #return self.plotImg(image_array)
        return hv.RGB(arr)
        
    def array_normalisation(self, array,new_min=0.0,new_max=1.0):
        """To normalise an input array."""

        array = array.astype(float)

        old_min = np.amin(array)
        old_max = np.amax(array)

        array = new_min + (array - old_min) * (new_max - new_min) / (old_max - old_min)

        return array
    
    def plotImg(self, array):
        
        rgb = np.ma.dstack(array)
        img = hv.RGB(rgb)
        
        rangexy = hv.streams.RangeXY(source=img)
        img << hv.DynamicMap(self.selected_hist, streams=[rangexy])
        return img << hv.DynamicMap(self.selected_hist, streams=[rangexy])
        
    def selected_hist(self, x_range, y_range):
        obj = img.select(x=x_range, y=y_range) if x_range and y_range else img
        return hv.operation.histogram(obj)
    
    def savePNG(self, array=None, TargetDir='./'):
        '''Plots the data in a projected view and saves it as a png figure.
        '''
        
        plt.close('all')
        
        if TargetDir=='./':
            self.targetDir = './'+self.out
        else:
            self.targetDir = TargetDir + self.out
            try:
                os.mkdir(self.targetDir)
                print('Create directory with the name %s' %(self.targetDir))
            except:
                print('Directory with the name %s already exists. Image saved in it.' %(self.targetDir))

        try:
            rgb0 = np.ma.array(np.empty(self.rgbscaled.shape),
                               mask=self.rgbscaled.mask.copy)
            rgb0[0] = array[:,:,0]
            rgb0[1] = array[:,:,1]
            rgb0[2] = array[:,:,2]
            rgb0[3] = array[:,:,3]
            rgb = rgb0.T
        except:
            rgb0 = self.rgbscaled
            rgb = self.rgbscaled.T
            
        color_tuple = rgb.transpose((1,0,2)).reshape((rgb.shape[0]*rgb.shape[1],rgb.shape[2]))/np.max(rgb0)

        plt.rcParams['figure.figsize'] = (5, 5)
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        plt.axis('off')
        if self.basemap is not None:
            self.createBasemap()
            
        self.cm = self.basemap.pcolormesh(self.xCorners, self.yCorners, rgb0[1,:,:], color=color_tuple, linewidth=0)
        self.cm.set_array(None)

        plt.savefig(self.out+'.png', transparent=True, bbox_inches=extent)
        plt.close('all')
        
    def getBoundaries(self):
        '''find the boundary coordinates for the basemap plot. called by the function createBasemap, only if proj=='laea'.'''
        
        try:
            topright   = [self.scaledx.max(), self.scaledy.max()]
            bottomleft = [self.scaledx.min(), self.scaledy.min()]
        except:
            self.transformCoords()
            topright   = [self.scaledx.max(), self.scaledy.max()]
            bottomleft = [self.scaledx.min(), self.scaledy.min()]
          
        topright = np.asarray(topright)
        bottomleft=np.asarray(bottomleft)
        if all(topright > 0):
            topright *= 1.1
        if all(topright < 0):
            topright *= 0.9
        if topright[0] > 0 and topright[1] < 0:
            topright[0] *= 1.1
            topright[1] *= 0.9
        if topright[0] < 0 and topright[1] > 0:
            topright[0] *= 0.9
            topright[1] *= 1.1
        
        if all(bottomleft < 0):
            bottomleft *= 1.1
        if all(bottomleft > 0):
            bottomleft *= 0.9
        if bottomleft[0] < 0 and bottomleft[1] > 0:
            bottomleft[0] *= 1.1
            bottomleft[1] *= 0.9
        if bottomleft[0] > 0 and bottomleft[1] < 0:
            bottomleft[0] *= 0.9
            bottomleft[1] *= 1.1
        
        # transform coords to epsg 4326:
        inProj = Proj(init='epsg:3575')
        outProj= Proj(init='epsg:4326')
        self.topright = transform(inProj, outProj, topright[0], topright[1])
        self.bottomleft = transform(inProj, outProj, bottomleft[0], bottomleft[1])  

    def MercatorPlot(self):

        ##### LOAD A GeoJSON MAP FOR THE PLOTTING

        center = [np.min(self.lats)+(np.max(self.lats)-np.min(self.lats))/2,
                  np.min(self.lons)+(np.max(self.lons)-np.min(self.lons))/2]
        #center = [0, 0]
        zoom = 4


        if self.proj=='laea':
            tls = TileLayer(opacity=1.0,
                                        url='https://{s}.tiles.arcticconnect.org/osm_3575/{z}/{x}/{y}.png',
                                        zoom=0,
                                        max_zoom=10,
                                        attribution='Map data (c) <a href="https://webmap.arcticconnect.org/">ArcticConnect</a> . Data (c) <a href="http://osm.org/copyright">OpenStreetMap</a>')
            M=Map(default_tiles=tls, center=center, zoom=zoom)
        else:
            M=Map(center=center, zoom=zoom)

        ##### PLOT THE PRODUCT ON TOP OF THE MAP WITH ImageOverlay

        imgName = self.out+'.png'

        try:
            img_bounds = [self.bottomleft, self.topright]
        except:
            img_bounds = [(np.min(self.lats),np.min(self.lons)), (np.max(self.lats),np.max(self.lons))]
        
        io = ImageOverlay(url=imgName, bounds=img_bounds)
        M.add_layer(io)

        return M
    
    def mapPlot(self, array=None):
        
        if self.proj=='laea':

            self.basemap.drawmapboundary()
            self.basemap.drawcoastlines()
            self.basemap.drawparallels(np.arange(-80.,81.,20.))
            self.basemap.drawmeridians(np.arange(-180.,181.,20.))
        
            try:
                rgb0 = np.ma.array(np.empty(self.rgbscaled.shape),
                                   mask=self.rgbscaled.mask.copy)
                rgb0[0] = array[:,:,0]
                rgb0[1] = array[:,:,1]
                rgb0[2] = array[:,:,2]
                rgb0[3] = array[:,:,3]
                rgb = rgb0.T
            except:
                rgb0 = self.rgbscaled
                rgb = self.rgbscaled.T

            color_tuple = rgb.transpose((1,0,2)).reshape((rgb.shape[0]*rgb.shape[1],rgb.shape[2]))/np.max(rgb0)
            self.cm = self.basemap.pcolormesh(self.xCorners,
                                              self.yCorners,
                                              self.rgbscaled[2,:,:],
                                              color=color_tuple,
                                              linewidth=0)
            self.cm.set_array(None)

            plt.show()
            
        else:

            self.basemap.drawmapboundary()
            self.basemap.drawcoastlines()
            self.basemap.drawparallels(np.arange(-80.,81.,20.))
            self.basemap.drawmeridians(np.arange(-180.,181.,20.))
        
            try:
                rgb0 = np.ma.array(np.empty(self.rgbscaled.shape),
                                   mask=self.rgbscaled.mask.copy)
                rgb0[0] = array[:,:,0]
                rgb0[1] = array[:,:,1]
                rgb0[2] = array[:,:,2]
                rgb0[3] = array[:,:,3]
                rgb = rgb0.T
            except:
                rgb0 = self.rgbscaled
                rgb = self.rgbscaled.T

            color_tuple = rgb.transpose((1,0,2)).reshape((rgb.shape[0]*rgb.shape[1],rgb.shape[2]))/np.max(rgb0)
            self.cm = self.basemap.pcolormesh(self.xCorners,
                                              self.yCorners,
                                              self.rgbscaled[2,:,:],
                                              color=color_tuple,
                                              linewidth=0)
            self.cm.set_array(None)

            plt.show()
            plt.close('all')
        