import pystac_client
import planetary_computer
import rasterio
import numpy
import pandas as pd
import geopandas as gpd
import rioxarray
import xarray
import xrspatial
import stackstac
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
import scipy.stats
import rasterstats
import streamlit as st
from datetime import datetime
from rasterstats import zonal_stats
from matplotlib.ticker import MaxNLocator

st.title('☘🌳🌴 Canopy Cover Percentage Detection ☘🌳🌴')

st.set_page_config(
    page_title = 'NDVI Canopy Cover Detection for Land Parcels',
    page_icon = '☘🌳🌴',
    layout = 'wide',
    initial_sidebar_state = 'expanded'
)

st.write('This Web Application will help critical decision making process aiding in project areas acceptance or rejection based on canopy cover percentage (10%)')

st.subheader(body = 'Steps for User',
             divider = 'rainbow' )

st.text('1. Upload Region of Interest (KML/GeoPackage/GeoJson)')
st.text('2. Select the Date Range (Start Date and End Date))')
st.text('3. Download your Final File (KML)')

roi = st.file_uploader(
    label = 'Upload Your File Here!',
    type = ['gpkg', 'KML', 'geojson'],
    accept_multiple_files = False,
)

if roi != None:
    st.success(
    body = 'Uploaded Successfully', 
    icon = '✔️')

    roi = gpd.read_file(roi)

    st.write(roi)

    roi_geometry = roi.geometry.union_all().__geo_interface__

    crs = st.text_input(
        label = 'Please Enter EPSG Code suitable for your Region of Interest (eg., EPSG:32644)'
        )
    
    roi_reprojected = roi.to_crs(str(crs))

    startDate = st.date_input(
        label= 'Select Start Date',
        min_value = datetime(2017, 3, 28),
        format = 'YYYY-MM-DD'
    )

    endDate = st.date_input(
        label= 'Select End Date',
        value = 'today',
        min_value = datetime(2017, 3, 28),
        format = 'YYYY-MM-DD'
    )

    dateRange = str(startDate) + '/' + str(endDate)

    st.text(f'Selected Date Range is {startDate} to {endDate}')

    
    if crs != 'None' and startDate != None and endDate != None:
        
        @st.cache_data
        def sentinel_extractor(cloud_cover = 20, proj_crs = str(crs)):
            
            global roi_reprojected
            roi_reprojected = roi.to_crs(proj_crs)
            roi_geometry = roi.geometry.union_all().__geo_interface__

            st.badge('Searching Sentinel 2 Images for Selected Dates and Region')

            catalog = pystac_client.Client.open(
                'https://planetarycomputer.microsoft.com/api/stac/v1',
                modifier = planetary_computer.sign_inplace
            )

            search = catalog.search(

                collections = 'sentinel-2-l2a',
                intersects = roi_geometry,
                datetime = dateRange,
                query = {'eo:cloud_cover': {'lt': cloud_cover}}
            )

            items = search.get_all_items()
            st.text(f'Found {len(items)} items from Sentinel 2 L2A (SR)')

            bandNames = ['B02', 'B03', 'B04', 'B08']

            epsg_code = proj_crs.split(':')
            epsg_code = int(epsg_code[1])

            stacked = stackstac.stack(items, assets = bandNames, epsg = epsg_code, chunksize = 1024,)
            stacked_clipped = stacked.rio.clip(roi_reprojected.geometry, roi_reprojected.crs, drop = True)
            stacked_median = stacked_clipped.median(dim = 'time', keep_attrs = True)
            stacked_scaled = stacked_median * 0.0001
            
            return stacked_scaled
        
        def show_msi_image(image, title, bands):

            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(8, 5)

            image.sel(band = bands).rio.reproject('EPSG:4326').plot.imshow(ax = ax,
                                                                           vmin = 0.0,
                                                                           vmax = 0.3)

            plt.title(title)
            plt.xlabel('')
            plt.ylabel('')
            plt.yticks(rotation = 90, ticks = None)
            plt.xticks(ticks = None)

            plt.show()

            return st.pyplot(fig)
        
        def resample_image(image):

            st.badge('Resampling the image to 1m......', color = 'green')

            resampled = image.rio.reproject(

                str(crs),
                resolution = (1, 1),
                resampling = Resampling.bilinear,
            )

            return resampled
            
        def ndvi_calculator(image):

            st.badge('calculating NDVI from resampled image', color = 'yellow')

            redBand = image.sel(band = 'B04')
            nirBand = image.sel(band = 'B08')

            ndvi = (nirBand - redBand) / (nirBand + redBand)
            
            return ndvi.rio.write_crs(str(crs))

        def ndvi_visualiser(image):

            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(8, 5)

            im = ax.imshow(image, cmap = 'Greens', vmin = 0, vmax = 1)

            ax.yaxis.set_major_locator(MaxNLocator(nbins = 3))
            plt.colorbar(im, ax = ax)

            plt.title('NDVI Image')
            plt.colorbar()
            plt.legend()
            plt.yticks(rotation = 90, ticks = None)
            plt.xticks(ticks = None)           
            plt.xlabel('')
            plt.ylabel('')   

            plt.show()

        
        def threshold_calculator(ndvi_image):

            ndvi_values = ndvi_image.values.flatten()
            ndvi_values = ndvi_values[~numpy.isnan(ndvi_values)]
            mean = ndvi_values.mean()
            stdDev = ndvi_values.std()

            k = 1.0

            stats, p = scipy.stats.normaltest(ndvi_values)
            
            if p > 0.05:
                st.text(f'The value p value is {p}, Hence there is a Normal Distribution in NDVI')

                threshold = mean - k * stdDev
                st.text(f'Calculated Threshold based on Normal Distribution (z-Score): {threshold.item()}')
                return threshold

            else:

                st.text(f'The value of p is {p}, Hence the NDVI values is skewed')

                threshold = numpy.percentile(ndvi_values, 25)
                st.text(f'Calculated Threshold based on percentile: {round(threshold.item(), 2)}')
                return round(threshold.item(), 2)
        
        def binary_ndvi(ndvi_image, threshold, proj_crs = str(crs)):

            st.badge(f'Classifying the NDVI image based on threshold {threshold}', )

            binary_classification = xarray.where(ndvi_image >= threshold, 1, 0, keep_attrs = True)
            binary_image = binary_classification.rio.write_crs(proj_crs)

            return binary_image

        @st.fragment()
        def download_geojson():

            st.download_button(
                label = 'Download Final geoJson',
                data = roi_reprojected.to_json(),
                file_name = 'Final_Parcels_Status.geojson',
                mime="application/geo+json"
            ) 


        stacked = sentinel_extractor()

        resampled = resample_image(stacked)

        ndvi = ndvi_calculator(resampled)

        threshold = threshold_calculator(ndvi)

        ndviClasses = binary_ndvi(ndvi, threshold = threshold)

        ndvi_array = ndviClasses.values
        transform = ndviClasses.rio.transform()
        crs = ndviClasses.rio.crs

        stats = zonal_stats(
            vectors = roi_reprojected,
            raster = ndvi_array,
            affine = transform,
            stats = ['mean']
        )

        mean = []

        for i in stats:
            for keys, values in i.items():
                mean.append(values)

        roi_reprojected['mean'] = mean
        roi_reprojected['canopy cover percentage'] = roi_reprojected['mean'] * 100
        roi_reprojected['status'] = numpy.where(roi_reprojected['canopy cover percentage'] < 10, 'Accepted', 'Rejected')

        roi_reprojected

        download = download_geojson()


