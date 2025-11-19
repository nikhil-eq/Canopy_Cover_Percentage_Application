import pystac_client
import planetary_computer
import localtileserver
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
import xarray as xr
import xrspatial
import stackstac
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from scipy import stats
import rasterstats
import streamlit as st
from datetime import datetime, date
from rasterstats import zonal_stats
from matplotlib.ticker import MaxNLocator
import leafmap.foliumap as leafmap
import os

os.environ['LOCALTILESERVER_CLIENT_PREFIX'] = 'proxy/{port}'

st.title('☘ Canopy Cover Percentage Detection ☘')

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
st.text('2. Enter the EPSG Code for you ROI (e.g., EPSG:32644) - Only Projected CRS')
st.text('2. Select the Date Range (Start Date and End Date))')
st.text('3. Download your Final File (GeoJSON)')

roi = st.file_uploader(
    label = 'Upload Your File Here!',
    type = ['gpkg', 'KML', 'geojson'],
    accept_multiple_files = False,
)

if roi is not None:

    try: 
        roi = gpd.read_file(roi)

        st.success(
        body = 'Uploaded Successfully', 
        icon = '✔️')

        st.dataframe(roi)

        roi_geometry = roi.geometry.union_all().__geo_interface__

        crs_input = st.text_input(
            label = 'Please Enter EPSG Code suitable for your Region of Interest (eg., EPSG:32644)',
            placeholder = 'EPSG:XXXX',
            help = 'Provide a projected CRS suitable for your region'
            )
        
        if not crs_input.startswith('EPSG:'):
            st.error('Please enter a valid EPSG code., e.g., EPSG:32644')
        else:
            crs = crs_input
            
            roi_reprojected = roi.to_crs(str(crs))

            startDate = st.date_input(
                label= 'Select Start Date',
                value = 'today',
                format = 'YYYY-MM-DD'
            )

            endDate = st.date_input(
                label= 'Select End Date',
                min_value = startDate,
                value = 'today',
                format = 'YYYY-MM-DD'
            )

            dateRange = str(startDate) + '/' + str(endDate)

            st.info(f'Selected Date Range is {startDate} to {endDate}')

            if startDate > endDate:
                st.error('End date must be after start date.')
                
            elif startDate != endDate:
                
                def sentinel_extractor(cloud_cover = 20, proj_crs = str(crs)):

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
                    st.success(f'Found {len(items)} items from Sentinel 2 L2A (SR)')

                    if not items:
                        st.warning('No images found. Try adjusting date range')
                        return None
                    
                    bandNames = ['B02', 'B03', 'B04', 'B08']

                    epsg_code = proj_crs.split(':')
                    epsg_code = int(epsg_code[1])

                    stacked = stackstac.stack(items, assets = bandNames, epsg = epsg_code, chunksize = 1024,)
                    stacked_clipped = stacked.rio.clip(roi_reprojected.geometry, roi_reprojected.crs, drop = True)
                    stacked_median = stacked_clipped.median(dim = 'time', keep_attrs = True)
                    stacked_scaled = stacked_median * 0.0001
                    
                    return stacked_scaled
                
                stacked_data = sentinel_extractor()

                if stacked_data is not None:

                    st.badge('🔄 Resampling to 1m resolution...')

                    resampled_data = stacked_data.rio.reproject(
                        str(crs),
                        resolution = (1, 1),
                        resampling = Resampling.bilinear
                    )

                    st.badge('🧮 Calculating NDVI...')
                    red = resampled_data.sel(band='B04')
                    nir = resampled_data.sel(band='B08')
                    ndvi = (nir - red) / (nir + red)
                    ndvi_data = ndvi.rio.write_crs(str(crs))

                    st.badge('📊 Calculating adaptive threshold...')
                    def calc_threshold(ndvi_img):
                        values = ndvi_img.values.flatten()
                        values = values[~np.isnan(values)]
                        if len(values) == 0:
                            st.error('No valid NDVI values found.')
                            return None
                        
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        _, p_val = stats.normaltest(values)
                        
                        if p_val > 0.05:
                            st.success(f'Normal distribution detected (p={p_val:.4f}). Using z-score.')
                            threshold = mean_val - 1.0 * std_val  # k=1.0
                            st.info(f'Threshold: {threshold:.4f}')
                        else:
                            st.warning(f'Skewed distribution (p={p_val:.4f}). Using 25th percentile.')
                            threshold = np.percentile(values, 25)
                            st.info(f'Threshold: {threshold:.4f}')
                        
                        return round(threshold.item(), 2)
                    
                    threshold_data = 0.30 #calc_threshold(ndvi_data)

                    if threshold_data is not None:

                        st.info(f'🔀 Binary classification (NDVI >= {threshold_data:.4f} = 1)')
                        binary = xr.where(ndvi >= threshold_data, 1, 0, keep_attrs=True)
                        binary = binary.rio.write_crs(str(crs))
                        
                        st.info('📈 Computing zonal statistics...')
                        binary_array = binary.values
                        transform = binary.rio.transform()
                        raster_crs = binary.rio.crs
                        
                        stats = zonal_stats(
                        vectors = roi_reprojected,
                        raster = binary_array,
                        affine = transform,
                        stats = ['mean']
                        )

                        mean = []

                        for i in stats:
                            for keys, values in i.items():
                                mean.append(values)

                        roi_reprojected['mean'] = mean
                        roi_reprojected['canopy cover percentage'] = roi_reprojected['mean'] * 100
                        roi_reprojected['status'] = np.where(roi_reprojected['canopy cover percentage'] < 10, 'Accepted', 'Rejected')

                        st.header('Results')
                        st.dataframe(roi_reprojected[['canopy cover percentage', 'status']])

                        m = leafmap.Map()
                        m.add_basemap('HYBRID', show = True)
                        m.add_gdf(roi_reprojected, 'ROI',
                                  zoom_to_layer = True,
                                  info_mode = "on_over",
                                  edgecolor = 'black',
                                  fillcolor = None)
                        
                        m.add_raster(ndvi_data, colormap = 'RdYlGn', 
                                     nodata = np.nan,
                                     layer_name = 'NDVI')


                        m.to_streamlit()
            
                        @st.fragment()
                        def download_geojson():

                            st.download_button(
                                label = '📥 Download Final GeoJSON',
                                data = roi_reprojected.to_json(),
                                file_name = f'canopy_analysis_{startDate}_to_{endDate}.geojson',
                                mime = "application/geo+json"
                            )   

                        download = download_geojson()

                    col1, col2 = st.columns(2)

                    with col1:
                        def plot_rgb(image, title):
                            fig, ax = plt.subplots(figsize = (8, 5))
                            rgb = image.sel(band = ['B04', 'B03', 'B02']).rio.reproject('EPSG:4326')
                            rgb.plot.imshow(ax = ax, vmin = 0, vmax = 0.3)
                            ax.set_title(title)
                            ax.set_xlabel('')
                            ax.set_ylabel('')
                            ax.set_xticks([])
                            ax.set_yticks([])
                            st.pyplot(fig)

                    with col2:
                        def plot_ndvi(ndvi_img):
                            fig, ax = plt.subplots(figsize=(8, 5))
                            im = ax.imshow(ndvi_img, cmap='RdYlGn', vmin=ndvi_img.values.flatten().min(), vmax=ndvi_img.values.flatten().max())
                            ax.set_title('NDVI Image')
                            ax.set_xlabel('')
                            ax.set_ylabel('')
                            ax.set_xticks([])
                            ax.set_yticks([])
                            plt.colorbar(im, ax=ax)
                            st.pyplot(fig)

    except Exception as e:
        st.error(f'Error processing file: {str(e)}')

else:
    st.info('👆 Please upload a file to get started.')







