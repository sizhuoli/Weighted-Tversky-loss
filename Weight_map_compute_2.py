#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
import matplotlib.pyplot as plt
import geopandas as gps
import pandas as pd

def create_boundary(polygon_file,scale_polygon = 1.5):    
    """compute weight map from polygons"""
    new_c = []
    #for each polygon in area, scale with a factor, then compare with other extended polygons
    for i in tqdm(range(len(polygon_file))):
        #extend the selected polygon
        pol1 = gps.GeoSeries(polygon_file.iloc[i][0])
        sc = pol1.scale(xfact=scale_polygon, yfact=scale_polygon, zfact=scale_polygon, origin='center')
        scc = pd.DataFrame(columns=['id', 'geometry'])
        scc = scc.append({'id': None, 'geometry': sc[0]}, ignore_index=True)
        #repeat the extended polygon certain times to match the total number of polygons for element-wise comparison
        scc = gps.GeoDataFrame(pd.concat([scc]*len(polygon_file), ignore_index=True))
        #extend other polygons as well and then take the intersection
        pol2 = gps.GeoDataFrame(polygon_file[~polygon_file.index.isin([i])])
        pol2 = gps.GeoDataFrame(pol2.scale(xfact=scale_polygon, yfact=scale_polygon, zfact=scale_polygon, origin='center'))
        pol2.columns = ['geometry']
        #intersection
        ints = scc.intersection(pol2)
        #collect intersection polygons from ints
        for k in range(len(ints)):
            if ints.iloc[k]!=None:
                if ints.iloc[k].is_empty !=1:
                    new_c.append(ints.iloc[k])
    #data frame for intersection polygons
    new_c = gps.GeoSeries(new_c)
    new_cc = gps.GeoDataFrame({'geometry': new_c})
    #plot the original polygons and the intersections
    fig, ax = plt.subplots(figsize = (10,10))
    new_cc.plot(ax=ax,color = 'red')
    polygon_file.plot(alpha = 0.2,ax = ax,color = 'b')
    plt.show()
    new_cc.columns = ['geometry']
    #subtract original polygons from intersection to get the boundaries
    c = gps.overlay(new_cc,polygon_file,how = 'difference')
    return c

def main():
    training_polygon_fn = './polygons.shp'
    training_polygon = gps.read_file(training_polygon_fn)
    #load polygons as a dataframe
    training_polygon1 = gps.GeoDataFrame(training_polygon['geometry'])
    #create boundary polygons
    boundary = create_boundary(training_polygon1,scale_polygon = 1.5)
    #plot the final boundaries
    fig, ax = plt.subplots(figsize = (10,10))
    boundary.plot(ax=ax,color = 'red')
    training_polygon1.plot(alpha = 0.2,ax = ax,color = 'b')
    plt.show() 
    
if __name__ == '__main__':
    main()
    