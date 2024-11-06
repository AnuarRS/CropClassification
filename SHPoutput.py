import geopandas as gpd
from shapely.geometry import Point

predictions = [
    'пшеница', 'ячмень', 'овес', 'подсолнечник','рис', 
]

points = []
classes = []

for idx, pred in enumerate(predictions):
   
    x = idx % 10  
    y = idx // 10 
    
    point = Point(x, y) 
    points.append(point)
    classes.append(pred)

gdf = gpd.GeoDataFrame({'class': classes, 'geometry': points}, crs="EPSG:4326")

output_shp = '/content/drive/MyDrive/classification_points_result.shp'
gdf.to_file(output_shp)

print("Shapefile:", output_shp)
