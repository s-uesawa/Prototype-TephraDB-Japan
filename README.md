# Prototype-TephraDB-Japan

All the scripts were written by Shimpei Uesawa@CRIEPI, Japan. 
(Last update: 16, July 2021, Please see tags for older version.)

This is the toolkit of drawing hazard curves for evaluating the tephra fall load hazard with tephra fall database in Japan. 
See the publication in detail (Uesawa et al., submitted, Journal of Applied Volcanology; https://dx.doi.org/10.21203/rs.2.23106/v1). The results at 47 prefectural offices are available at https://doi.org/10.5281/zenodo.3608350.

Preparation for processing:
1. Install R (Statistical Computing). Please refer to the website https://www.r-project.org/ for installation instructions.

Workflow using the Tephra database:
1. Copy the folder entitled "TephraDB_Prototype_ver1.1" to your computer. The dataset can be downloaded at http://doi.org/10.5281/zenodo.3608346
2. Open the file entitled "Tephra_Hazard_Curve_Generator_011.r" with R.
3. Install all the libraries that you need. See the script.
4. Revise the directory to the path of the folder that you copied.
5. Set the locality name and coordinate.
6. Run the entire script of R.

cf) Plotting with python for better scientific plots
1. Install Python 3 (I recommend using Anaconda). Please refer to the website https://www.anaconda.com/ for installation instructions. 
2. Open the file entitled "Tephra_fall_Hazard_curve_Plotter.py" and revise the directory to your path using a text editor. Then, put the file in the "TephraDB_Prototype_ver1.1" folder.
3. Edit F = "Place name " where you want to draw the prototype hazard curve with spyder etc in the python script.

Workflow using ArcGIS:
If you have ArcGIS Spatial Analyst Tools, you can generate the hazard curves using the Python Console of ArcGIS and view all the raster data of tephra distributions.
1. Copy the folder entitled "TephraDB_Prototype_ver1" to your computer. 
2. Open all the raster data and shapefiles ("PrefOffices.shp") with ArcGIS in the folder entitled "TephraDB_Prototype_ver1". Any locality shapefile can be used for the raster sampling.
3. Open the file entitled "Extract_MultiValues_TO_Points_ver2.py" in the folder entitled "For ArcGIS" using a text editor.
4. Copy and paste to Python Console of ArcGIS and run the script. If you use your locality shapefile, revise the PointFeature file name to your file name in the text.
5. Write down the attribute table of "PrefOffices.shp" as csv to the "TephraDB_Prototype_ver1" folder entitled "combinedPointValue.csv".
6. Open the file entitled "Tephra_fall_Hazard.py" and revise the directory to your path using a text editor.
7. Run the script with the Python Console of ArcGIS.
