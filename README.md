# Prototype-TephraDB-Japan

All the scripts were written by Shimpei Uesawa@CRIEPI, Japan. 
(Last update: 07, February 2020)

This is the toolkit of drawing hazard curves for evaluating the tephra fall load hazard with tephra fall database in Japan. 
See the publication in detail (Uesawa et al., submitted, Journal of Applied Volcanology; doi: in preparation). The results at 47 prefectural offices are available at https://doi.org/10.5281/zenodo.3608350.

I recommend running the program with Windows (32bit).

Preparation for processing:

1. Install Python 3 (I recommend using Anaconda). Please refer to the website https://www.anaconda.com/ for installation instructions. 
2. Install R (Statistical Computing). Please refer to the website https://www.r-project.org/ for installation instructions.

Workflow using the Tephra database:
1. Copy the folder entitled "TephraDB_Prototype_ver1" to your computer. The dataset can be downloaded at http://doi.org/10.5281/zenodo.3608346
2. Open the file entitled "Tephra_Hazard_Curve_Generator.r" with R.
3. Install all the libraries that you need. See the script.
4. Revise the directory to the path of the folder that you copied.
5. Revise the directory to the path of python.exe.
6. Set the locality name and coordinate.
7. Open the file entitled "Tephra_fall_Hazard.py" and revise the directory to your path using a text editor. Then, put the file in the "TephraDB_Prototype_ver1" folder.
8. Run the entire script of R. (If you can not run the python with R, you can run the python script edited F = "Place name " where you want to draw the hazard curve with spyder etc.)

Workflow using ArcGIS:
If you have ArcGIS Spatial Analyst Tools, you can generate the hazard curves using the Python Console of ArcGIS and view all the raster data of tephra distributions.
1. Copy the folder entitled "TephraDB_Prototype_ver1" to your computer. 
2. Open all the raster data and shapefiles ("PrefOffices.shp") with ArcGIS in the folder entitled "TephraDB_Prototype_ver1". Any locality shapefile can be used for the raster sampling.
3. Open the file entitled "Extract_MultiValues_TO_Points_ver2.py" in the folder entitled "For ArcGIS" using a text editor.
4. Copy and paste to Python Console of ArcGIS and run the script. If you use your locality shapefile, revise the PointFeature file name to your file name in the text.
5. Write down the attribute table of "PrefOffices.shp" as csv to the "TephraDB_Prototype_ver1" folder entitled "combinedPointValue.csv".
6. Open the file entitled "Tephra_fall_Hazard.py" and revise the directory to your path using a text editor.
7. Run the script with the Python Console of ArcGIS.
