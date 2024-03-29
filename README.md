# Prototype-TephraDB-Japan

All the scripts were written by Shimpei Uesawa@CRIEPI, Japan. 
(Last update: 26, April 2023, Please see tags for previous versions.)

This is the toolkit of drawing cumulative frequency curves and hazard curves for assessing the tephra fall hazard in thickness (mm) with tephra fall database in Japan. 
See the publication in detail (Uesawa et al., 2022, Journal of Applied Volcanology; https://appliedvolc.biomedcentral.com/articles/10.1186/s13617-022-00126-x). 

Preparation for processing:
1. Install R (Statistical Computing). Please refer to the website https://www.r-project.org/ for installation instructions. We recommend to use version 4.1.0 or earlier.

Workflow using the R script:
1. Copy the folder entitled "TephraDB_Prototype_ver1.4" to your computer (directly under "C:"). The dataset can be downloaded at [https://zenodo.org/record/7857457](https://doi.org/10.5281/zenodo.10846798)
2. Open the file entitled "Tephra_Hazard_Curve_Generator_120.r" with R.
3. Install all the libraries that you need. See the script.
4. Set the locality name and coordinate.
5. Edit Loc <- "Place name (default "Tokyo") " (line 2067) where you want to draw the prototype hazard curve.
6. Run the entire script of R.

cf.) Plotting with python for better scientific plots:
1. Install Python 3 (I recommend using Anaconda). Please refer to the website https://www.anaconda.com/ for installation instructions. 
2. Open the file entitled "Tephra_fall_Hazard_curve_Plotter_120.py" and revise the directory to your path using a text editor (line 14, 17, 36 and 39). Then, put the file in the "TephraDB_Prototype_ver1.3" folder.
4. Edit F = "Place name (default "Tokyo") " (line 41) where you want to draw the prototype hazard curve with spyder etc..
5. Run the entire script of Python.

Workflow using ArcGIS:
If you have ArcGIS Spatial Analyst Tools, you can generate the hazard curves using the Python Console of ArcGIS and view all the raster data of tephra distributions.
1. Copy the folder entitled "TephraDB_Prototype_ver1.4" to your computer. 
2. Open all the raster data and shapefiles ("PrefOffices.shp") with ArcGIS in the folder entitled "TephraDB_Prototype_ver1.3". Any locality shapefile can be used for the raster sampling.
3. Open the file entitled "Extract_MultiValues_TO_Points_ver2.py" in the folder entitled "For ArcGIS" using a text editor.
4. Copy and paste to Python Console of ArcGIS and run the script. If you use your locality shapefile, revise the PointFeature file name to your file name in the text.
5. Write down the attribute table of "PrefOffices.shp" as csv to the "TephraDB_Prototype_ver1.3" folder entitled "combinedPointValue_012.csv".
6. Open the file entitled "Tephra_fall_Hazard_curve_Plotter_120.py" and revise the directory to your path using a text editor.
11. Run the script with the Python Console of ArcGIS.

Disclaimer:
CRIEPI, the original data acquirer/creator, and the database administrator shall not be held liable for any loss or damage arising from the use of this database.

R is a free software environment for statistical computation and graphics (Visit at https://www.r-project.org/ for more detail ).

Author: Shimpei Uesawa (contact: uesawa<at>criepi.denken.or.jp) <at> -> @
Affiliation: Central Research Institute of Electric Power Industory, Nuclear Risk Research Center, Abiko, Chiba, Japan
Source Code License: GPL. Use at your own risk.

Reference summary:
  R-project: https://www.r-project.org/,
  Python: https://www.anaconda.com/ ,
  arcpy: https://desktop.arcgis.com/en/arcmap/latest/tools/spatial-analyst-toolbox/extract-multi-values-to-points.htm
