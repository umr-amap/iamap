# Tutorials

This section aims to provide the user with a few useful tutorials. The version of QGIS used is long term version 3.34.3, please 
advise us is problems arise with other versions.

## Create a shapefile for cosine similarity

To create a shapefile to compute the cosine similarity of an image relatively to those points, you have to go to ``Layer --> create Layer --> new shapefile Layer`` in
the QGIs toolbar. Then a window should open. You can name your file like you want, please use ``File encoding : UTF-8`` and select ``Geometry Type : Point``.
You can then click ``Ok`` to close the window and create your shapefile.
You can then update your shapefile by clicking ``Toogle editing`` and ``Add point feature`` in the toolbar of QGIS. 
You can then add your point.

Using some points can be good, to ensure that you are not considering a very specific patch or pixel. However, as a mean is computed between 
all the points you place, if you use too much you may confuse the model by losing the specificity of those points.
We recommend tipycally to use a nomber of points between 1 and 10.





## Create a shapefile for Random Forest 

To create a shapefile to train a random forest algorithm, you have to go to ``Layer --> create Layer --> new shapefile Layer`` in
the QGIs toolbar. Then a window should open. You can name your file like you want, please use "File encoding : UTF-8" and select "Geometry Type : Point".
You can then suppress "id" in the field list. You then need to add the column name you want the random forest to use for training and predicting.
For that, in "new field" type your name, and the click to "add to field list" (please Type : Text string). A name we recommend to use is "Type", as this is the default value
of the random forest algorithm. However you can use an another name but you will have to change the "Name of the column you want random forest to work on" in 
the random forest interface.

You can then click "Ok" to close the window and create your shapefile.
You can then update your shapefile by clicking "Toogle editing" and "Add point feature" in the toolbar of QGIS. 
You can then add your point with a Type of your choice.

We recommend to place at least 100 points for training data set. The more points you use for test data set the more robust the interpretation will be.
If you want to use only one dataset (will be split into 80% training points and 20% testing points) please place around 150 points.

Keep in mind that the more points you place the better the results will be.

