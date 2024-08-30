.. |tuto_shapefile_cosine| image:: _static/animation_tuto_shapefile_cosine_sim.gif
    :alt: Description of the image
    :width: 500px
    :align: middle

.. |tuto_shapefile_RF| image:: _static/animation_tuto_RF.gif
    :alt: Description of the image
    :width: 400px
    :align: middle



Tutorials
===========

This section aims to provide the user with a few useful tutorials. The version of QGIS used is long term version 3.34.3, please 
advise us is problems arise with other versions.

Create a shapefile for cosine similarity
-----------------------------------------
To create a shapefile to compute the cosine similarity of an image relatively to those points, you have to go to ``Layer --> create Layer --> new shapefile Layer`` in
the QGIs toolbar. Then a window should open. You can name your file like you want, please use ``File encoding : UTF-8`` and select ``Geometry Type : Point``.
You can then click ``Ok`` to close the window and create your shapefile.
You can then update your shapefile by clicking ``Toogle editing`` and ``Add point feature`` in the toolbar of QGIS. 
You can then add your point.

This animation shows how to do it :
    |tuto_shapefile_cosine|

Using some points can be good, to ensure that you are not considering a very specific patch or pixel. However, as a mean is computed between 
all the points you place, if you use too much you may confuse the model by losing the specificity of those points.
We recommend tipycally to use a nomber of points between 1 and 10.





Create a shapefile for Random Forest 
--------------------------------------
To create a shapefile to train a random forest algorithm, you have to go to ``Layer --> create Layer --> new shapefile Layer`` in
the QGIs toolbar. Then a window should open. You can name your file like you want, please use "File encoding : UTF-8" and select "Geometry Type : Point".
You can then suppress "id" in the field list. You then need to add the column name you want the random forest to use for training and predicting.
For that, in "new field" type your name, and the click to "add to field list" (please Type : Text string). A name we recommend to use is "Type", as this is the default value
of the random forest algorithm. However you can use an another name but you will have to change the "Name of the column you want random forest to work on" in 
the random forest interface.

You can then click "Ok" to close the window and create your shapefile.
You can then update your shapefile by clicking "Toogle editing" and "Add point feature" in the toolbar of QGIS. 
You can then add your point with a Type of your choice.

This animation shows how to add some points to use random forest algorithm :
    |tuto_shapefile_RF|

We recommend to place at least 100 points for training data set. The more points you use for test data set the more robust the interpretation will be.
If you want to use only one dataset (will be split into 80% training points and 20% testing points) please place around 150 points.

Keep in mind that the more points you place the better the results will be.





Using a new encoder
--------------------
If you use Windows and you want to use a new encoder, you first need to create it in the OSGEO4W shell.
For that, you can open OSGEO4W shell and run the following commands (assuming that "name" is the name of your encoder of choice) :
::
    python
    import timm
    #print(timm.__version__) if you want to see which timm version you are currently using
    model = timm.create_model(name, pretrained=True)
    print(model)  #to verify it was indeed created
    exit()

After those lines of code you should be able to use the new model by going into the "backbone choice" option in the encoder interface and typing
the name of the model you want to use.

As timm is a library regularly updated, be sure that the version you use is compatible with the model you want to use.



Changing the parameters of the Random Forest
-----------------------------------------------
If you want to change the parameters of the random forest (such as the random state, number of trees, etc...), go to the random_forest.py file of this plugin.

You can then go to the line 278 and 292 of this code which should look this :
::
    rf_classifier = RandomForestClassifier(n_estimators=100, min_samples_split=4, random_state=42)

You can then change the parameters of the random forest freely. Please make sure to use the same parameters in both of those two lines of code
to avoid confusion when interpretating the results.
