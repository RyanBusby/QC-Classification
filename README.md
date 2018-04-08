# __Binary Classification__


[<img src="img/logo.png" style="width: 5px;"/>](https://www.kaggle.com/c/bosch-production-line-performance/data) ↖  __the data__

#### The goal is to predict which parts will fail quality control to reduce manufacturing failures.

- The data represents measurements of parts as they move through Bosch's production lines.

- They are very sparse: *81% empty.* The classes are highly imbalanced: *0.58% failed.*

1. [Cluster based on shared features](clustering.ipynb)
2. [Perform principal component analysis on each cluster](pca.ipynb)
3. [Use various classification algorithms on each cluster](training.ipynb)
