# optimizers-in-pyspark

This project demonstrates the implementation of many gradient based optimizers like AdaGrad, AdaDelta, Adam,etc. in pyspark with the help of mapreduce jobs.

Running the main file(driver.py) requires a successful spark setup on the system.

Steps to run:

1) spark-submit driver.py

On running the above command, you will see the logs of map-reduce jobs being written into the terminal. After half an hour, it would show the loss curve of all the gradient based optimizers.

