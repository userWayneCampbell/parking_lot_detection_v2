# Parking Lot Detection
This is a refactored version of my senior design project from ohio northern university.

It uses an already trained neural net (located in tf_files), and opencv to detect cars and
display results once the image is run throught the neural network. 

This version also displays these results using a simple flask server.

Potentially this will be running at my current work location, which has electric
car hookups.

## Usage
Edit csv/1.csv with the coordinates of the lots you want to observe in your parking
lot. (Using numpy notation for cropping)

### Run the tensorflow/opencv application
`./main.py`

### Run the flash application
`./run_flask.py`
(This shows the results of the tensorflow application on port 5000.)
