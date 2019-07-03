#!/usr/bin/python
import csv
import cv2
import tensorflow as tf
import time
import imutils

def main():
    CAMERA_INPUT = 0
    model_file = "tf_files/retrained_graph.pb"
    label_file = "tf_files/retrained_labels.txt"
    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    graph = load_graph(model_file)
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    # Setup opencv video capture
    vc = cv2.VideoCapture(CAMERA_INPUT)
    time.sleep(2)

    flask_output = ""

    if (vc.isOpened() == False):
        print("Error opening video file or stream")
        return False

    while(True):
        # Grab new frame
        rval, frame = vc.read()

        with open('out.txt', 'w') as f:
            f.write(flask_output)

        # Reset flask string
        flask_output = ""


        # Open CSV file for vehicle locations
        with open('csv/' + '1.csv', 'r') as np:
            readerOfCSVData = csv.reader(np, delimiter=',')

            # Interate through car locations
            for row in readerOfCSVData:

                # Crop
                new_frame = frame[int(row[1]):int(row[2]), int(row[3]):int(row[4])]

                # For testing... Display
                cv2.imshow("frame" + str(row[0]), new_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Start TF session
                with tf.Session(graph=graph) as sess:
                    try:
                        cropped_image1 = cv2.resize(new_frame, (224, 224))
                        one_dimension = cropped_image1.reshape(1,
                                        cropped_image1.shape[0],
                                        cropped_image1.shape[1],
                                        cropped_image1.shape[2])
                        #Start Tensorflow Session
                        with sess.as_default():
                        	tensor = tf.constant(one_dimension)

                        resized = tf.image.resize_bilinear(tensor, [input_height, input_width])
                        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
                        with sess.as_default():
                        	result = sess.run(normalized)
                        results = sess.run(output_operation.outputs[0],{input_operation.outputs[0]: result})

                        #Debug print
                        resultList = results.tolist()

                        # Save outut to file
                        output = str(row[0]) + ' Car Prediction: ' + str(resultList[0][0])
                        print(output)
                        flask_output += output + '\n'
                    except RuntimeError:
                        print("[INFO] caught a RuntimeError")

def load_graph(model_file):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(model_file, "rb") as f:
                graph_def.ParseFromString(f.read())
        with graph.as_default():
                tf.import_graph_def(graph_def)

        return graph

def load_labels(label_file):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
                label.append(l.rstrip())
        return label

#Crops image
def crop_image(this_image, r0,r1,r2,r3):
        if r0 > r2:
                x1 = r2
                x2 = r0
        else:
                x1 = r0
                x2 = r2
        if r1 > r3:
                y1 = r3
                y2 = r1
        else:
                y1 = r1
                y2 = r3
        #print("opencv display values: " + y1 + " " + y2 + " " + x1 + " " + x2)
        crop_img = this_image[int(y1) : int(y2),int(x1) : int(x2)]
        #cv2.imshow("name", crop_img)
        return crop_img

if __name__ == "__main__":
    main()
