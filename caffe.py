import cv2
import numpy as np

def load_model(proto_text_path, model_weights_path):
    """ Load the pre-trained FCN model using OpenCV's DNN module. """
    net = cv2.dnn.readNetFromCaffe(proto_text_path, model_weights_path)
    return net

def preprocess_input(image, dim =(500, 500)):
    """ Preprocess the input image for the FCN model. """
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(image.shape[1], image.shape[0]),
                                 mean=(104.00698793, 116.66876762, 122.67891434),
                                 swapRB=False, crop=False)
    return blob

def postprocess_output(output, image_shape):
    """ Convert the output of the network to a segmentation map. """
    # Output is (N, C, H, W) - we take the first result of the batch
    output = output[0]
    output = output.transpose(1, 2, 0)  # H, W, C
    output = cv2.resize(output, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC)
    print('resized', output)
    output = np.argmax(output, axis=2)
    print('argmaxed',output)
    return output

def visualize_segmentation(segmentation_map, num_classes=21):
    """ Visualize the segmentation map with random colors for each class. """
    colors = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
    vis_image = colors[segmentation_map]
    return vis_image

def main(image_path, proto_text_path, model_weights_path):
    # Load the model
    net = load_model(proto_text_path, model_weights_path)

    # Read and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image,(620,480))
    print(image.shape)
    if image is None:
        print("Error: Image not found.")
        return

    input_blob = preprocess_input(image)

    # Set the input to the network
    net.setInput(input_blob)

    # Perform forward pass and get the output
    output = net.forward()

    # Postprocess the output to get the segmentation map
    segmentation_map = postprocess_output(output, image.shape)

    # Visualize the segmentation results
    vis_image = visualize_segmentation(segmentation_map)

    # Display the images
    cv2.imshow("Input Image", image)
    cv2.imshow("Segmentation", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = '/home/ankan_opencv/officework/VS_Code/caffe-dnn/dog.jpg'
    proto_text_path = '/home/ankan_opencv/officework/VS_Code/caffe-dnn/fcn8s-heavy-pascal.prototxt'
    model_weights_path = '/home/ankan_opencv/officework/VS_Code/caffe-dnn/fcn8s-heavy-pascal.caffemodel'
    main(image_path, proto_text_path, model_weights_path)