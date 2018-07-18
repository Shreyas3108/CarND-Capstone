from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

class TLClassifier(object):
    def __init__(self):
        self.imgcount = 0
        #TODO load classifier
        'SSD Mobile requires TF higher than 1.3'
#         PATH_TO_CKPT = "light_classification/tf_models/ssd_mobilenet_v1_coco_2017_11_08/frozen_inference_graph.pb"
        PATH_TO_CKPT = "light_classification/tf_models/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb"
#         PATH_TO_CKPT = "tf_models/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb"
#         PATH_TO_CKPT = "tf_models/faster_rcnn_resnet101_coco_11_06_2017/frozen_inference_graph.pb"
        PATH_TO_CKPT = "light_classification/tf_models/faster_rcnn_resnet101_coco_11_06_2017/frozen_inference_graph.pb"
        self.detection_graph = tf.Graph()
        with self. detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                print("REading Graph")
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                print(".Done")

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                
                image_np_expanded = np.expand_dims(image, axis=0)
                # Actual detection.
                print("Detection ---")
                (boxes, scores, classes, num) = sess.run(
                  [detection_boxes, detection_scores, detection_classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})
                
                
                    
                print("--")
                print(boxes)
                print("--")
                print(classes)
                print("--")
                print(scores)
                
                light_boxes = boxes[classes == 10.]
                
                if len(light_boxes) > 0: 
                    box = light_boxes[0]
                    print("Using box ")
                    print(box)
                    
                    img_h = image.shape[0]
                    img_w = image.shape[1]
                    print(img_w) 
                    print(img_h)
                    (x1,y1,x2,y2) = int(img_w*box[1]), int(img_h*box[0]),int(img_w*box[3]), int(img_h*box[2])
                    print("Dim " + str(x1) + ","+str(y1)+","+str(x2)+","+str(y2))
                    
                    im = Image.fromarray(image)
                    im.save("tl_a_"+str(self.imgcount)+".png")
                    im = im.crop((x1,y1,x2,y2))
                    
                    tl_im = self.load_image_into_numpy_array(im)
                    tl_im =  cv2.cvtColor(tl_im, cv2.COLOR_RGB2HSV)

                    G_MIN = np.array([100/2, 160, 160], np.uint8)
                    G_MAX = np.array([190/2, 255, 255], np.uint8)
                    
                    dst = cv2.inRange(tl_im, G_MIN, G_MAX)
                    no_green = cv2.countNonZero(dst)
                    im.save("tl_b_"+str(self.imgcount)+"_"+str(no_green)+".png")
                    cv2.imwrite("tl_b_"+str(self.imgcount)+"_"+str(no_green)+"_ir.png",dst)
                    self.imgcount = self.imgcount + 1
                    
                    'if we are not sure, we return red for safety reasons'
                    if(no_green > 30):
                        return TrafficLight.GREEN 
                    else:
                        return TrafficLight.RED
                 
        #TODO implement light color prediction
        return TrafficLight.UNKNOWN
#         return 0
    
    def load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

if __name__ == '__main__':
    x = TLClassifier()
#    print(x.detection_graph)
    image = Image.open("test_images/carla_green.jpg")
    image_np = x.load_image_into_numpy_array(image)
    x.get_classification(image_np)
    
    

