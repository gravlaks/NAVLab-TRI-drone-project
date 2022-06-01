import sys
from datetime import date, datetime, timedelta
from tracemalloc import start
from tqdm import tqdm
# setting path
sys.path.append('../src')
from os.path import exists
import numpy as np
from apriltag_tools.Detector import Detector 
from cars.Car import Car
from cars.relative_distances import get_ratio, get_relative_distance
from dynamics_models.CV import CV
from plotting.rel_dists import  plot_relative_distances
from dynamics_models.CT import CT
import matplotlib.pyplot as plt
import cv2
import rosbag
import cv_bridge
import imageio
from cv2_tools.cv2utils import draw_center, draw_relative




def get_rel_distances(bag_filename, MAX_TIME, visualize=True):
    bag = rosbag.Bag(bag_filename)
    cap = cv2.VideoCapture(2)

    rel_dists = []

    feature_det = cv2.SIFT_create()
    car1 = Car(2, "tag16h5", CV(), feature_det=feature_det)
    car2 = Car(1, "tag16h5", CV(), feature_det=feature_det)
    apriltag_and_sift=True
    only_apriltag = True

    dt = 0.01
    start_time = datetime.now()
    camera = '/usb_cam/image_raw'
    bridge = cv_bridge.CvBridge()
    t2 = start_time
    final_images = []
    for topic, msg, t in bag.read_messages(topics=['/usb_cam/image_raw',
                                 '/vrpn_client_node/RigidBody01/pose', 
                                 '/vrpn_client_node/RigidBody02/pose']):
        
        t1 = datetime.now()
        if t1-t2<timedelta(seconds=dt):
            continue 
        if topic == camera:        
            #img = msg.data
            img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            #cv2.imshow('frame', img)

        else:
            continue
        if cv2.waitKey(1) == ord('q'):
            break
        if t1-start_time >= timedelta(seconds=MAX_TIME):
            break
        car1.predict(dt)
        car2.predict(dt)
        

        
        detect1, detect2 = False, False
        if apriltag_and_sift or (car1.sift_tag is None or car2.sift_tag is None):
            
            if car1.sift_tag is None or car2.sift_tag is None:
                
                detector = Detector(img=img)
                detections = detector.detect(turn_binary=True, units=3, visualize=False, tag_family=car1.tag_family)
                print(len(detections))
                print(detections)
                detect1 = car1.update_state(detections, img, dt)
                detect2 = car2.update_state(detections, img, dt)
            else:
                detect1 = car1.update_state_apriltag(img, dt,  units=2, tag_family=car1.tag_family)
                detect2 = car2.update_state_apriltag(img, dt,  units=2, tag_family=car1.tag_family)
            #   

            

        if not (detect1 and detect2):
            

            if not (car1.sift_tag is None or car2.sift_tag is None or only_apriltag): 
                
                if not detect1:
                    detect1 = car1.update_state_sift(img, dt)
                if not detect2:
                    detect2 = car2.update_state_sift(img, dt)
            

        
        t2 = datetime.now()
        
        print("Det1 and 2: ", detect1, detect2)

        if detect1 and detect2:
            print(car1.get_center(), car2.get_center())
            rel_dist = get_relative_distance(car1, car2)
            rel_dists.append(rel_dist)
            print(rel_dist)
            #raise Exception
        if visualize:
            img = draw_relative(img, [car1, car2])
            final_images.append(img)
            cv2.imshow("frame", img)

        print(t2-t1)
    # print("One: ", one_detections, "avg: ", one_detections/idx)
    rel_dists = np.array(rel_dists).squeeze()
    print(rel_dists.shape)
    if visualize:
        imageio.mimsave('../plots/gifs/flightroom1.gif', final_images, fps=60)
    np.save("../data/flightroom_traj/flightroom1", rel_dists)
    return rel_dists





    
if __name__ == '__main__':
    rel_dist_filepath="../plots/flightroom1.png"
    bag_filename = '/home/torstein/Stanford/nav_lab/flightroom/2022-05-27-11-38-06.bag'

    
    rel_dists = get_rel_distances(bag_filename, MAX_TIME=50)
  
    plot_relative_distances(rel_dists, filepath=rel_dist_filepath)
    print(rel_dists.shape)
    #plot_locations(np.array([d.center for d in det1s]), np.array([d.center for d in det2s]))
    #plot_imgs(images, det1s, det2s)