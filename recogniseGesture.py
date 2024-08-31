#imports
import numpy as np
import cv2
import os
import pickle
import imagePreprocessingUtils as ipu


CAPTURE_FLAG = False

class_labels = ipu.get_labels()
                

def recognise(cluster_model, classify_model):
    global CAPTURE_FLAG
    gestures = ipu.get_all_gestures()
    cv2.imwrite("all_gestures.jpg", gestures)
    camera = cv2.VideoCapture(0)
    print('Now camera window will be open, then \n1) Place your hand gesture in ROI (rectangle) \n2) Press esc key to exit.')
    count = 0
    while(True):
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame from camera.")
            break
        
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, ipu.START, ipu.END, (0, 255, 0), 2)
        cv2.imshow("All_gestures", gestures)
        pressedKey = cv2.waitKey(1)
        if pressedKey == 27:
            break
        elif pressedKey == ord('p'):
            CAPTURE_FLAG = not CAPTURE_FLAG
        
        if CAPTURE_FLAG:
            # Region of Interest
            roi = frame[ipu.START[1]+5:ipu.END[1], ipu.START[0]+5:ipu.END[0]]
            if roi is not None:
                print("ROI extracted")
                roi = cv2.resize(roi, (ipu.IMG_SIZE, ipu.IMG_SIZE))
                img = ipu.get_canny_edge(roi)[0]
                cv2.imshow("Edges", img)
                sift_disc = ipu.get_SIFT_descriptors(img)
                print(f"SIFT Descriptors: {sift_disc}")

            if sift_disc is not None:
                visual_words = cluster_model.predict(sift_disc)
                print(f"Visual Words: {visual_words}")
                bovw_histogram = np.array(np.bincount(visual_words, minlength=ipu.N_CLASSES * ipu.CLUSTER_FACTOR))
                print(f"BoVW Histogram: {bovw_histogram}")
                pred = classify_model.predict([bovw_histogram])
                label = class_labels[pred[0]]
                print(f"Predicted Label: {label}")

                # Draw a filled rectangle as background for the text
                cv2.rectangle(frame, (ipu.START[0], ipu.START[1]-40), (ipu.END[0], ipu.START[1]), (0, 0, 0), cv2.FILLED)
                # Display the predicted gesture on the frame
                frame = cv2.putText(frame, f"Detected: {label}", (ipu.START[0]+10, ipu.START[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
       
        cv2.imshow("Video", frame)
    
    camera.release()
    cv2.destroyAllWindows()


  
    
clustering_model = pickle.load(open('mini_kmeans_model.sav', 'rb'))    
classification_model = pickle.load(open('svm_model.sav', 'rb'))
recognise(clustering_model,classification_model)


