__author__ = 'Iacopo'
import renderer
import facial_feature_detector as feature_detection
import camera_calibration as calib
import scipy.io as io
import cv2
import numpy as np
import os
import check_resources as check
import matplotlib.pyplot as plt
import sys
import myutil
import ThreeD_Model
import config

this_path = os.path.dirname(os.path.abspath(__file__))

## 3D Models we are gonna use to to the rendering {0, -40, -75}
pose_models = ['model3D_aug_-00_dense']#,'model3D_aug_-40','model3D_aug_-75',]
## In case we want to crop the final image for each pose specified above/
## Each bbox should be [tlx,tly,brx,bry]
crop_models = [None,None,None]       
#crop_models = [[23,0,23+125,160],[0,0,210,230],[0,0,210,230]]


def demo():
    opts = config.parse()
    nSub = opts.getint('general', 'nTotSub')
    cnnSize = opts.getint('general', 'cnnSize')
    #exit(1)
    fileList, outputFolder, frtOutputFile,\
    hpOutputFile, fpOutputFile = myutil.parse(sys.argv)
    ## Making sure outputfolder exits
    #myutil.mymkdir( outputFolder )
	##Opening files if needed
    if frtOutputFile is not None:
    	frtOutput = open(frtOutputFile, "w")
    if hpOutputFile is not None:
    	hpOutput = open(hpOutputFile, "w")
    if fpOutputFile is not None:
    	fpOutput = open(fpOutputFile, "w")
    ## Dictionary to save image path
    if frtOutputFile is not None:
    	fileDict = dict()
    	fileDict[pose_models[0]]=frtOutput
    	fileDict[pose_models[1]]=hpOutput
    	fileDict[pose_models[2]]=fpOutput
    # check for dlib saved weights for face landmark detection
    # if it fails, dowload and extract it manually from
    # http://sourceforge.net/projects/dclib/files/d.10/shape_predictor_68_face_landmarks.dat.bz2
    check.check_dlib_landmark_weights()
    ## Preloading all the models
    allModels = myutil.preload(this_path,pose_models,nSub)
    
    for f in fileList:
        if '#' in f: #skipping comments
            continue
        splitted = f.split(',')
        image_key = splitted[0]
        image_path1 = splitted[1]
        image_path2 = splitted[3]
        image_landmarks1 = splitted[2]
        image_landmarks2 = splitted[4]
        #image_landmarks = splitted[2]
        img1 = cv2.imread(image_path1, 1)
        img2 = cv2.imread(image_path2, 1)

        if image_landmarks1 != "None":
            lmark1 = np.loadtxt(image_landmarks1)
            lmark2 = np.loadtxt(image_landmarks2)
            #lmarks = np.zeros((1,68,2))
            lmarks1=[]
            lmarks1.append(lmark1)
            lmarks2=[]
            lmarks2.append(lmark2)
        else:
            print('> Detecting landmarks')
            lmarks1 = feature_detection.get_landmarks(img1, this_path)
            lmarks2 = feature_detection.get_landmarks(img2, this_path)


        #img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC )
        #lmarks[0]=lmarks[0]*0.2

        if len(lmarks1) != 0:
            ## Copy back original image and flipping image in case we need
            ## This flipping is performed using a single frontal 3D model but we could use all the model or all the poses
            ## To refine the estimation of yaw. Yaw can change from model to model...
            img_display = img1
            img1, lmarks1, yaw1 = myutil.flipInCase(img1,lmarks1,allModels)
            img2, lmarks2, yaw2 = myutil.flipInCase(img2,lmarks2,allModels)
            listPose = [0]#myutil.decidePose(yaw1,opts)
            #count = 0
            ## Looping over the poses
            for poseId in listPose:
                posee = pose_models[poseId]
                ## Looping over the subjects
                for subj in range(1,nSub+1):
                    #print "> Using pose model in " + pose
                    print '> Looking at file: ', image_path1
                    print '> Looking at file: ', image_path2
                    pose = posee + '_' + str(subj).zfill(2) + '.mat'
                    # load detections performed by dlib library on 3D model and Reference Image
                    print "> Using pose model in " + pose
                    ## Indexing the right model instead of loading it each time from disk.
                    model3D = allModels[pose]
                    eyemask = model3D.eyemask
                    mouthmask = model3D.mouthmask
                    nosemask = model3D.nosemask
                    # perform camera calibration according to the first face detected
                    proj_matrix1, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks1[0])
                    proj_matrix2, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks2[0])
					## We use eyemask only for frontal
                    if not myutil.isFrontal(pose):
                        eyemask = None
                    allmask=eyemask+mouthmask+nosemask
                    ##### Main part of the code: doing the rendering #############
                    img1a, img2b = renderer.swap(img1.copy(), img2.copy(), proj_matrix1, proj_matrix2,\
				                    model3D.ref_U, allmask, model3D.facemask, opts)
                    #img1c, img2d = renderer.swap(img1.copy(), img2.copy(), proj_matrix1, proj_matrix2,\
                    #               model3D.ref_U, eyemask, model3D.facemask, opts)
                    ########################################################

                    plt.subplot(231)
                    plt.title('Source Img 1')
                    plt.imshow(img1[:, :, ::-1])
                    plt.axis('off')

                    plt.subplot(234)
                    plt.title('Source Img 2')
                    plt.imshow(img2[:, :, ::-1])
                    plt.axis('off')

                    plt.subplot(232)
                    plt.title('Img 1 [all masks (eye+nose+mask) ]')
                    plt.imshow(img1a[:, :, ::-1])
                    plt.axis('off')

                    plt.subplot(235)
                    plt.title('Img 2 [all masks] (eye+nose+mask) ')
                    plt.imshow(img2b[:, :, ::-1])
                    plt.axis('off')

                    # plt.subplot(233)
                    # plt.title('Img 1 [eye only masks] ')
                    # plt.imshow(img1c[:, :, ::-1])
                    # plt.axis('off')

                    # plt.subplot(236)
                    # plt.title('Img 2 [eye only masks]')
                    # plt.imshow(img2d[:, :, ::-1])
                    #plt.axis('off')
                    plt.draw()
                    plt.pause(0.001)
                    enter = raw_input("Press [enter] to continue.")
                    plt.clf()

                #count+=1
        else:
            print '> Landmark not detected for this image...'
	##Closing files            
    if frtOutputFile is not None:
        frtOutput.close()
    if hpOutputFile is not None:
        hpOutput.close()        
    if fpOutputFile is not None:
        fpOutput.close()        

if __name__ == "__main__":
    demo()
