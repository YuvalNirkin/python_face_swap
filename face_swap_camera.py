import faceswap
import argparse
import cv2

if __name__ == '__main__':
    # Parse program arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', required=True, help='Source image')
    parser.add_argument('--device', '-d', default=0, type=int, help='Camera device id')
    parser.add_argument('--landmarks', '-l', default='dlib_models/shape_predictor_68_face_landmarks.dat', help='Face landmarks model')
    parser.add_argument('--model3d', '-m', default='models3d/candide.npz', help='Face 3D model')
    args = parser.parse_args()
    
    # Read source image
    source_img = cv2.imread(args.source)    

    # Initialize camera
    cap = cv2.VideoCapture(args.device)
    frame = cap.read()[1]

    # Initialize face swap
    fs = faceswap.faceswap(frame.shape[1], frame.shape[0], source_img, args.landmarks, args.model3d)

    # Capture loop
    while True:
        # Read camera frame
        ret, frame = cap.read()
        
        # Do face swap
        rendered_frame = fs.swap(frame)

        # Display result
        cv2.imshow('frame', frame)
        if rendered_frame is not None:
            cv2.imshow('render', rendered_frame)
        if cv2.waitKey(1) >= 0: break

    