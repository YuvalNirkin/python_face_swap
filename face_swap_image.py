import faceswap
import argparse
import cv2

if __name__ == '__main__':
    # Parse program arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', required=True, help='Source image')
    parser.add_argument('--target', '-t', required=True, help='Target image')
    parser.add_argument('--landmarks', '-l', default='dlib_models/shape_predictor_68_face_landmarks.dat', help='Face landmarks model')
    parser.add_argument('--model3d', '-m', default='models3d/candide.npz', help='Face 3D model')
    args = parser.parse_args()
    
    # Read source and target images
    source_img = cv2.imread(args.source)    
    target_img = cv2.imread(args.target)

    # Initialize face swap
    fs = faceswap.faceswap(target_img.shape[1], target_img.shape[0], source_img, args.landmarks, args.model3d)

    # Do face swap
    rendered_img = fs.swap(target_img)

    # Display result
    if rendered_img is not None:
        cv2.imshow('render', rendered_img)
        cv2.waitKey(0)
    else: print 'Could not find faces!'