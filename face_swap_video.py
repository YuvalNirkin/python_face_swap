import faceswap
import argparse
import cv2

if __name__ == '__main__':
    # Parse program arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', required=True, help='Source image')
    parser.add_argument('--target', '-t', required=True, help='Target video')
    parser.add_argument('--output', '-o', default='output/out.mp4', help='Output video')
    parser.add_argument('--landmarks', '-l', default='dlib_models/shape_predictor_68_face_landmarks.dat', help='Face landmarks model')
    parser.add_argument('--model3d', '-m', default='models3d/candide.npz', help='Face 3D model')
    args = parser.parse_args()
    
    # Read source image
    source_img = cv2.imread(args.source)    

    # Initialize Video
    cap = cv2.VideoCapture(args.target)
    ret, frame = cap.read()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    vid_writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc('X', '2', '6', '4'), fps, (width, height))

    # Initialize face swap
    fs = faceswap.faceswap(width, height, source_img, args.landmarks, args.model3d)

    # Capture loop
    while True:
        # Read video frame
        ret, frame = cap.read()
        if not ret: break
        
        # Do face swap
        rendered_frame = fs.swap(frame)

        # Write output frame
        vid_writer.write(rendered_frame)

        # Display result
        cv2.imshow('frame', frame)
        if rendered_frame is not None:
            cv2.imshow('render', rendered_frame)
        if cv2.waitKey(1) >= 0: break

    # Clean up
    cap.release()
    vid_writer.release()

    