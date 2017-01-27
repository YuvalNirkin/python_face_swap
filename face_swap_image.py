import faceswap
import argparse
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Parse program arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', required=True, help='Source image')
    parser.add_argument('--target', '-t', required=True, help='Target image')
    parser.add_argument('--output', '-o', default='output/out.jpg', help='Output image')
    parser.add_argument('--landmarks', '-l', default='dlib_models/shape_predictor_68_face_landmarks.dat', help='Face landmarks model')
    parser.add_argument('--model3d', '-m', default='models3d/candide.npz', help='Face 3D model')
    parser.add_argument('--bidirectional', '-b', help='Swap both directions', action="store_true")
    parser.add_argument('--ss', help='Source segmentation')
    parser.add_argument('--ts', help='Target segmentation')
    args = parser.parse_args()
    
    # Read source and target images
    source_img = cv2.imread(args.source)    
    target_img = cv2.imread(args.target)

    # Read source and target segmentations
    source_seg, target_seg = None, None
    if args.ss is not None:
        source_seg = cv2.imread(args.ss, cv2.IMREAD_GRAYSCALE)
    if args.ts is not None:
        target_seg = cv2.imread(args.ts, cv2.IMREAD_GRAYSCALE)

    # Initialize face swap
    fs = faceswap.faceswap(target_img.shape[1], target_img.shape[0], source_img, args.landmarks, args.model3d, source_seg)

    # Do face swap
    rendered_img = fs.swap(target_img, target_seg)

    if args.bidirectional:
        fs2 = faceswap.faceswap(source_img.shape[1], source_img.shape[0], target_img, args.landmarks, args.model3d, target_seg)
        rendered_img2 = fs2.swap(source_img, source_seg)

    # Display result
    if args.bidirectional:    
        plt.subplot(231)
        plt.title('img1')
        plt.imshow(source_img[:, :, ::-1])
        plt.axis('off')

        plt.subplot(234)
        plt.title('img2')
        plt.imshow(target_img[:, :, ::-1])
        plt.axis('off')

        plt.subplot(232)
        plt.title('img1 -> img2')
        plt.imshow(rendered_img[:, :, ::-1])
        plt.axis('off')

        plt.subplot(235)
        plt.title('img2 -> img1')
        plt.imshow(rendered_img2[:, :, ::-1])
        plt.axis('off')

        plt.draw()
        plt.pause(0.001)

        # Write output image
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5, forward=True)
        plt.savefig(args.output, bbox_inches='tight')
        
        #enter = raw_input("Press [enter] to continue.")
        #plt.clf()   
    else:
        if rendered_img is not None:
            cv2.imshow('render', rendered_img)
            cv2.waitKey(1)
        else: print 'Could not find faces!'

        # Write output image
        cv2.imwrite(args.output, rendered_img)