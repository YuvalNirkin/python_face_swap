import utils
import models
import face_renderer
import NonLinearLeastSquares
import numpy as np
import cv2
import dlib

class faceswap(object):
    """Swap faces"""
    
    def __init__(self, width, height, source_img, landmarks_model_path, model3d_path, source_seg = None):
        self.w = width
        self.h = height
        self.detector = dlib.get_frontal_face_detector()
        self.landmarks_model = dlib.shape_predictor(landmarks_model_path)
        self.mean3DShape, self.blendshapes, self.mesh, self.idxs3D, self.idxs2D = \
            utils.load3DFaceModel(model3d_path)
        self.projectionModel = models.OrthographicProjectionBlendshapes(self.blendshapes.shape[0])

        landmarks_arr = self.calcLandmarks(source_img)
        if landmarks_arr:
            landmarks = landmarks_arr[0]
        else: raise Exception("Couldn't find a face in the source image!") 
        textureCoords = self.calcTextureCoords(landmarks)
        self.renderer = face_renderer.FaceRenderer(self.w, self.h, source_img, textureCoords, self.mesh, source_seg)

    def swap(self, target_img, target_seg = None):
        landmarks_set = self.calcLandmarks(target_img)
        if landmarks_set is None: return None
        for landmarks in landmarks_set:
            # 3D model parameter optimization
            modelParams = self.projectionModel.getInitialParameters(self.mean3DShape[:, self.idxs3D], landmarks[:, self.idxs2D])
            modelParams = NonLinearLeastSquares.GaussNewton(modelParams, self.projectionModel.residual, self.projectionModel.jacobian, ([self.mean3DShape[:, self.idxs3D], self.blendshapes[:, :, self.idxs3D]], landmarks[:, self.idxs2D]), verbose=0)

            # Render the model to the target image
            shape3D = utils.getShape3D(self.mean3DShape, self.blendshapes, modelParams)
            rendered_img = self.renderer.render(shape3D)

            # Blend images
            mask = np.zeros(rendered_img.shape, rendered_img.dtype)
            red, green, blue = rendered_img[:,:,0], rendered_img[:,:,1], rendered_img[:,:,2]
            binmask = (red != 0) | (green != 0) | (blue != 0)
            if target_seg is not None: binmask = binmask & (target_seg != 0)
            mask[:,:,:3][binmask] = [255, 255, 255]

            cmin, rmin, cmax, rmax = self.calcBBox(binmask)
            center = ((cmin + cmax)/2, (rmin + rmax)/2)
            output = cv2.seamlessClone(rendered_img, target_img, mask, center, cv2.NORMAL_CLONE)
            return output

    def calcLandmarks(self, img):
        faces = self.detector(img, 1)
        if len(faces) == 0:
            return None

        landmarks = []
        for face in faces:
            shape = self.landmarks_model(img, face)
            points = np.array([[p.x, p.y] for p in shape.parts()])
            points = points.T
            landmarks.append(points)

        return landmarks

    def calcTextureCoords(self, landmarks):
        #projectionModel = models.OrthographicProjectionBlendshapes(self.blendshapes.shape[0])
        modelParams = self.projectionModel.getInitialParameters(self.mean3DShape[:, self.idxs3D], landmarks[:, self.idxs2D])
        modelParams = NonLinearLeastSquares.GaussNewton(modelParams, self.projectionModel.residual, self.projectionModel.jacobian, ([self.mean3DShape[:, self.idxs3D], self.blendshapes[:, :, self.idxs3D]], landmarks[:, self.idxs2D]), verbose=0)
        textureCoords = self.projectionModel.fun([self.mean3DShape, self.blendshapes], modelParams)
        return textureCoords
    
    def calcBBox(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return cmin, rmin, cmax, rmax 