import utils
import models
import face_renderer
import NonLinearLeastSquares
import numpy as np
import cv2
import dlib

class faceswap(object):
    """Swap faces"""
    
    def __init__(self, width, height, source_img, landmarks_model_path, model3d_path):
        self.w = width
        self.h = height
        self.detector = dlib.get_frontal_face_detector()
        self.landmarks_model = dlib.shape_predictor(landmarks_model_path)
        self.mean3DShape, self.blendshapes, self.mesh, self.idxs3D, self.idxs2D = \
            utils.load3DFaceModel(model3d_path)
        self.projectionModel = models.OrthographicProjectionBlendshapes(self.blendshapes.shape[0])

        landmarks = self.calcLandmarks(source_img)[0]
        textureCoords = self.calcTextureCoords(landmarks)
        self.renderer = face_renderer.FaceRenderer(self.w, self.h, source_img, textureCoords, self.mesh)

    def swap(self, target_img):
        landmarks_set = self.calcLandmarks(target_img)
        if landmarks_set is None: return None
        for landmarks in landmarks_set:
            # 3D model parameter optimization
            modelParams = self.projectionModel.getInitialParameters(self.mean3DShape[:, self.idxs3D], landmarks[:, self.idxs2D])
            modelParams = NonLinearLeastSquares.GaussNewton(modelParams, self.projectionModel.residual, self.projectionModel.jacobian, ([self.mean3DShape[:, self.idxs3D], self.blendshapes[:, :, self.idxs3D]], landmarks[:, self.idxs2D]), verbose=0)
            #modelParams = self.projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])
            #modelParams = NonLinearLeastSquares.GaussNewton(modelParams, self.projectionModel.residual, self.projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]), verbose=0)

            # Render the model to the target image
            shape3D = utils.getShape3D(self.mean3DShape, self.blendshapes, modelParams)
            renderedImg = self.renderer.render(shape3D)

            return renderedImg

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