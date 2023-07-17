import os
import sys
import glob
import numpy as np
import cv2
import time
import urllib.error
import urllib.request
import requests
import base64

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ------------


# ------------
#grpc

import grpc
from proto.recognizer_pb2 import *
from proto import recognizer_pb2_grpc,recognizer_pb2

# ------------

from concurrent import futures


COSINE_THRESHOLD = 0.363
NORML2_THRESHOLD = 1.128

class ImageRegistor(recognizer_pb2_grpc.ImageRegistorServicer):
    def __init__(self):
        # self.base_url = os.environ.get('BLOB_URL')
        self.image = None
        self.url = None
        self.id = None
        self.uuid = None
        
    def ImageReqURL(self, request, context):
        self.image = self.download_image(request.url)
        self.id = request.id
        self.uuid = os.path.splitext(os.path.basename(request.url))[0]
        self.generate_feature_dictionary()
        return recognizer_pb2.Notice(res="OK")   
    
    def download_image(self,url):
        try:
            response = requests.get(url,stream=True)
            nparr = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imwrite("test1.jpg",image)
            
            # img = self.imgEncodeDecode([image],cv2.IMREAD_COLOR)
            # img = Image.open(image)
            # img_r = img.resize((100,100))
            # response.raise_for_status()
        
            return image
        except requests.exceptions.RequestException as e:
            print(f"Error occurred while downloading the image: {e}")
            return "error" 
        
    def generate_aligned_faces(image_path):

        # 引数から画像ファイルのパスを取得
        path = image_path
        directory = os.path.dirname(image_path)
        if not directory:
            directory = os.path.dirname(__file__)
            path = os.path.join(directory, image_path)

        # 画像を開く
        image = cv2.imread(path)
        if image is None:
            exit()

        # 画像が3チャンネル以外の場合は3チャンネルに変換する
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # モデルを読み込む
        weights = os.path.join(directory, "./models/yunet_n_640_640.onnx") #yunet_n_640_640.onnx
        face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
        weights = os.path.join(directory, "./models/face_recognizer_fast.onnx")
        face_recognizerq = cv2.FaceRecognizerSF_create(weights, "")

        # 入力サイズを指定する
        height, width, _ = image.shape
        face_detector.setInputSize((width, height))

        # 顔を検出する
        _, faces = face_detector.detect(image)

        # 検出された顔を切り抜く
        aligned_faces = []
        if faces is not None:
            for face in faces:
                aligned_face = face_recognizerq.alignCrop(image, face)
                aligned_faces.append(aligned_face)

        # 画像を表示、保存する
        for i, aligned_face in enumerate(aligned_faces):
            # cv2.imshow("aligned_face {:03}".format(i + 1), aligned_face)
            cv2.imwrite(os.path.join(
                directory, "./face/face{:03}.jpg".format(i + 1)), aligned_face)
        
    def generate_feature_dictionary(self):

        # 引数から画像ファイルのパスを取得
        # path = image_path
        directory = os.getcwd()
        # モデルを読み込む
        weights = os.path.join(directory, "./models/face_recognizer_fast.onnx")
        face_recognizerq = cv2.FaceRecognizerSF_create(weights, "")
        if not directory:
            directory = os.path.dirname(__file__)
            # path = os.path.join(directory,image_path)
        # files = glob.glob(image_path)
        
        if self.image is None:
            
                exit()
        image = self.image
        # 画像が3チャンネル以外の場合は3チャンネルに変換する
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
        # 特徴を抽出する
        face_feature = face_recognizerq.feature(image)
        # print(face_feature)
        # print(type(face_feature))

        # 特徴を保存する
        # basename = os.path.splitext(os.path.basename(file))[0]
        dictionary = os.path.join(directory, f"data/{self.id}-{self.uuid}")
        np.save(dictionary, face_feature)
    
class ImageService(recognizer_pb2_grpc.ImageServiceServicer):
    def __init__(self):
        # self.base_url = os.environ.get('BLOB_URL')
        self.image = None
        self.url = None
        self.uuid = None
        
        
    def ImageReqBase64(self, request, context):
        self.image = self.base64_decode(request.base)
        user = self.face_recognizer()
        print(user)
        return recognizer_pb2.Account(msg=user[0].split('-')[0])
        
    def base64_decode(self,base):
        img_binary = base64.b64decode(base)
        nparr=np.frombuffer(img_binary,dtype=np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite("test.jpg",image)
        return image
        
        
    def imgEncodeDecode(self,in_imgs, ch, quality=5):
        """
        入力された画像リストを圧縮する
        [in]  in_imgs:  入力画像リスト
        [in]  ch:       出力画像リストのチャンネル数 （OpenCV形式）
        [in]  quality:  圧縮する品質 (1-100)
        [out] out_imgs: 出力画像リスト
        """

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        out_imgs = []

        for img in in_imgs:
            result, encimg = cv2.imencode('.jpg', img, encode_param)
            if False == result:
                print('could not encode image!')
                exit()

            decimg = cv2.imdecode(encimg, ch)
            out_imgs.append(decimg)

        return out_imgs
    
    def generate_aligned_faces(image_path):
        # 引数をパースする
        # parser = argparse.ArgumentParser(
        #     "generate aligned face images from an image")
        # parser.add_argument("image", help="input image file path (./image.jpg)")
        # args = parser.parse_args()

        # 引数から画像ファイルのパスを取得
        path = image_path
        directory = os.path.dirname(image_path)
        if not directory:
            directory = os.path.dirname(__file__)
            path = os.path.join(directory, image_path)

        # 画像を開く
        image = cv2.imread(path)
        if image is None:
            exit()

        # 画像が3チャンネル以外の場合は3チャンネルに変換する
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # モデルを読み込む
        weights = os.path.join(directory, "./models/yunet_n_640_640.onnx") #yunet_n_640_640.onnx
        face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
        weights = os.path.join(directory, "./models/face_recognizer_fast.onnx")
        face_recognizerq = cv2.FaceRecognizerSF_create(weights, "")

        # 入力サイズを指定する
        height, width, _ = image.shape
        face_detector.setInputSize((width, height))

        # 顔を検出する
        _, faces = face_detector.detect(image)

        # 検出された顔を切り抜く
        aligned_faces = []
        if faces is not None:
            for face in faces:
                aligned_face = face_recognizerq.alignCrop(image, face)
                aligned_faces.append(aligned_face)

        # 画像を表示、保存する
        for i, aligned_face in enumerate(aligned_faces):
            # cv2.imshow("aligned_face {:03}".format(i + 1), aligned_face)
            cv2.imwrite(os.path.join(
                directory, "./face/face{:03}.jpg".format(i + 1)), aligned_face)
        

    def match(self,recognizer, feature1, dictionary):
        max = 0.0
        result = ("", 0.0)
        for element in dictionary:
            user_id, feature2 = element
            score = recognizer.match(
                feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
            if result[1] < score:
                result = (user_id, score)
            # if score > COSINE_THRESHOLD:
            #     return True, (user_id, score)
        return True, result


    def face_recognizer(self):

        # 特徴を読み込む
        dictionary = []
        directory = os.getcwd()
        
        files = glob.glob(os.path.join(directory, "./data/*.npy"))
        for file in files:
            feature = np.load(file)
            user_id = os.path.splitext(os.path.basename(file))[0]
            dictionary.append((user_id, feature))
        # モデルを読み込む
        weights = os.path.join(directory, "./models/yunet_n_640_640.onnx")
        face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
        weights = os.path.join(directory, "./models/face_recognizer_fast.onnx")
        face_recognizerq = cv2.FaceRecognizerSF_create(weights, "")

        image = self.image


        # 画像が3チャンネル以外の場合は3チャンネルに変換する
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # 入力サイズを指定する
        height, width, _ = image.shape
        face_detector.setInputSize((width, height))

        # 顔を検出する
        result, faces = face_detector.detect(image)
        faces = faces if faces is not None else []
        # max = ("",0.0)
        # aligned_face = face_recognizerq.alignCrop(image, face)
        feature = face_recognizerq.feature(image)
        print(1)

        # 辞書とマッチングする
        
        
        
        result, user = self.match(face_recognizerq, feature, dictionary)
        # if result :
        #     if max[1] < user[1]:
        #         max = user
        # for face in faces:
        #     # 顔を切り抜き特徴を抽出する
        #     aligned_face = face_recognizerq.alignCrop(image, face)
        #     feature = face_recognizerq.feature(aligned_face)
        #     print(1)

        #     # 辞書とマッチングする
         
            
            
        #     result, user = self.match(face_recognizerq, feature, dictionary)
        #     if result :
        #         if max[1] < user[1]:
        #             max = user
                

        return user
            




class Server():
    def __init__(self):
        self.posServer = ImageService()

    def start(self):

        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=3))
        recognizer_pb2_grpc.add_ImageServiceServicer_to_server(
        self.posServer, self.server
        )

    # self.server.add_insecure_port(Conf.PosServer)
        self.server.start()
    # logger.info("Start server {0}".format(Conf.PosServer))

    def stop(self):
        self.server.stop(0)
    
    def Serve(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        recognizer_pb2_grpc.add_ImageServiceServicer_to_server(
            ImageService() , server
            )
        recognizer_pb2_grpc.add_ImageRegistorServicer_to_server(
            ImageRegistor(),server
            )
        d = server.add_insecure_port('[::]:50052')
        server.start()
        server.wait_for_termination()

if __name__ == "__main__":
  server = Server()
  server.Serve()


