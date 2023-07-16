import os
import sys
import glob
import numpy as np
import cv2
import time
import urllib.error
import urllib.request
import requests

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


class ImageService(recognizer_pb2_grpc.ImageServiceServicer):
    def __init__(self):
        # self.base_url = os.environ.get('BLOB_URL')
        self.image = None
        self.url = None
        self.uuid = None
        
        
    def ImageReq(self, request, context):
        self.image = self.download_image(request.url)        
        self.uuid = os.path.splitext(os.path.basename(request.url))[0]
        print("test")
        self.generate_feature_dictionary()
        print("done generate_feature_dictionary()")
        
        user = self.face_recognizer()
        print("done face_recognizerq()")
        
        f = {"msg":user}
        print(f)
        return recognizer_pb2.Account(msg=user[0])
  
    def download_image(self,url):
        try:
                
            response = requests.get(url,stream=True)
            nparr = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = self.imgEncodeDecode([image],cv2.IMREAD_COLOR)
            # img = Image.open(image)
            # img_r = img.resize((100,100))
            # response.raise_for_status()
       
            return image
        except requests.exceptions.RequestException as e:
            print(f"Error occurred while downloading the image: {e}")
            return "error"
        
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
        print("ok")
        
        if self.image is None:
                print("ng")
            
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
        print(face_feature)
        print(type(face_feature))

        # 特徴を保存する
        # basename = os.path.splitext(os.path.basename(file))[0]
        dictionary = os.path.join(directory, f"data/{self.uuid}")
        np.save(dictionary, face_feature)
        # for file in files:
        #     # feature = np.load(file)
        #     # user_id = os.path.splitext(os.path.basename(file))[0]
        #     # dictionary.append((user_id, feature))
        #     # 画像を開く
        #     image = cv2.imread(file)
        #     if image is None:
        #         exit()

        #     # 画像が3チャンネル以外の場合は3チャンネルに変換する
        #     channels = 1 if len(image.shape) == 2 else image.shape[2]
        #     if channels == 1:
        #         image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        #     if channels == 4:
        #         image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                
        #     # 特徴を抽出する
        #     face_feature = face_recognizerq.feature(image)
        #     print(face_feature)
        #     print(type(face_feature))

        #     # 特徴を保存する
        #     basename = os.path.splitext(os.path.basename(file))[0]
        #     dictionary = os.path.join(directory, f"data/{basename}")
        #     np.save(dictionary, face_feature)

    def match(self,recognizer, feature1, dictionary):
        print(2)
        for element in dictionary:
            user_id, feature2 = element
            score = recognizer.match(
                feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
            if score > COSINE_THRESHOLD:
                return True, (user_id, score)
        return False, ("", 0.0)


    def face_recognizer(self):
        # キャプチャを開く
        
        # directory = os.path.dirname(__file__)
        # capture = cv2.VideoCapture(os.path.join(directory, image_path))  # 画像ファイル
        # if not capture.isOpened():
        #     exit()

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

        # while True:
        # フレームをキャプチャして画像を読み込む
        # result, image = capture.read()
        image = self.image
        # if result is False:
        #     cv2.waitKey(0)
        #     break

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
        max = ("",0.0)
        for face in faces:
            # 顔を切り抜き特徴を抽出する
            aligned_face = face_recognizerq.alignCrop(image, face)
            feature = face_recognizerq.feature(aligned_face)
            print(1)

            # 辞書とマッチングする
         
            
            
            result, user = self.match(face_recognizerq, feature, dictionary)
            if result :
                if max[1] < user[1]:
                    max = user
                

                # # 顔のバウンディングボックスを描画する
                # box = list(map(int, face[:4]))
                # color = (0, 255, 0) if result else (0, 0, 255)
                # thickness = 2
                # cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

                # # 認識の結果を描画する
                # id, score = user if result else ("unknown", 0.0)
                # text = "{0} ({1:.2f})".format(id, score)
                # position = (box[0], box[1] - 10)
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # scale = 0.6
                # cv2.putText(image, text, position, font, scale,
                #             color, thickness, cv2.LINE_AA)
        return max
            




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
            ImageService() , server)
        d = server.add_insecure_port('[::]:50052')
        print(d)
        server.start()
        server.wait_for_termination()

if __name__ == "__main__":
  server = Server()
  server.Serve()

#   z = 0.
#   azimuth = 0.
#   aziShift = 5* np.pi / 180.

#   def azi2pos(azimuth):
#     x = np.cos(azimuth)
#     y = np.sin(azimuth)
#     return x, y

