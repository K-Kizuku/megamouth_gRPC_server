import os
import sys
import argparse
import numpy as np
import cv2
import glob


def generate_feature_dictionary(image_path):

    # 引数から画像ファイルのパスを取得
    path = image_path
    directory = os.getcwd()
    # モデルを読み込む
    weights = os.path.join(directory, "./models/face_recognizer_fast.onnx")
    face_recognizer = cv2.FaceRecognizerSF_create(weights, "")
    if not directory:
        directory = os.path.dirname(__file__)
        # path = os.path.join(directory,image_path)
    files = glob.glob(image_path)
    for file in files:
        # feature = np.load(file)
        # user_id = os.path.splitext(os.path.basename(file))[0]
        # dictionary.append((user_id, feature))
        # 画像を開く
        image = cv2.imread(file)
        if image is None:
            exit()

        # 画像が3チャンネル以外の場合は3チャンネルに変換する
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)



        # 特徴を抽出する
        face_feature = face_recognizer.feature(image)
        print(face_feature)
        print(type(face_feature))

        # 特徴を保存する
        basename = os.path.splitext(os.path.basename(file))[0]
        dictionary = os.path.join(directory, f"data/{basename}")
        np.save(dictionary, face_feature)


if __name__ == '__main__':
    generate_feature_dictionary()
