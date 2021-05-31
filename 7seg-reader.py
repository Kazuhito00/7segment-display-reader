#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
from collections import deque
from collections import Counter

import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

from utils import CvFpsCalc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--crop_width", type=int, default=96)
    parser.add_argument("--crop_height", type=int, default=96)

    parser.add_argument("--num_digits", type=int, default=4)
    parser.add_argument("--check_count", type=int, default=5)

    parser.add_argument('--use_binarize', action='store_true')
    parser.add_argument('--use_binarize_inverse', action='store_true')
    parser.add_argument('--binarize_th', type=int, default=None)

    args = parser.parse_args()

    return args


click_points = deque(maxlen=4)
def mouse_callback(event, x, y, flags, param):
    global click_points
    if event == cv.EVENT_LBUTTONDOWN:
        click_points.append([x, y])


def load_7seg_classifier(use_binarize):
    interpreter = None

    # if use_binarize is False:
    #     print('Load : 02.model/7seg_classifier.tflite')
    #     interpreter = tf.lite.Interpreter(
    #         model_path='02.model/7seg_classifier.tflite')
    # else:
    #     print('Load : 02.model/7seg_classifier(monochrome).tflite')
    #     interpreter = tf.lite.Interpreter(
    #         model_path='02.model/7seg_classifier.tflite')

    print('Load : 02.model/7seg_classifier.tflite')
    interpreter = tf.lite.Interpreter(
        model_path='02.model/7seg_classifier.tflite')

    interpreter.allocate_tensors()

    return interpreter


def extract_click_point_image(
    image, 
    click_points_, 
    num_digits, 
    crop_width, 
    crop_height,
):
    extract_image = None
    
    if len(click_points_) == 4:
        # 射影変換
        pts1 = np.float32([
            click_points_[0],
            click_points_[1],
            click_points_[2],
            click_points_[3],
        ])
        pts2 = np.float32([
            [0, 0],
            [crop_width * num_digits, 0],
            [crop_width * num_digits, crop_height],
            [0, crop_height],
        ])
        M = cv.getPerspectiveTransform(pts1, pts2)
        extract_image = cv.warpPerspective(
            image, M, (crop_width * num_digits, crop_height))
            
    return extract_image


def preprocess_binarization(image, inverse_flag, binarize_th):
    temp_image = copy.deepcopy(image)

    # グレースケール化
    temp_image = cv.cvtColor(temp_image, cv.COLOR_BGR2GRAY)

    # ヒストグラム平坦化
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    temp_image = clahe.apply(temp_image)

    # ガウシアンブラー
    temp_image = cv.GaussianBlur(temp_image, (3, 3), 0)

    # 2値化
    if binarize_th is None:
        _, temp_image = cv.threshold(
            temp_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    else:
        _, temp_image = cv.threshold(temp_image, binarize_th, 255, cv.THRESH_BINARY)

    # 反転
    if inverse_flag is True:
        temp_image = cv.bitwise_not(temp_image)

    # BGR形式へ戻す
    temp_image = cv.cvtColor(temp_image, cv.COLOR_GRAY2BGR)

    return temp_image


def inference_7seg_classifier(
    interpreter, 
    number_image, 
    input_details, 
    output_details,
):
    # リサイズ、および正規化
    input_image = cv.resize(number_image, (96, 96))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = img_to_array(input_image)
    input_image = input_image.reshape(-1, 96, 96, 3)
    input_image = input_image.astype('float32')
    input_image = input_image * 1.0 / 255

    # 推論実行
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    # 結果取得
    result = interpreter.get_tensor(output_details[0]['index'])
    predict_number = np.argmax(np.squeeze(result))

    return predict_number


def concat_results(results, num_digits):
    result_string = None
    result_value = 0
    if (len(results[0]) > 0):
        result_string = ''
        for index in range(num_digits):
            counter = Counter(results[index])
            most_common_number = counter.most_common()[0][0]
            if 0 <= most_common_number <= 9:
                result_string = result_string + str(most_common_number)
    if (result_string is not None) and (result_string != ''):
        result_value = int(result_string)

    return result_value

def draw_debug_info(
    resize_frame, 
    extract_image, 
    click_points_, 
    num_digits, 
    results, 
    display_fps,
):
    for click_point in click_points_:
        cv.circle(resize_frame, (click_point[0], click_point[1]), 5,
                    (0, 255, 0), -1)
    if len(click_points) >= 3:
        cv.drawContours(resize_frame, [np.array(click_points)], -1,
                        (0, 255, 0), 2)
    if extract_image is not None:
        for index in range(num_digits):
            temp_x = int((extract_image.shape[1] / num_digits) * index)
            temp_y = extract_image.shape[0]
        
            if index > 0 :
                cv.line(extract_image, (temp_x, 0), (temp_x, temp_y),
                        (0, 255, 0), 1)
                    
            counter = Counter(results[index])
            most_common_number = counter.most_common()[0][0]
            most_common_number_str = str(most_common_number)
            if most_common_number == 10:
                most_common_number_str = '-'
            elif most_common_number == 11:
                most_common_number_str = ' '
            cv.putText(extract_image, most_common_number_str, (temp_x + 10, 30),
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                        cv.LINE_AA)

    cv.putText(resize_frame, "FPS:" + str(display_fps), (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
    
    return resize_frame, extract_image


def main():
    global click_points

    # コマンドライン引数
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    crop_width = args.crop_width
    crop_height = args.crop_height

    num_digits = args.num_digits
    check_count = args.check_count

    use_binarize = args.use_binarize
    use_binarize_inverse = args.use_binarize_inverse
    binarize_th = args.binarize_th
    if (use_binarize_inverse is True) or (binarize_th is not None):
        use_binarize = True

    # GUI準備
    window_name = '7seg Reader'
    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, mouse_callback)

    # カメラ準備
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # 認識対象座標 格納用
    click_points = deque(maxlen=4)

    # tfliteモデルロード
    interpreter = load_7seg_classifier(use_binarize)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 推論結果格納用変数
    results = []
    for index in range(num_digits):
        results.append(deque(maxlen=check_count))

    # FPS計測モジュール
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ
        ret, frame = cap.read()
        if not ret:
            print('Error : cap.read()')
        resize_frame = cv.resize(frame, (int(cap_width), int(cap_height)))

        # 指定領域抜き出し
        extract_image = extract_click_point_image(
            resize_frame, 
            click_points, 
            num_digits, 
            crop_width, 
            crop_height,
        )

        # 7セグメント推論
        if extract_image is not None:
            if use_binarize is True:
                # 前処理：2値化
                extract_image = preprocess_binarization(
                    extract_image, 
                    use_binarize_inverse, 
                    binarize_th,
                )

            for index in range(num_digits):
                # 識別対象画像を切り抜き
                temp_start_point = int((extract_image.shape[1] / num_digits) * index)
                temp_width = int(extract_image.shape[1] / num_digits)
                temp_height = extract_image.shape[0]
                number_image = extract_image[
                    0:temp_height,
                    temp_start_point:temp_start_point + temp_width
                ]

                # 7セグメント識別
                predict_number = inference_7seg_classifier(
                    interpreter, 
                    number_image, 
                    input_details, 
                    output_details,
                )
                results[index].append(predict_number)

        # 各桁の識別結果を結合
        # ※直近で頻出する値を結果として採用
        result_value = concat_results(results, num_digits)
        print('Result : ' + str(result_value))

        # デバッグ情報描画
        resize_frame, extract_image = draw_debug_info(
            resize_frame, 
            extract_image, 
            click_points, 
            num_digits, 
            results, 
            display_fps,
        )

        # 描画更新
        cv.imshow(window_name, resize_frame)
        if extract_image is not None:
            cv.imshow('Result', extract_image)

        # キー入力(ESC:プログラム終了)
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
