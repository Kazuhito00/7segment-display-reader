#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse
from collections import deque
from collections import Counter

import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

click_points = deque(maxlen=4)


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

    args = parser.parse_args()

    return args


def mouse_callback(event, x, y, flags, param):
    global click_points
    if event == cv.EVENT_LBUTTONDOWN:
        click_points.append([x, y])


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
    interpreter = None
    if (use_binarize is False) and (use_binarize_inverse is False):
        print('Load : 02.model/7seg_classifier.tflite')
        interpreter = tf.lite.Interpreter(
            model_path='02.model/7seg_classifier.tflite')
    else:
        print('Load : 02.model/7seg_classifier(monochrome).tflite')
        interpreter = tf.lite.Interpreter(
            model_path='02.model/7seg_classifier(monochrome).tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 推論結果格納用変数
    results = []
    for index in range(num_digits):
        results.append(deque(maxlen=check_count))

    elapsed_time = 0.0
    while True:
        start_time = time.time()

        extract_image = None

        # カメラキャプチャ
        ret, frame = cap.read()
        if not ret:
            print('Error : cap.read()')
        resize_frame = cv.resize(frame, (int(cap_width), int(cap_height)))

        # 指定領域抜き出し
        if len(click_points) == 4:
            # 射影変換
            pts1 = np.float32([
                click_points[0],
                click_points[1],
                click_points[2],
                click_points[3],
            ])
            pts2 = np.float32([
                [0, 0],
                [crop_width * num_digits, 0],
                [crop_width * num_digits, crop_height],
                [0, crop_height],
            ])
            M = cv.getPerspectiveTransform(pts1, pts2)
            extract_image = cv.warpPerspective(
                resize_frame, M, (crop_width * num_digits, crop_height))

        # 7セグメント推論
        if extract_image is not None:
            if (use_binarize is True) or (use_binarize_inverse is True):
                # グレースケール化
                extract_image = cv.cvtColor(extract_image, cv.COLOR_BGR2GRAY)

                # ヒストグラム平坦化
                clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                extract_image = clahe.apply(extract_image)

                # ガウシアンブラー
                extract_image = cv.GaussianBlur(extract_image, (3, 3), 0)

                # 2値化
                _, extract_image = cv.threshold(
                    extract_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

                # 反転
                if use_binarize_inverse is True:
                    extract_image = cv.bitwise_not(extract_image)

                # BGR形式へ戻す
                extract_image = cv.cvtColor(extract_image, cv.COLOR_GRAY2BGR)

            for index in range(num_digits):
                # リサイズ、および正規化
                temp_width = int((extract_image.shape[1] / num_digits) * index)
                temp_height = extract_image.shape[0]

                number_image = extract_image[
                    0:temp_height,
                    temp_width:int(extract_image.shape[1] / num_digits) +
                    temp_width]

                input_image = cv.resize(number_image, (96, 96))
                input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
                input_image = img_to_array(input_image)
                input_image = input_image.reshape(-1, 96, 96, 3)
                input_image = input_image.astype('float32')
                input_image = input_image * 1.0 / 255

                # 推論実行
                interpreter.set_tensor(input_details[0]['index'], input_image)
                interpreter.invoke()

                # 結果確認
                result = interpreter.get_tensor(output_details[0]['index'])
                predict_number = np.argmax(np.squeeze(result))

                # 直近で頻出する値を推論値として採用
                results[index].append(predict_number)
                counter = Counter(results[index])
                most_common_number = counter.most_common()[0][0]

                # デバッグ表示
                most_common_number_str = str(most_common_number)
                if most_common_number == 10:
                    most_common_number_str = '-'
                elif most_common_number == 11:
                    most_common_number_str = ' '
                cv.putText(number_image, most_common_number_str, (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                           cv.LINE_AA)

        # デバッグ情報描画
        for click_point in click_points:
            cv.circle(resize_frame, (click_point[0], click_point[1]), 5,
                      (0, 255, 0), -1)
        if len(click_points) >= 3:
            cv.drawContours(resize_frame, [np.array(click_points)], -1,
                            (0, 255, 0), 2)
        if extract_image is not None:
            for index in range(1, num_digits):
                temp_x = int((extract_image.shape[1] / num_digits) * index)
                temp_y = extract_image.shape[0]
                cv.line(extract_image, (temp_x, 0), (temp_x, temp_y),
                        (0, 255, 0), 1)
        cv.putText(
            resize_frame,
            "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)

        # GUI描画更新
        cv.imshow(window_name, resize_frame)
        if extract_image is not None:
            cv.imshow('Result', extract_image)

        # キー入力(ESC:プログラム終了)
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        elapsed_time = time.time() - start_time

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
