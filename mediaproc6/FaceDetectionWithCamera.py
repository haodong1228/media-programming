'''
メディアプログラミング 顔画像検出 発展的演習

カメラが内蔵されたPCをお使いの方は、このプログラムでカメラの映像に対する顔画像認識を試してみましょう。

'''

import matplotlib.pyplot as plt
import cv2

# 「()」の中の数字はデバイスの番号
# カメラが１台しかない場合は通常「0」ですが、カメラが複数ついている場合は「1」以上の整数になります
cap = cv2.VideoCapture(0) 

# Haar-like特徴量のモデルを読み込み
cascade_path = "opencv_data/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_path) # 

while True:
    ret, img = cap.read() # カメラデバイスから画像を読み込み

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # opencv形式のグレースケール画像に変換
    faces = cascade.detectMultiScale(img_gray, 
    scaleFactor=1.01, 
    minNeighbors=2#, 
    #    minSize=(40, 40) # 画像における顔の領域の最小値が決まっている場合はここを有効
    )
    print(faces)

    # 検出された顔領域に矩形を描画
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('camera capture', img)
    # カメラは`q`のキーを押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラデバイスを解放
cap.release()
# ウィンドウを閉じる
cv2.destroyAllWindows()

