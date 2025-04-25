![image](https://github.com/user-attachments/assets/e331c8e5-a361-4a3a-97a2-76e4ef01ff33)### 前言
在肌力訓練，有氧訓練或是專項訓練都是依靠專業教練的專業程度和素養息息相關。
使用YOLO v5影像辨識、MediaPipe、Dlib追蹤技術程式搭配同時和TIST(Try it! Sport Train)專業教練團合作。研究動機為透過程式讓訓練模式視覺化並提高訓練品質，幫助民眾理解關節角度變化在重量訓時的重要性，以及在未來訓練中可以達成自我訓練和檢視。
本次研究將以程式系統作為輔助教練進行上課會以三項訓練為主，舉重訓練為輔助訓練，會利用MediaPipe的Pose LandMark進行人體關節辨識，用YOLO v5辨識槓片並以Dlib追蹤技術去追縱槓片的移動路線會在健身房運動環境進行教學，以達到實際訓練時的情況

## 使用函式庫
TensorFlow、PyTorch、YOLOv5、Mediapipe、NumPy、Matplotlib
## 主要功能
人體關節偵測、動作分類、視覺化回饋
## 計算的角度方法
使用的方法是用Pose Landmark提供關節點x和y的座標，先利用NumPy陣列轉換座標數值，我們可以使用Numpy內建函示庫np.arctan2來幫忙算出radians(弧度)。
假設我們要計算髖關節角度，我們需要使用肩關節的 (x, y) 座標、髖關節的 (x, y) 座標，以及膝關節的 (x, y) 座標。我們可以使用公式（2）來計算肩關節到髖關節的向量和肩關節到膝關節的向量之間的夾角。我們使用反正切函數接受這兩個向量的y和x座標差值，並返回夾角的弧度值。

![image](https://github.com/user-attachments/assets/e1a6afbd-f12a-41f8-842f-4d2b5513e697)

![image](https://github.com/user-attachments/assets/80c55e9c-3985-4ad1-a4ed-939fd2f0871c)

![image](https://github.com/user-attachments/assets/f7b71842-9f19-4f44-b2b3-5dc59cdaf9f0)

將弧度值轉換為角度值的公式(3)。將弧度值乘以 180.0 再除以 π (𝝅)，即可獲得以角度表示的值。透過上面的計算關節角度的方法計算出各個關節角度並且顯示在影像上，並且為了確認Pose Landmark的關節角度計算正確，必須要針對影片的每一幀進行辨識人體姿態辨識以確保角度持續計算。
使用上述方法可以讓教練可以去透過分析後的影像去和學員做動作架構講說和關節變化的重點。

![image](https://github.com/user-attachments/assets/da84c973-c92d-4fae-a4d5-54a59fe35058)

以深蹲動作為例，假如要知道髖關節的角度要利用膝蓋關節座標點和肩膀的座標點來進行髖關節角度計算。

##DataSet

![plates](https://github.com/user-attachments/assets/1085f04c-64de-4b63-9844-7ffc7782ad0a)

以上有700以上的槓片圖

## 最後畫面

![螢幕擷取畫面 2023-05-11 231543](https://github.com/user-attachments/assets/0ab6c173-6885-4c6a-aff4-b631374fad4e)

![螢幕擷取畫面 2023-05-16 223915](https://github.com/user-attachments/assets/50fb2c83-c017-4212-8ee6-09b244a0c6ae)

# 最後感謝 台北教練團 TIST 總教練 SUN 大力協助


