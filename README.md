# Pupil Detection (人臉瞳孔偵測與五官定位)

## 題目說明

隨機匯入一張人臉圖片，偵測並標記瞳孔位置與五官位置。需處理以下情境：

- 人物低頭、斜視等姿態變化
- 人臉方向不一定正視前方（側臉、歪頭、旋轉角度）
- 人物遠近距離不同
- 瞳孔反光干擾

## 功能需求

1. **圈出瞳孔範圍** — 在圖片上標記左右眼瞳孔區域
2. **計算兩眼瞳孔中心距離** — 輸出兩瞳孔中心點之間的像素距離
3. **偵測五官位置** — 標記眼睛、眉毛、鼻子、嘴巴位置

## 使用工具

以下影像處理工具（順序不限），以及 OpenCV 內建工具皆可使用：

| 工具 | 說明 |
|------|------|
| Gaussian Blur | 高斯模糊降噪 |
| Binarization | 二值化處理 |
| Sobel | Sobel 邊緣偵測 |
| Canny | Canny 邊緣偵測 |
| Contour | 輪廓偵測 |
| Hough | 霍夫圓/線偵測 |
| Perspective Transform | 透視變換 |
| Reference Pt | 參考點定位 |
| Haar Cascade | OpenCV 人臉/眼睛/鼻子/嘴巴分類器 |
| Morphology | 形態學運算 (開/閉運算) |
| Rotation (warpAffine) | 旋轉校正 |

## 每次執行輸出

程式執行時會列出本次使用的工具順序，例如：

```
工具使用順序: Perspective Transform → Reference Pt → Sobel → Gaussian Blur → Binarization → Canny → Contour → Hough
```

## 使用方式

```bash
pip install -r requirements.txt
python pupil_detection.py <image_path>
```

## 輸出結果

- 標記瞳孔與五官的結果圖片（存為 `result.jpg`）
- 終端輸出兩眼瞳孔中心座標與距離
- 終端輸出五官位置座標

## 處理流程

```
輸入圖片 → 多角度人臉偵測 (正面/側臉/旋轉±45°)
        → Perspective Transform 校正傾斜
        → Reference Pt 建立眼角參考點
        → Sobel 分析眼區邊緣
        → 左右眼 ROI 擷取
        → Gaussian Blur 降噪
        → Binarization 二值化
        → Canny 邊緣偵測
        → Contour 輪廓篩選
        → Hough 圓偵測 (備援)
        → 五官偵測 (Haar Cascade + Sobel + Contour)
        → 繪製結果 + 計算瞳孔距離
```
