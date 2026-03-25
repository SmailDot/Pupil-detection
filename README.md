# Pupil Detection (人臉瞳孔偵測)

## 題目說明

隨機匯入一張人臉圖片，偵測並標記瞳孔位置。需處理以下情境：

- 人物低頭、斜視等姿態變化
- 人物遠近距離不同
- 瞳孔反光干擾

## 功能需求

1. **圈出瞳孔範圍** — 在圖片上標記左右眼瞳孔區域
2. **計算兩眼瞳孔中心距離** — 輸出兩瞳孔中心點之間的像素距離

## 限制使用工具

僅能使用以下影像處理工具（順序不限）：

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

## 每次執行輸出

程式執行時會列出本次使用的工具順序，例如：

```
工具使用順序: Gaussian Blur → Binarization → Canny → Contour → Hough
```

## 使用方式

```bash
pip install -r requirements.txt
python pupil_detection.py <image_path>
```

## 輸出結果

- 標記瞳孔的結果圖片（存為 `result.jpg`）
- 終端輸出兩眼瞳孔中心座標與距離
