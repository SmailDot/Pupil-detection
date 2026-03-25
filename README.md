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
3. **偵測五官位置** — 標記以下五官：
   - 眼睛 (Eyes)
   - 眉毛 (Eyebrows)
   - 鼻子 (Nose)
   - 嘴唇 (Lips/Mouth)
   - 耳朵 (Ears)

## 使用工具

以下影像處理工具（順序不限），以及 OpenCV 內建工具皆可使用：

| 工具 | 說明 |
|------|------|
| Gaussian Blur | 高斯模糊降噪 |
| Binarization | 二值化處理 (含 OTSU 自適應閾值) |
| Sobel | Sobel 邊緣偵測 |
| Canny | Canny 邊緣偵測 |
| Contour | 輪廓偵測 |
| Connected Component Labeling | 連通元件標記 |
| Hough | 霍夫圓偵測 |
| Perspective Transform | 透視變換 |
| Reference Pt | 參考點定位 |
| Haar Cascade | OpenCV 人臉/眼睛分類器 |
| Morphology | 形態學運算 (侵蝕/膨脹/開/閉) |
| Rotation (warpAffine) | 旋轉校正 |

工具可重複多次使用以達成最佳偵測效果。

## 每次執行輸出

程式執行時會列出本次使用的工具順序，例如：

```
Tool order: Perspective Transform -> Reference Pt -> Sobel -> Gaussian Blur -> Binarization -> Connected Component Labeling -> Canny -> Contour -> Hough
```

## 使用方式

```bash
pip install -r requirements.txt
python pupil_detection.py <image_path>
```

支援格式：jpg, png, bmp, webp（安裝 Pillow 可額外支援 avif 等格式）

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
        → 眼睛 ROI 擷取 (Haar Cascade, 去除眉毛區域)
        → 瞳孔偵測:
            Gaussian Blur → OTSU Binarization → Erosion
            → Connected Component Labeling (主要)
            → Canny + Contour (輔助)
            → Hough 圓偵測 (備援)
        → 五官偵測:
            眼睛: Haar Cascade (多分類器)
            眉毛: Sobel + Contour (水平長條形篩選)
            鼻子: Gaussian Blur + Sobel + Canny + Contour (鼻尖區域)
            嘴唇: Canny + Contour (下半臉水平邊緣)
            耳朵: Sobel + Contour (臉部兩側)
        → 繪製結果 + 計算瞳孔距離
```
