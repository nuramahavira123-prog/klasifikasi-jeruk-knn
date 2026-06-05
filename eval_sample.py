import cv2, numpy as np, math

data_training = [
    [137,112,74,'baik'],[125,103,65,'baik'],[135,80,28,'baik'],[153,110,24,'baik'],[123,96,23,'baik'],[127,107,40,'baik'],[109,92,72,'baik'],[112,100,44,'baik'],[122,88,38,'baik'],
    [110,115,19,'sedang'],[116,118,13,'sedang'],[108,154,14,'sedang'],[93,102,27,'sedang'],[121,152,16,'sedang'],[113,116,20,'sedang'],[100,120,28,'sedang'],[107,114,7,'sedang'],[101,129,20,'sedang'],
    [73,132,2,'buruk'],[87,128,16,'buruk'],[63,113,21,'buruk'],[95,151,28,'buruk'],[89,128,52,'buruk'],[75,100,13,'buruk'],[84,138,13,'buruk'],[99,138,4,'buruk'],[92,124,0,'buruk']
]

ORANYE_BAWAH=np.array([5,80,80]); ORANYE_ATAS=np.array([25,255,255])
HIJAU_BAWAH=np.array([30,50,50]); HIJAU_ATAS=np.array([90,255,255])
KUNING_BAWAH=np.array([20,80,80]); KUNING_ATAS=np.array([35,255,255])
K=5; AREA_MIN=2000; CIRCULARITY_MIN=0.45

def hitung_jarak(a,b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def klasifikasi_knn(r,g,b):
    jarak = sorted([[hitung_jarak([r,g,b],d[:3]), d[3]] for d in data_training], key=lambda x: x[0])
    tetangga = jarak[:K]
    skor = {'baik': 0.0, 'sedang': 0.0, 'buruk': 0.0}
    for dist, label in tetangga:
        skor[label] += 1.0 / (dist + 1e-5)
    hasil = max(skor, key=skor.get)
    return hasil, skor[hasil] / sum(skor.values()) * 100

def buat_mask(hsv):
    m_oranye = cv2.inRange(hsv, ORANYE_BAWAH, ORANYE_ATAS)
    m_hijau = cv2.inRange(hsv, HIJAU_BAWAH, HIJAU_ATAS)
    m_kuning = cv2.inRange(hsv, KUNING_BAWAH, KUNING_ATAS)
    mask = cv2.bitwise_or(m_oranye, cv2.bitwise_or(m_hijau, m_kuning))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask

def rgb_dari_roi_termasking(img, mask, x, y, w, h):
    roi = img[y:y+h, x:x+w]
    roi_mask = mask[y:y+h, x:x+w]
    roi_mask_bin = (roi_mask > 127).astype(np.uint8)
    r = roi[:, :, 2][roi_mask_bin == 1]
    g = roi[:, :, 1][roi_mask_bin == 1]
    b = roi[:, :, 0][roi_mask_bin == 1]
    if len(r) == 0:
        return int(np.mean(roi[:, :, 2])), int(np.mean(roi[:, :, 1])), int(np.mean(roi[:, :, 0]))
    return int(np.mean(r)), int(np.mean(g)), int(np.mean(b))

def classify_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (800, 600))
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = buat_mask(hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < AREA_MIN:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter ** 2)
        if circularity < CIRCULARITY_MIN:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        r, g, b = rgb_dari_roi_termasking(img, mask, x, y, w, h)
        label, conf = klasifikasi_knn(r, g, b)
        results.append({'label': label, 'conf': conf, 'area': area, 'circ': circularity, 'bbox': (x,y,w,h)})
    return results

if __name__ == '__main__':
    for name in ['jeruk.jpg', 'jeruk2.jpg']:
        res = classify_image(name)
        print(name, 'detected', len(res), 'objects')
        for i, r in enumerate(res, 1):
            print(' ', i, r)
