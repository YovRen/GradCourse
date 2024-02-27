import numpy as np
import cv2
import os


def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pts2wh(pts):
    (tl, tr, br, bl) = pts
    blr = ((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)**0.5
    tlr = ((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)**0.5
    tbr = ((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)**0.5
    tbl = ((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)**0.5
    w = max(int(blr), int(tlr))
    h = max(int(tbr), int(tbl))
    return w, h


def cv2blue(image):
    blue = image[:, :, 0] ** 1.2 - 0.3 * \
        image[:, :, 1] ** 1.4 - 0.3 * image[:, :, 2] ** 1.4
    blue = np.maximum(blue, 0)
    blue = (255 * (blue - 0) / np.max(blue))
    blue = blue.astype("uint8")
    return blue


def fea_vectors(images):
    samples = []
    for img in images:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n)
                 for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= np.linalg.norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


class Model:
    def __init__(self, C=1, gamma=0.5):
        self.provinces = [
            "zh_cuan", "川",
            "zh_e", "鄂",
            "zh_gan", "赣",
            "zh_gan1", "甘",
            "zh_gui", "贵",
            "zh_gui1", "桂",
            "zh_hei", "黑",
            "zh_hu", "沪",
            "zh_ji", "冀",
            "zh_jin", "津",
            "zh_jing", "京",
            "zh_jl", "吉",
            "zh_liao", "辽",
            "zh_lu", "鲁",
            "zh_meng", "蒙",
            "zh_min", "闽",
            "zh_ning", "宁",
            "zh_qing", "靑",
            "zh_qiong", "琼",
            "zh_shan", "陕",
            "zh_su", "苏",
            "zh_sx", "晋",
            "zh_wan", "皖",
            "zh_xiang", "湘",
            "zh_xin", "新",
            "zh_yu", "豫",
            "zh_yu1", "渝",
            "zh_yue", "粤",
            "zh_yun", "云",
            "zh_zang", "藏",
            "zh_zhe", "浙"
        ]
        self.modelchar = cv2.ml.SVM_create()
        self.modelchar.setGamma(gamma)
        self.modelchar.setC(C)
        self.modelchar.setKernel(cv2.ml.SVM_RBF)
        self.modelchar.setType(cv2.ml.SVM_C_SVC)
        self.modelchinese = cv2.ml.SVM_create()

        self.modelchinese.setGamma(gamma)
        self.modelchinese.setC(C)
        self.modelchinese.setKernel(cv2.ml.SVM_RBF)
        self.modelchinese.setType(cv2.ml.SVM_C_SVC)
        if not os.path.exists("svmchar.dat") or not os.path.exists("svmchinese.dat"):
            self.train()
            self.save()
        self.load()

    def load(self):
        self.modelchar = self.modelchar.load("svmchar.dat")
        self.modelchinese = self.modelchinese.load("svmchinese.dat")

    def save(self):
        self.modelchar.save("svmchar.dat")
        self.modelchinese.save("svmchinese.dat")

    def train(self):
        inputs = []
        labels = []
        for root, dirs, files in os.walk("images/train/char"):
            for filename in files:
                filepath = os.path.join(root, filename)
                img = cv2.imread(filepath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                inputs.append(cv2.resize(img, (20, 20)))
                labels.append(ord(os.path.basename(root)))
        inputs = fea_vectors(inputs)
        labels = np.array(labels)
        self.modelchar.train(inputs, cv2.ml.ROW_SAMPLE, labels)
        inputs = []
        labels = []
        for root, dirs, files in os.walk("images/train/chinese"):
            if not os.path.basename(root).startswith("zh_"):
                continue
            for filename in files:
                filepath = os.path.join(root, filename)
                digit_img = cv2.imread(filepath)
                digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                inputs.append(cv2.resize(digit_img, (20, 20)))
                labels.append(self.provinces.index(
                    os.path.basename(root))+1001)
        inputs = fea_vectors(inputs)
        labels = np.array(labels)
        self.modelchinese.train(inputs, cv2.ml.ROW_SAMPLE, labels)

    def predict(self, mode, samples):
        if mode == "chinese":
            r = self.modelchinese.predict(samples)
            charactor = self.provinces[int(r[1].ravel()[0]-1000)]
        if mode == "char":
            r = self.modelchar.predict(samples)
            charactor = chr(r[1].ravel()[0])
        return charactor


# 以蓝色调读入图片
image_path = "car3.jpg"
image = cv2.imread("images/test/"+image_path)
ratio = image.shape[0] / 500.0
orig = image.copy()
image = resize(orig, height=500)
blue = cv2blue(image)
cv_show("Blue", blue)

# 形态学操作
kernel = np.ones((5, 15), np.uint8)
morgh = cv2.threshold(blue, 75, 255, cv2.THRESH_BINARY)[1]
morgh = cv2.morphologyEx(morgh, cv2.MORPH_CLOSE, kernel)
cv_show("Morgh", morgh)

# 轮廓检测
cnts, hirearchy = cv2.findContours(
    morgh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
rects = []
contours = image.copy()
for i, c in enumerate(cnts):
    rect = cv2.minAreaRect(c)
    pts = cv2.boxPoints(rect)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    box = np.int0(rect)
    w, h = pts2wh(box)
    if cv2.contourArea(box) > 250 and 1.5*h < w:
        cv2.drawContours(contours, [box], 0, (0, 255, 0), 2)
        cv2.drawContours(contours, cnts, i, (0, 0, 255), 2, 8)
        rects.append(box)
cv_show("Contours", contours)

# 转化图形
for (i, rect) in enumerate(rects):
    w, h = pts2wh(rect*ratio)
    # 变换后对应坐标位置
    rect = np.array(rect, dtype="float32")
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1],
                   [0, h - 1]], dtype="float32")
    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    # 变换矩形
    warped = cv2.warpPerspective(image, M, (w, h))
    # 固定大小
    warped = resize(warped, height=100)
    cv2.imwrite("images/warped/"+image_path, warped)

img = cv2.imread("images/warped/"+image_path)
ori = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0.7*np.max(gray), 255, cv2.THRESH_BINARY)[1]
locs = []

# 对左侧处理
left = cv2.dilate(gray, np.ones((10, 4), np.uint8), iterations=1)
left = cv2.erode(left, np.ones((10, 4), np.uint8), iterations=1)
cnts, hierarchy = cv2.findContours(left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boundingBoxes = [cv2.boundingRect(c) for c in cnts]  # 用一个最小的矩形，把找到的形状包起来x,y,h,w
cnts, boundingBoxes = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][0], reverse=False))

lefter = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
cv2.drawContours(lefter, cnts, -1, (0, 255, 0), 2)
for box in boundingBoxes:
    x, y, w, h = box
    cv2.rectangle(lefter, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv_show('Contours', lefter)

for (i, c) in enumerate(cnts):
    rect = cv2.minAreaRect(c)
    (x, y, w, h) = cv2.boundingRect(c)
    if 1000 < w*h < 4900 and x < 30 and h>50:
        print(x,y,w,h)
        roi = gray[y:y + h, x:x + w]
        locs.append(roi)


# 对右侧数字处理
cnts, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boundingBoxes = [cv2.boundingRect(c)for c in cnts]  # 用一个最小的矩形，把找到的形状包起来x,y,h,w
cnts, boundingBoxes = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][0], reverse=False))

grayer = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
# 这里的 (0, 255, 0) 是轮廓线的颜色，2 是线条宽度
cv2.drawContours(grayer, cnts, -1, (0, 255, 0), 2)
for box in boundingBoxes:
    x, y, w, h = box
    cv2.rectangle(grayer, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv_show('Contours', grayer)

for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    if w*h < 4900 and x > 30 and h > 50:
        print(x,y,w,h)
        roi = gray[y:y + h, x:x + w]
        locs.append(roi)

# 匹配评分
output = []
model = Model(C=1, gamma=0.5)
for i, loc in enumerate(locs):
    cv_show("l", loc)
    w = abs(loc.shape[1] - 20) // 2
    loc = cv2.copyMakeBorder(
        loc, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    loc = cv2.resize(loc, (20, 20), interpolation=cv2.INTER_AREA)
    loc = fea_vectors([loc])
    if i == 0:
        charactor = model.predict("chinese", loc)
    else:
        charactor = model.predict("char", loc)
    output.extend(charactor)
print(output)
