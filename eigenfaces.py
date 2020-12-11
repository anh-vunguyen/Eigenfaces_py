import cv2
import numpy as np

# Useful coefficients
# Height
H = 192
# Width
W = 168
# Number of class
NoC = 30
# Number of images per class
NoI = 15
# Dimension of each image
dimImg = 32256
# 50 largest eigen vectors
nb = 50

# Load HaarCascade Frontal Face File
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Video Capture
cap = cv2.VideoCapture(0)

# Load matrices created from Matlab
M = np.loadtxt('meanImg.txt')
vectorL = np.loadtxt('vectorL_PCA.txt', delimiter=',')
Coeff = np.loadtxt('Coeff_PCA.txt', delimiter=',')
A = np.loadtxt('ImDatabase.txt', delimiter=',')

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    inputImg = np.zeros((H, W))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (214, 255, 10), 1)

        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (192, 192))
        inputImg = roi_gray[:, 12:180]
        cv2.imshow('ROI', inputImg)

    # Show Face Detection
    cv2.imshow('img', img)

    # Face Recognition with method "EigenFace"
    inputImg = inputImg/255
    inputImg_db = np.reshape(inputImg, (H*W, 1), order="F")

    # Test
    # inputImg_db = np.copy(A[:, 0])

    for x in range(H*W):
        inputImg_db[x] = inputImg_db[x] - M[x]

    inputCoeff = np.dot(np.transpose(inputImg_db), vectorL)

    # Result
    comp_Coeff = np.zeros(NoC*NoI)
    tmp_res = 0

    for k in range(NoI*NoC):
        for t in range(nb):
            tmp_res = tmp_res + (inputCoeff[0, t] - Coeff[k, t]) * (inputCoeff[0, t] - Coeff[k, t])
        comp_Coeff[k] = np.sqrt(tmp_res)
        tmp_res = 0

    MinCoeff = np.amin(comp_Coeff)
    I = np.argmin(comp_Coeff)
    print('MinCoeff = ', MinCoeff, ' ', ' I = ', I)

    # Show Database Image
    Org = np.multiply(A[:, I], 255)
    # Org = np.round(Org)
    imgRes = np.array(Org, dtype=np.uint8)
    imgResReshaped = np.reshape(imgRes, (H, W), order="F")
    cv2.imshow('Database Image', imgResReshaped)


    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        print(inputCoeff)
        print(comp_Coeff)
        break

cap.release()
cv2.destroyAllWindows()
