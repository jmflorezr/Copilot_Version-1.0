
import os
import cv2



def tomarPunto(idImg, image_draw, colorPunto):
    points = []
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
    cv2.namedWindow(idImg)
    cv2.setMouseCallback(idImg, click)
    points1 = []
    point_counter = 0
    while True:
        cv2.imshow(idImg, image_draw)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("x"):
            points1 = points.copy()
            points = []
            break
        if len(points) > point_counter:
            point_counter = len(points)
            cv2.circle(image_draw, (points[-1][0], points[-1][1]), 3, colorPunto, -1)
            print(points)
    cv2.destroyWindow(idImg) #una vez selecionados los puntos cierra la imagen
    return points1 # retorna la imagen con los puntos y los puntos



red = [0, 0, 255] #color para pintar puntos


cap = cv2.VideoCapture("Autopista_resize.mp4")
obj_detec = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)#20
#obj_detec = cv2.createBackgroundSubtractorMOG2()
ret, frame = cap.read()
punto1 = tomarPunto("Primer frame", frame, red) #llama al metodo que pinta rojo
print(punto1)


while True:
    ret, frame = cap.read()
    alto, ancho, _ = frame.shape
    print(alto, ancho)
    areaInteres = frame[150:300, 100:400] #[200:300, 250:400]
    mask = obj_detec.apply(areaInteres)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)#ayuda a eliminar sombras#254
    #mask = obj_detec.apply(frame)
    contornos, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#encuentra los contornos
    for cont in contornos:
        area = cv2.contourArea(cont)#calcula el area de cada contorno
        if area > 200: #visualiza los contornos con areas mayores #800
            #cv2.drawContours(areaInteres, [cont], -1, (0, 0, 255), 2)# dibuja solo en el area de interes
            #cv2. drawContours(frame, [cont], -1, (0, 0, 255), 2)# Dibuja en toda la imagen
            x, y, w, h = cv2.boundingRect(cont)# extrae los puntos del rectangulo de cada contorno
            #calcula el punto medio de cada rectangulo
            centroX = int((2 * x + w) / 2)
            centroY = int((2 * y + h) / 2)
            cv2.circle(areaInteres, (centroX, centroY), 5, (0, 0, 255), -1)#dibuja un circulo
            cv2.rectangle(areaInteres, (x, y), (x + w, y + h), (255, 0, 0), 2)# dibuja rectangulos sobre cada contorno

    cv2.imshow("area de interes", areaInteres)
    cv2.imshow("mask", mask)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(30)
    if key ==27:
        break
cap.release()
cv2.destroyAllWindows()
