import numpy as np
from ultralytics import YOLO
import cv2
import math
from sort import *

from collections import deque
import serial
import time

import simpleaudio as sa
import random

cap = cv2.VideoCapture(0) #Para camara en vivo
cap.set(3, 640)
cap.set(4, 320)

model = YOLO("../Y_Mag/yolov8n.pt")

ids = ["person","bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
             "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush","cup"]

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

contador = []

mapeo = cv2.imread("../Video Andamio/mapeoblack.png")

###########################################################################
#ubicacion_material = "../Output/resultado.avi"
#codec = cv2.VideoWriter_fourcc('X','V','I','D')
#fps_video = cap.get(cv2.CAP_PROP_FPS)
#fps_video = 15
#resolucion = (640, 320)
#cap_out = cv2.VideoWriter(ubicacion_material, codec, fps_video, resolucion)
###########################################################################
cap_bgrd = cv2.createBackgroundSubtractorKNN(history=100, dist2Threshold=100, detectShadows=False)
frequency_min = 50
frequency_max = 300
sample_rate = 44100
seconds = 5
###########################################################################
data_deque = {}
###########################################################################
colores = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
###########################################################################
e1 = [27, 293, 125, 247]
e2 = [e1[2], e1[1], 222, e1[3]]
e3 = [e2[2], e1[1], 320, e1[3]]
e4 = [e3[2], e1[1], 417, e1[3]]
e5 = [e4[2], e1[1], 515, e1[3]]
e6 = [e5[2], e1[1], 612, e1[3]]

d1 = [e1[0], e1[3], e1[2], 200]
d2 = [e1[2], e1[3], e2[2], d1[3]]
d3 = [e2[2], e1[3], e3[2], d1[3]]
d4 = [e3[2], e1[3], e4[2], d1[3]]
d5 = [e4[2], e1[3], e5[2], d1[3]]
d6 = [e5[2], e1[3], e6[2], d1[3]]

c1 = [e1[0], d1[3], e1[2], 154]
c2 = [e1[2], d1[3], e2[2], c1[3]]
c3 = [e2[2], d1[3], e3[2], c1[3]]
c4 = [e3[2], d1[3], e4[2], c1[3]]
c5 = [e4[2], d1[3], e5[2], c1[3]]
c6 = [e5[2], d1[3], e6[2], c1[3]]

b1 = [e1[0], c1[3], e1[2], 108]
b2 = [e1[2], c1[3], e2[2], b1[3]]
b3 = [e2[2], c1[3], e3[2], b1[3]]
b4 = [e3[2], c1[3], e4[2], b1[3]]
b5 = [e4[2], c1[3], e5[2], b1[3]]
b6 = [e5[2], c1[3], e6[2], b1[3]]

a1 = [e1[0], b1[3], e1[2], 62]
a2 = [e1[2], b1[3], e2[2], a1[3]]
a3 = [e2[2], b1[3], e3[2], a1[3]]
a4 = [e3[2], b1[3], e4[2], a1[3]]
a5 = [e4[2], b1[3], e5[2], a1[3]]
a6 = [e5[2], b1[3], e6[2], a1[3]]

contex = [43, 0, 596, 320]

me1 = [76, 270]
me2 = [173, me1[1]]
me3 = [271, me1[1]]
me4 = [368, me1[1]]
me5 = [446, me1[1]]
me6 = [563, me1[1]]

md1 = [me1[0], 223]
md2 = [me2[0], md1[1]]
md3 = [me3[0], md1[1]]
md4 = [me4[0], md1[1]]
md5 = [me5[0], md1[1]]
md6 = [me6[0], md1[1]]

mc1 = [me1[0], 177]
mc2 = [me2[0], mc1[1]]
mc3 = [me3[0], mc1[1]]
mc4 = [me4[0], mc1[1]]
mc5 = [me5[0], mc1[1]]
mc6 = [me6[0], mc1[1]]

mb1 = [me1[0], 131]
mb2 = [me2[0], mb1[1]]
mb3 = [me3[0], mb1[1]]
mb4 = [me4[0], mb1[1]]
mb5 = [me5[0], mb1[1]]
mb6 = [me6[0], mb1[1]]

ma1 = [me1[0], 85]
ma2 = [me2[0], ma1[1]]
ma3 = [me3[0], ma1[1]]
ma4 = [me4[0], ma1[1]]
ma5 = [me5[0], ma1[1]]
ma6 = [me6[0], ma1[1]]

###########################################################################

while True:

    success, img = cap.read()
    #cv2.normalize(img, img, 0, 5*255, cv2.NORM_MINMAX)

    results = model(img, stream=True)

    detections = np.empty((0, 5))

    ###########################################################################
    camara_musica_bn = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    camara_musica_bn_bgrd = cap_bgrd.apply(camara_musica_bn)
    ###########################################################################
    # marco
    cv2.rectangle(camara_musica_bn, (0, 0), (640, 320), (128, 129, 126), 1)
    #zonificacion
    def draw_rectangle_and_text(img, label, coordinates, text_coordinates):
        cv2.rectangle(img, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), (83, 83, 86), 1)
        cv2.putText(img, label, (text_coordinates[0], text_coordinates[1]), cv2.FONT_HERSHEY_PLAIN, 1, (83, 83, 86), 2)

    # Define coordinates and labels
    coordinates_list = [e6, e5, e4, e3, e2, e1, d6, d5, d4, d3, d2, d1, c6, c5, c4, c3, c2, c1,
                        b6, b5, b4, b3, b2, b1, a6, a5, a4, a3, a2, a1]
    text_labels = ['e6', 'e5', 'e4', 'e3', 'e2', 'e1', 'd6', 'd5', 'd4', 'd3', 'd2', 'd1',
                   'c6', 'c5', 'c4', 'c3', 'c2', 'c1', 'b6', 'b5', 'b4', 'b3', 'b2', 'b1',
                   'a6', 'a5', 'a4', 'a3', 'a2', 'a1']
    text_coordinates_list = [me6, me5, me4, me3, me2, me1, md6, md5, md4, md3, md2, md1, mc6, mc5, mc4, mc3, mc2, mc1,
                             mb6, mb5, mb4, mb3, mb2, mb1, ma6, ma5, ma4, ma3, ma2, ma1]

    # Draw rectangles and text using a loop
    for label, coordinates, text_coordinates in zip(text_labels, coordinates_list, text_coordinates_list):
        draw_rectangle_and_text(img, label, coordinates, text_coordinates)

    ###########################################################################

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box como en el rhino
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2-x1,y2-y1

            conf = math.ceil((box.conf[0]*100))/100

            # Nombre de las identidades
            cls = int(box.cls[0])
            idPresente = ids[cls]

            if idPresente == "person" and conf > 0.3:
                #cv2.putText(img,str(cls),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(255, 0, 255), 2)
                #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1) # esto es lo que detecta yolo
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))

    resultsTracker = tracker.update(detections)
    ###########################################################################
    sound_coordinates = []
    ###########################################################################
    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        print('Coordenadas: ', (x1, y1, x2, y2), 'Id: ', Id)

        w, h = x2-x1, y2-y1
        cv2.rectangle(img, (x1, y1), (x2, y2), (colores[int(Id) % len(colores)]), 2) # esto es lo que detecta sort
        cv2.putText(img, str(int(Id)), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (colores[int(Id) % len(colores)]), 2)

        cx, cy = x1+w//2,y1+h//2
        centro = cx, cy
        cv2.circle(img, (cx,cy), 5, (colores[int(Id) % len(colores)]), cv2.FILLED)
        ###########################################################################
        #contador personas
        if contex[0] < cx < contex[2] and contex[1] < cy < contex[3]:
            if contador.count(Id) == 0:
                contador.append(Id)
        ###########################################################################
        if Id not in data_deque:
            data_deque[Id] = deque(maxlen= 45)

        data_deque[Id].appendleft(centro)
        for i in range(1, len(data_deque[Id])):
            if data_deque[Id][i-1] is None or data_deque[Id][i] is None:
                continue

            grosor = int(np.sqrt(64 / float(i + 1)) * 1.5)

            for vistas_combinadas in [camara_musica_bn, mapeo]:
                cv2.line(vistas_combinadas, data_deque[Id][i - 1], data_deque[Id][i], (colores[int(Id) % len(colores)]), (grosor if vistas_combinadas is camara_musica_bn else 2))
        ###########################################################################
        sound_coordinates.append([int(cx), int(cy)])
        sound_pixels = []
        ###########################################################################
        for i in range(0, len(sound_coordinates)):
            cx, cy = sound_coordinates[i][0], sound_coordinates[i][1]
            boxs = 50
            try:
                sound_pixels.append(np.mean(camara_musica_bn_bgrd[cy - int(boxs / 2): cy + int(boxs / 2),
                                            cx - int(boxs / 2): cx + int(boxs / 2)]))
            except BaseException as error:
                print('An exception occurred: {}'.format(error))

        if len(sound_pixels) > 0 and ~np.isnan(sound_pixels).any():
            frequency = np.interp(np.mean(sound_pixels), (0, 255), (frequency_min, frequency_max))

            print('Pixel sonoro: ', sound_pixels, ' frecuencia: ', frequency)

            breath_duration_before = 0.5
            breath_duration_after = 0.5

            breath_samples_before = int(breath_duration_before * sample_rate)
            breath_samples_after = int(breath_duration_after * sample_rate)

            breath_before = np.zeros(breath_samples_before)
            breath_after = np.zeros(breath_samples_after)

            time_array = np.linspace(0, seconds, seconds * sample_rate, False)

            num_frequencies = 2
            frequencies = np.linspace(frequency - 50, frequency + 50, num_frequencies)

            guttural_effect = np.sum([np.sin(f * time_array * 2 * np.pi) for f in frequencies], axis=0)

            fade_out_duration = 0.9
            fade_out_samples = int(fade_out_duration * sample_rate)

            fade_out_curve = np.linspace(1.0, 0.0, fade_out_samples)
            guttural_effect[-fade_out_samples:] *= fade_out_curve

            # Normalize audio levels
            target_amplitude = 0.5
            guttural_effect = guttural_effect * (target_amplitude / np.max(np.abs(guttural_effect)))

            audio_data = np.concatenate((breath_before, guttural_effect, breath_after))

            audio_data = audio_data * (2 ** 15 - 1) / np.max(np.abs(audio_data))
            audio_data = audio_data.astype(np.int16)

            wave_obj = sa.WaveObject(audio_data, 2, 2, sample_rate)

            play_obj = wave_obj.play()

        ###########################################################################
        #Conexion serial
        rectangles = [
            (e6, "e6"), (e5, "e5"), (e4, "e4"), (e3, "e3"), (e2, "e2"), (e1, "e1"),
            (d6, "d6"), (d5, "d5"), (d4, "d4"), (d3, "d3"), (d2, "d2"), (d1, "d1"),
            (c6, "c6"), (c5, "c5"), (c4, "c4"), (c3, "c3"), (c2, "c2"), (c1, "c1"),
            (b6, "b6"), (b5, "b5"), (b4, "b4"), (b3, "b3"), (b2, "e2"), (b1, "e1"),
            (a6, "a6"), (a5, "a5"), (a4, "a4"), (a3, "a3"), (a2, "a2"), (a1, "a1")
        ]

        for rect, label in rectangles:
            if rect[0] < cx < rect[2] and rect[3] < cy < rect[1]:
                print('zona activa: ', label)
                # ser.write(label.encode() + b'\n')  # Writes the label followed by a newline to the serial port
                cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)
    ###########################################################################
    cv2.putText(img, str(len(contador)), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (83, 83, 86), 2)

    cv2.imshow("Image", img)
    cv2.imshow("Mapeo", mapeo)
    cv2.imshow("camara_musica_bn_bgrd", camara_musica_bn_bgrd)

    #cap_out.write(img)
    #cv2.imwrite(ubicacion_material,mapeo)
    ###########################################################################
    #cv2.imshow("camara_musica_bn", camara_musica_bn_bgrd)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        #ser.close()
        break

cap.release()
cv2.destroyAllWindows()