# contador/contador.py
import cv2
import numpy as np
import datetime
import os
import psycopg2

# Conectar ao banco de dados usando a variável de ambiente
DATABASE_URL = os.getenv("DATABASE_URL")
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# Cria a tabela se ela não existir
cursor.execute('''
    CREATE TABLE IF NOT EXISTS contagem (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP,
        count INTEGER
    )
''')
conn.commit()

def salvar_contagem(contagem):
    ts = datetime.datetime.now()
    cursor.execute("INSERT INTO contagem (timestamp, count) VALUES (%s, %s)", (ts, contagem))
    conn.commit()

# Carregar o modelo YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Abrir a captura de vídeo (pode ser substituído por uma stream da câmera, ex.: 0 para webcam)
cap = cv2.VideoCapture("video_avenida.mp4")  # Substitua "video_avenida.mp4" pelo caminho do seu vídeo ou use 0 para webcam

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define a linha de contagem: por exemplo, uma linha horizontal em 70% da altura do frame
line_y = int(frame_height * 0.7)

vehicle_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Pré-processar o frame para o YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Processar as detecções
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Considera apenas classes que representam veículos (exemplo: carro, caminhão, ônibus, moto)
            if confidence > 0.5 and classes[class_id] in ["car", "bus", "truck", "motorbike"]:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Remover detecções duplicadas
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Para cada detecção válida, desenha a caixa e verifica se o centro cruzou a linha
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Desenha a caixa e o centro do veículo
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), -1)
        
        # Verifica se o centro do veículo está próximo da linha de contagem
        margin = 5  # margem para evitar ruído
        if (line_y - margin) <= center_y <= (line_y + margin):
            vehicle_count += 1
            salvar_contagem(1)
            # Desenha uma indicação para o veículo já contado
            cv2.putText(frame, "Contado", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Desenhar a linha de contagem
    cv2.line(frame, (0, line_y), (frame_width, line_y), (255, 0, 0), 2)
    cv2.putText(frame, f"Contagem: {vehicle_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Contagem de Veiculos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
