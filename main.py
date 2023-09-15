import cv2
import mediapipe as mp
from gtts import gTTS
import os

# Função para reproduzir o áudio
def play_audio(text):
    tts = gTTS(text, lang='pt-br')  # Define o idioma como português do Brasil
    tts.save("output.mp3")
    os.system("mpg123 output.mp3")  # Reproduz o áudio com o mpg123

hand = mp.solutions.hands
mpDraws = mp.solutions.drawing_utils

# Configuração da câmera
cap = cv2.VideoCapture(0)

# Configuração do detector de mãos
with hand.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        handsPoints = results.multi_hand_landmarks
        h, w, _ = image.shape
        pontos = []

        if handsPoints:
            for points in handsPoints:
                mpDraws.draw_landmarks(image, points, hand.HAND_CONNECTIONS)
                for id, cord in enumerate(points.landmark):
                    cx, cy = int(cord.x * w), int(cord.y * h)
                    pontos.append((cx, cy))

            dedos = [8, 12, 16, 20]
            contador = 0

            if points:
                if pontos[4][0] < pontos[2][0]:
                    contador += 1
                for x in dedos:
                    if pontos[x][1] < pontos[x - 2][1]:
                        contador += 1
            print(contador)

            # Converta o contador em texto
            text_to_speak = str(contador) + " dedos"
            cv2.putText(image, text_to_speak, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)

            # Reproduza o áudio
            play_audio(text_to_speak)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
