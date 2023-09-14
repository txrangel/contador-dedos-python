import cv2
import mediapipe as mp

hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=1)
mpDraws = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0)
with hand.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = cv2.flip(image,1)
    results = hands.process(image)

    handsPoints = results.multi_hand_landmarks
    h,w,_ = image.shape
    pontos = []

    if handsPoints:
        for points in handsPoints:
            #print(points)
            mpDraws.draw_landmarks(image, points, hand.HAND_CONNECTIONS)
            for id,cord in enumerate(points.landmark):
                cx,cy = int(cord.x*w), int(cord.y*h)
                #cv2.putText(image,str(id),(cx,cy+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                pontos.append((cx,cy))
                #print(pontos)

        dedos = [8,12,16,20]
        contador = 0

        if points:
            if pontos[4][0] < pontos [2][0]:
                contador += 1
            for x in dedos:
                if pontos[x][1] < pontos[x-2][1]:
                    contador += 1
        print(contador)

        cv2.putText(image,str(contador),(100,100),cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255),5)

    # Draw the hand annotations on the image.
    # image.flags.writeable = True
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # if results.multi_hand_landmarks:*#
    #   for hand_landmarks in results.multi_hand_landmarks:
    #     mp_drawing.draw_landmarks(
    #         image,
    #         hand_landmarks,
    #         mp_hands.HAND_CONNECTIONS,
    #         mp_drawing_styles.get_default_hand_landmarks_style(),
    #         mp_drawing_styles.get_default_hand_connections_style())

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
