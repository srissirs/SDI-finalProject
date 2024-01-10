import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from openai import OpenAI
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import time
import pandas as pd

predicted_word = ""

model = load_model('smnist.h5')

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

_, frame = cap.read()

h, w, c = frame.shape

saved_words = []
img_counter = 0
analysisframe = ''
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
while True:
    _, frame = cap.read()

    # Display predicted word with highest confidence
    cv2.rectangle(frame, (10, 10), (600, 50), (255, 255, 255), -1)
    cv2.putText(frame, f"Predicted Word: {predicted_word}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
    
        analysisframe = frame
        showframe = analysisframe
        cv2.imshow("Frame", showframe)
        framergbanalysis = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2RGB)
        resultanalysis = hands.process(framergbanalysis)
        hand_landmarksanalysis = resultanalysis.multi_hand_landmarks
        if hand_landmarksanalysis:
            for handLMsanalysis in hand_landmarksanalysis:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lmanalysis in handLMsanalysis.landmark:
                    x, y = int(lmanalysis.x * w), int(lmanalysis.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                y_min -= 20
                y_max += 20
                x_min -= 20
                x_max += 20 

        analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
        analysisframe = analysisframe[y_min:y_max, x_min:x_max]
        analysisframe = cv2.resize(analysisframe,(28,28))


        nlist = []
        rows,cols = analysisframe.shape
        for i in range(rows):
            for j in range(cols):
                k = analysisframe[i,j]
                nlist.append(k)
        
        datan = pd.DataFrame(nlist).T
        colname = []
        for val in range(784):
            colname.append(val)
        datan.columns = colname

        pixeldata = datan.values
        pixeldata = pixeldata / 255
        pixeldata = pixeldata.reshape(-1,28,28,1)
        prediction = model.predict(pixeldata)
        predarray = np.array(prediction[0])
        letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
        predarrayordered = sorted(predarray, reverse=True)
        high1 = predarrayordered[0]
        high2 = predarrayordered[1]
        high3 = predarrayordered[2]
        for key,value in letter_prediction_dict.items():
            if value==high1:
                print("Predicted Character 1: ", key)
                print('Confidence 1: ', 100*value)
            elif value==high2:
                print("Predicted Character 2: ", key)
                print('Confidence 2: ', 100*value)
            elif value==high3:
                print("Predicted Character 3: ", key)
                print('Confidence 3: ', 100*value)
        time.sleep(5)
        for key, value in letter_prediction_dict.items():
            if value == high1:
                predicted_word += key
                break
    elif k%256 == 13:  # Enter to save word
        saved_words.append(predicted_word)
        print(saved_words)
        predicted_word = "" 
    elif k%256 == 8:  # Backspace to delete character 
        predicted_word = predicted_word[:-1]
    elif k%256 != 255:  
        predicted_word += chr(k%256).upper()


    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()

client = OpenAI(api_key='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')

content = "You act as a story generator and I will give you a list of six words that describe the 'protagonist', 'antagonist', 'occupation of the protagonist', 'location', 'the protagonist's weakness' and 'motivation of the antagonist'. Then with these six words that i give you, you will make a short story not longer than 10 sentences and you can get creative and surrealistic, try to avoid a typical storyline. Prompts: "

prompts = ', '.join(saved_words)
content = content + prompts

client = OpenAI(api_key='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')

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": content
        }
    ],
    model="gpt-3.5-turbo",
)

print(chat_completion.choices[0].message.content)
