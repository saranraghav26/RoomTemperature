import pandas as pd
import random
import cv2
from sklearn.ensemble import RandomForestRegressor

try:
    df = pd.read_csv('datas.csv')
    print("Dataset Loaded")
except:
    df = pd.DataFrame(columns=["Temp", "Humidity", "People", "Fan", "AC"])
    print("New Dataset Created")


def count_people():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not working")
        return None

    print("Press Q to capture")
    people_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        people_count = len(faces)

        cv2.putText(
            frame,
            f'People: {people_count}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return people_count


def generate_environment():
    temp = random.randint(24, 38)
    humidity = random.randint(40, 85)
    return temp, humidity


def rule_based(temp, humidity, people):
    fan = min(5, max(1, people + temp // 10))
    ac = max(18, 30 - people - (temp // 5))
    return fan, ac


def train_model(df):
    X = df[["Temp", "Humidity", "People"]]
    y_fan = df["Fan"]
    y_ac = df["AC"]

    model_fan = RandomForestRegressor(n_estimators=100)
    model_ac = RandomForestRegressor(n_estimators=100)

    model_fan.fit(X, y_fan)
    model_ac.fit(X, y_ac)

    return model_fan, model_ac


print("Opening camera...")

people = count_people()

if people is None:
    print("Fix camera first")
    exit()

temp, humidity = generate_environment()

print(f"\nTemp: {temp}")
print(f"Humidity: {humidity}")
print(f"People: {people}")

if len(df) > 10:
    print("\nUsing Machine Learning")

    model_fan, model_ac = train_model(df)

    input_data = pd.DataFrame(
        [[temp, humidity, people]],
        columns=["Temp", "Humidity", "People"]
    )

    fan = int(model_fan.predict(input_data)[0])
    ac = int(model_ac.predict(input_data)[0])

else:
    print("\nUsing Rule-Based (collecting data...)")

    fan, ac = rule_based(temp, humidity, people)


new_row = pd.DataFrame(
    [[temp, humidity, people, fan, ac]],
    columns=["Temp", "Humidity", "People", "Fan", "AC"]
)

df = pd.concat([df, new_row], ignore_index=True)
df.to_csv("datas.csv", index=False)

print("\nOutput:")
print("Fan Speed:", fan)
print("AC Temp:", ac)