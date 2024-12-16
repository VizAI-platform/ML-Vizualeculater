
import turtle
import math
import numpy as np
import sys
import os
from sklearn.linear_model import LinearRegression
from PIL import Image

# Function to generate training data (angles and corresponding trigonometric values)
def generate_training_data():
    angles = np.arange(0, 360, 1)
    sine_values = np.sin(np.radians(angles))
    cosine_values = np.cos(np.radians(angles))
    tangent_values = np.tan(np.radians(angles))
    
    return angles, sine_values, cosine_values, tangent_values

# Training a simple machine learning model for trigonometric functions
def train_trigonometric_model():
    angles, sine_values, cosine_values, tangent_values = generate_training_data()
    angles = angles.reshape(-1, 1)

    sine_model = LinearRegression()
    sine_model.fit(angles, sine_values)

    cosine_model = LinearRegression()
    cosine_model.fit(angles, cosine_values)

    tangent_model = LinearRegression()
    tangent_model.fit(angles, tangent_values)

    return sine_model, cosine_model, tangent_model

# Function to predict the trigonometric values using the trained models
def predict_trigonometric_function(models, angle, trig_function):
    sine_model, cosine_model, tangent_model = models
    
    if trig_function == "sine":
        return sine_model.predict([[angle]])[0]
    elif trig_function == "cosine":
        return cosine_model.predict([[angle]])[0]
    elif trig_function == "tangent":
        return tangent_model.predict([[angle]])[0]
    else:
        return None

# Function to draw the Trigonometric circle, axes, and labels
def draw_trigonometric_circle(t):
    t.color('black')
    t.penup()
    t.goto(-200, 0)
    t.pendown()
    t.goto(200, 0)
    t.penup()
    t.goto(0, -200)
    t.pendown()
    t.goto(0, 200)

    t.penup()
    t.goto(0, -100)
    t.pendown()
    t.circle(100)

    for angle in range(0, 360, 30):
        radian_angle = math.radians(angle)
        x = 100 * math.cos(radian_angle)
        y = 100 * math.sin(radian_angle)

        t.penup()
        t.goto(0, 0)
        t.pendown()
        t.goto(x, y)

        t.penup()
        t.goto(x, y)
        t.pendown()
        t.write(f"{angle}°", align="center", font=("Arial", 16, "bold"))

# Function to draw and calculate using Turtle and the trained ML model
def draw_triangle_and_calculate(t, angle, trig_function, models):
    t.clear()
    draw_trigonometric_circle(t)
    
    t.pencolor("orange")
    t.pensize(4)
    t.penup()
    t.goto(0, 0)
    t.pendown()
    t.goto(100 * math.cos(math.radians(angle)), 100 * math.sin(math.radians(angle)))
    t.goto(100 * math.cos(math.radians(angle)), 0)

    value = predict_trigonometric_function(models, angle, trig_function)

    result_text = f"{trig_function.capitalize()}({angle}°): {value:.2f}"
    print(result_text)

    # Ensure output folder exists
    if not os.path.exists("./outputimages"):
        os.makedirs("./outputimages")

    with open("./outputimages/outputdata.txt", "a") as o:
        o.write(result_text + "\n")

    t.penup()
    t.goto(50 * math.cos(math.radians(angle)), 50 * math.sin(math.radians(angle)))
    t.pendown()
    t.write(f"{trig_function.capitalize()}: {value:.2f}", align="center")

# Main function to execute the drawing and machine learning process
def main(action, num):
    models = train_trigonometric_model()

    t = turtle.Turtle()
    s = turtle.Screen()
    s.tracer(0)
    t.speed(0)
    t.pensize(2)

    draw_triangle_and_calculate(t, num, action, models)

    con = turtle.getcanvas()
    con.postscript(file="./outputimages/drawing.eps")
    img = Image.open("./outputimages/drawing.eps")
    img = img.convert("RGBA")
    data = img.getdata()

    new_data = []
    for item in data:
        if item[0] in range(200, 256) and not (item[0] == 255 and item[1] == 165 and item[2] == 0):
            new_data.append((255, 255, 255, 0))  # Transparent
        else:
            new_data.append(item)

    img.putdata(new_data)
    img.save("./outputimages/drawing_no_background.png", "PNG")

    sys.exit(0)

# Ensure the output folder exists before running the main function
if __name__ == "__main__":
    if not os.path.exists("./outputimages"):
        os.makedirs("./outputimages")

    main(sys.argv[1], int(sys.argv[2]))
