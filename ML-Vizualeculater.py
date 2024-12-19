import sqlite3
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor 

def generate_training_data():
    angles = np.arange(0, 360, 1)
    sine_values = np.sin(np.radians(angles))
    cosine_values = np.cos(np.radians(angles))
    tangent_values = np.tan(np.radians(angles))
    return angles, sine_values, cosine_values, tangent_values


def train_trigonometric_model():
    angles, sine_values, cosine_values, tangent_values = generate_training_data()
    angles = angles.reshape(-1, 1)


    sine_model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1, warm_start=True, activation='tanh', random_state=42)
    cosine_model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1, warm_start=True, activation='tanh', random_state=42)
    tangent_model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1, warm_start=True, activation='tanh', random_state=42)
    sine_loss = []
    cosine_loss = []
    tangent_loss = []
    sine_accuracy = []
    cosine_accuracy = []
    tangent_accuracy = []

    for i in range(1000):
        sine_model.fit(angles, sine_values)
        cosine_model.fit(angles, cosine_values)
        tangent_model.fit(angles, tangent_values)

        sine_loss.append(sine_model.loss_)
        cosine_loss.append(cosine_model.loss_)
        tangent_loss.append(tangent_model.loss_)

        sine_pred = sine_model.predict(angles)
        cosine_pred = cosine_model.predict(angles)
        tangent_pred = tangent_model.predict(angles)

        sine_accuracy.append(np.mean(np.abs(sine_pred - sine_values) < 0.05) * 100)
        cosine_accuracy.append(np.mean(np.abs(cosine_pred - cosine_values) < 0.05) * 100)
        tangent_accuracy.append(np.mean(np.abs(tangent_pred - tangent_values) < 0.05) * 100)

    models = {
        "sine": sine_model,
        "cosine": cosine_model,
        "tangent": tangent_model
    }

    return models, sine_loss, cosine_loss, tangent_loss, sine_accuracy, cosine_accuracy, tangent_accuracy


def predict_trigonometric_function(models, angle, trig_function):
    if trig_function == "tangent" and angle % 180 == 90:
        return float('inf')
    if trig_function == "sine":
        return models["sine"].predict([[angle]])[0]
    elif trig_function == "cosine":
        return models["cosine"].predict([[angle]])[0]
    elif trig_function == "tangent":
        return models["tangent"].predict([[angle]])[0]
    else:
        return None



def plot_trigonometric_functions(models, angle, sine_loss, cosine_loss, tangent_loss, sine_accuracy, cosine_accuracy, tangent_accuracy):
    angles, sine_values, cosine_values, tangent_values = generate_training_data()


    plt.subplot(2, 2, 1)
    plt.plot(angles, sine_values, label='True Sine', color='blue')
    plt.plot(angles, models["sine"].predict(angles.reshape(-1, 1)), label='Predicted Sine', linestyle='dashed', color='cyan')
    plt.plot(angles, cosine_values, label='True Cosine', color='green')
    plt.plot(angles, models["cosine"].predict(angles.reshape(-1, 1)), label='Predicted Cosine', linestyle='dashed', color='lime')
    plt.plot(angles, tangent_values, label='True Tangent', color='red')
    plt.plot(angles, models["tangent"].predict(angles.reshape(-1, 1)), label='Predicted Tangent', linestyle='dashed', color='orange')
    plt.legend()
    plt.title('Trigonometric Functions - True vs Predicted')
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(sine_loss, label='Sine Loss', color='blue')
    plt.plot(cosine_loss, label='Cosine Loss', color='green')
    plt.plot(tangent_loss, label='Tangent Loss', color='red')
    plt.legend()
    plt.title('Loss during Training')
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(sine_accuracy, label='Sine Accuracy', color='blue')
    plt.plot(cosine_accuracy, label='Cosine Accuracy', color='green')
    plt.plot(tangent_accuracy, label='Tangent Accuracy', color='red')
    plt.legend()
    plt.title('Accuracy during Training')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid()

    plt.subplot(2, 2, 4)
    circle = plt.Circle((0, 0), 1, color='lightgray', fill=False)
    plt.gca().add_artist(circle)

    rad_angle = np.radians(angle)
    true_point = (np.cos(rad_angle), np.sin(rad_angle))
    predicted_point = (models["cosine"].predict([[angle]])[0], models["sine"].predict([[angle]])[0])

    plt.scatter(*true_point, color='green', label=f"True ({angle}°)")
    plt.scatter(*predicted_point, color='orange', label=f"Predicted ({angle}°)")

    plt.text(true_point[0] + 0.1, true_point[1] + 0.1, f"{true_point[1]:.2f}, {true_point[0]:.2f}", color="green")
    plt.text(predicted_point[0] + 0.1, predicted_point[1] + 0.1, f"{predicted_point[1]:.2f}, {predicted_point[0]:.2f}", color="orange")

    plt.plot([0, true_point[0]], [0, true_point[1]], color='green', linestyle='dotted')
    plt.plot([0, predicted_point[0]], [0, predicted_point[1]], color='orange', linestyle='dotted')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.title('Unit Circle - True vs Predicted')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def setup_database():
    conn = sqlite3.connect("results.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        angle INTEGER,
        function TEXT,
        value REAL
    )
    """)
    conn.commit()
    conn.close()

def save_to_database(angle, function, value):
    conn = sqlite3.connect("results.db")
    c = conn.cursor()
    c.execute("INSERT INTO results (angle, function, value) VALUES (?, ?, ?)", (angle, function, value))
    conn.commit()
    conn.close()

def run_gui(models, sine_loss, cosine_loss, tangent_loss, sine_accuracy, cosine_accuracy, tangent_accuracy):
    def on_submit():
        try:
            angle = int(entry_angle.get())
            if not (0 <= angle < 360):
                raise ValueError
            trig_function = combo_function.get()
            value = predict_trigonometric_function(models, angle, trig_function)
            label_result.config(text=f"{trig_function.capitalize()}({angle}°): {value:.2f}")
            save_to_database(angle, trig_function, value)
            plot_trigonometric_functions(models, angle, sine_loss, cosine_loss, tangent_loss, sine_accuracy, cosine_accuracy, tangent_accuracy)
        except ValueError:
            label_result.config(text="Invalid input. Please enter an angle between 0 and 360.")

    root = tk.Tk()
    root.title("Trigonometric Functions Predictor")

    tk.Label(root, text="Enter Angle (0-360):").grid(row=0, column=0, padx=5, pady=5)
    entry_angle = tk.Entry(root)
    entry_angle.grid(row=0, column=1, padx=5, pady=5)

    tk.Label(root, text="Select Function:").grid(row=1, column=0, padx=5, pady=5)
    combo_function = ttk.Combobox(root, values=["sine", "cosine", "tangent"])
    combo_function.grid(row=1, column=1, padx=5, pady=5)
    combo_function.current(0)

    tk.Button(root, text="Submit", command=on_submit).grid(row=2, column=0, columnspan=2, pady=10)

    label_result = tk.Label(root, text="", font=("Arial", 14))
    label_result.grid(row=3, column=0, columnspan=2, pady=10)

    root.mainloop()

if __name__ == "__main__":
    # setup_database()
    models, sine_loss, cosine_loss, tangent_loss, sine_accuracy, cosine_accuracy, tangent_accuracy = train_trigonometric_model()
    run_gui(models, sine_loss, cosine_loss, tangent_loss, sine_accuracy, cosine_accuracy, tangent_accuracy)
