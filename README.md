# ML-Vizualeculater

Using Python and Machine Learning to Visualize Mathematics

---

## Overview

An inventive project called AI ML-Vizualeculater uses Python visualization tools and machine learning to assist users comprehend mathematical ideas. Through the use of machine learning models and the creation of visual representations of trigonometric functions, this project streamlines theoretical mathematics and closes the gap between abstract ideas and practical implementations.

---

## Important Features

- Construct dynamic representations of trigonometric functions, such as sine, cosine, and tangent on a circle.

- **Machine Learning Integration**: Use angles to predict trigonometric values by training basic regression models.

- **Interactive Turtle Graphics**: Draw geometric representations, such as the unit circle and labeled angles, using Turtle graphics.

- **Output Management**: Store findings and graphics on disk for further review or distribution.

- **Smooth Operation**: Enter trigonometric functions and angles straight into the application for immediate results and visualization.

---

## Utilized Technologies

- **Python**

- **Turtle Graphics**: For making geometric visuals that are interactive.

- **Scikit-learn**: For trigonometric value prediction training machine learning models.

- **NumPy**: For trigonometric data processing and numerical computations.

- **Pillow (PIL)**: For processing and storing the results of visualization.

- **OS & SYS Libraries**: For running scripts and managing files.

- **Machine Learning Models**: Predicting trigonometric functions using linear regression models.

---

## How It Operates

1. **Produce training data**

   - Sine, cosine, and tangent values are computed using angles ranging from 0° to 360°.
   - To forecast these trigonometric values, linear regression models are trained.

2. **Use Turtle Graphics to visualize**:

   - The angles on a trigonometric circle are labeled.
   - The chosen angle and trigonometric function are used to build a triangle.

3. **Forecast and Conserve**:

   - The trigonometric value for the specified angle is predicted by the machine learning model.
   - The outcomes are stored to a text file and shown on the visualization.
   - The visualizations' images are stored in PNG format for later usage.

---

## Setting up

1. **Make a clone of the repository**:

   ```bash
   git clone https://github.com/your-username/ML-Vizualeculater.git
   cd AI-VisualCalculator
   ```

2. **Set up the necessary dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Use the following syntax to launch the program:

```bash
python ML-Vizualeculater.py <function> <angle>
```

- `<function>`: The trigonometric function (tangent, cosine, or sine) that needs to be calculated.
- `<angle>`: The angle in degrees that needs to be used to compute the function.

For instance, to determine the sine of 30°:

```bash
python ML-Vizualeculater.py sine 30
```

---

## Results

- **Text Results**: `outputimages/outputdata.txt` has the trigonometric values.

- **Images**: The outputs of visualization are saved as:
  - `outputimages/drawing.eps`: EPS visualization
  - `outputimages/drawing_no_background.png`: Final PNG image

---

## Folder Structure

```
ML-Vizualeculater/
│
├── ML-Vizualeculater.py       # Main project script
├── requirements.txt          # List of dependencies
├── outputimages/             # Folder for saving outputs
│   ├── outputdata.txt        # Trigonometric values
│   ├── drawing.eps           # EPS visualization
│   └── drawing_no_background.png # Final PNG image
└── README.md                 # Project documentation
```

---

## Taking part

Contributions are welcome to improve this project. If you want to make a contribution:

1. **The repository should be forked**:

   ```bash
   git fork
   ```

2. **Make a fresh branch**:

   ```bash
   git checkout -b feature/your-feature
   ```

3. **Make your modifications and push**:

   ```bash
   git commit -m "Add your feature"
   git push origin feature/your-feature
   ```

4. **Start a pull request**.

---

## License

The MIT License governs the use of this project. For further information, see the LICENSE file.

---

## Make contact

For inquiries, recommendations, or joint ventures, please contact:

- **Email address**: abolfazlsli911@gmail.com , visualizemathematical@gmail.com
- **Author**: Shaghayegh Bagherian and Abolfazl Salehi
