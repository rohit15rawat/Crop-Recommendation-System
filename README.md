**Crop Recommendation System**

### Overview
The Crop Recommendation System is an intelligent tool designed to predict the most suitable crop for cultivation based on soil and environmental parameters. By leveraging machine learning algorithms, the system analyzes factors such as nutrient content, pH level, and other soil properties to provide accurate recommendations.

### Features
The system employs multiple machine learning models, including:
- Decision Tree
- Naive Bayes
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest

### Dataset
Each dataset sample consists of the following soil attributes:
- **N** (Nitrogen) - ppm
- **P** (Phosphorus) - ppm
- **K** (Potassium) - ppm
- **pH** level
- **EC** (Electrical Conductivity) - mS/cm
- **S** (Sulfur) - ppm
- **Cu** (Copper) - ppm
- **Fe** (Iron) - ppm
- **Mn** (Manganese) - ppm
- **Zn** (Zinc) - ppm
- **B** (Boron) - ppm

The system predicts one of the following crop types:
- Grapes
- Mango
- Mulberry
- Pomegranate
- Potato
- Ragi

### Model Performance
The accuracy of different machine learning models is as follows:

| Model               | Accuracy |
|---------------------|----------|
| Decision Tree      | 0.94     |
| Naive Bayes       | 0.97     |
| SVM               | 0.91     |
| Logistic Regression | 0.96     |
| Random Forest     | 0.97     |

### Usage
#### Predicting a Crop
1. **Train the Model**: Train the system using the dataset and chosen algorithms.
2. **Make Predictions**: Input soil parameters into the trained model to receive crop recommendations.

#### Sample Prediction
```python
import numpy as np
# Sample input data
sample_data = np.array([[150, 70, 217, 6, 0.6, 0.25, 10, 116, 60, 55, 22]])
# Predicting using Random Forest model
prediction = RF.predict(sample_data)
print(prediction)  # Output: ['pomegranate']
```

### GUI Application
A graphical user interface (GUI) built with Tkinter enhances usability for farmers and agronomists.

#### Running the GUI Application
1. **Install Dependencies**: Ensure Tkinter and other necessary libraries are installed.
2. **Execute the Script**: Run the provided Python script to launch the interface.

#### GUI Implementation
```python
import tkinter as tk
from tkinter import ttk

def predict_crop():
    data = [float(entry.get()) for entry in entry_fields]
    prediction = RF.predict([data])
    result_label.config(text=f'Predicted Crop: {prediction[0]}')

root = tk.Tk()
root.title("Crop Prediction")
root.geometry("500x400")

labels = ['N (Nitrogen)', 'P (Phosphorus)', 'K (Potassium)', 'pH', 'EC (Electrical Conductivity)',
          'S (Sulfur)', 'Cu (Copper)', 'Fe (Iron)', 'Mn (Manganese)', 'Zn (Zinc)', 'B (Boron)']
entry_fields = []
units = ['ppm', 'ppm', 'ppm', '', 'mS/cm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm']

for idx, label in enumerate(labels):
    ttk.Label(root, text=f"Enter {label} ({units[idx]}):").grid(row=idx, column=0, padx=10, pady=5, sticky="e")
    entry = ttk.Entry(root)
    entry.grid(row=idx, column=1, padx=10, pady=5, sticky="w")
    entry_fields.append(entry)

predict_button = ttk.Button(root, text="Predict", command=predict_crop)
predict_button.grid(row=len(labels), column=0, columnspan=2, pady=10)

result_label = ttk.Label(root, text="")
result_label.grid(row=len(labels)+1, column=0, columnspan=2)

root.mainloop()
```

### Conclusion
The Crop Recommendation System integrates multiple machine learning algorithms to deliver reliable and accurate crop recommendations based on soil characteristics. Its high performance and user-friendly interface make it an invaluable tool for farmers and agronomists seeking to optimize agricultural productivity.

