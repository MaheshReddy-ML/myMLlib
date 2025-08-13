🤖 myMLlib – Machine Learning from Scratch in Python
myMLlib is a growing collection of machine learning algorithms implemented from scratch in Python.
The goal is to make ML concepts clear, accessible, and customizable — while keeping the code clean and framework-free.

📌 Currently Includes
📈 Linear Regression (Gradient Descent)

R² Score, MSE metrics

Optional feature/target normalization

Plot regression fit

📊 Logistic Regression (Binary Classification)

Binary Cross-Entropy (Log Loss)

Probability prediction (predict_proba)

Customizable decision threshold

Plot loss convergence

More algorithms (Decision Trees, KNN, Naive Bayes, etc.) coming soon 🚀

🌟 Features
📝 Educational – Learn ML by building it from scratch

⚙️ Customizable – Choose learning rate, epochs, normalization options, thresholds

📦 Lightweight – Requires only Python, NumPy, Matplotlib

🔍 Metrics Included – R² Score, MSE, Log Loss

📊 Visualizations – Plot regression lines & loss curves

📦 Installation
Clone the repository and start using immediately:

bash
Copy
Edit
git clone https://github.com/MaheshReddy-ML/myMLlib.git
cd myMLlib
🚀 Example Usage
Linear Regression:
from myMLlib import MyLinearRegression

X = [1, 2, 3, 4, 5]
y = [3, 4, 2, 5, 6]

model = MyLinearRegression(lr=0.01, epochs=1000)
model.fit(X, y)
predictions = model.predict([6, 7])

print("Predictions:", predictions)
print("R² Score:", model.r2_score(y, model.predict(X)))

model.plot_fit(X, y)
Logistic Regression
from myMLlib import LogisticRegression

X = [0.1, 0.4, 0.5, 0.8, 1.2]
y = [0, 0, 1, 1, 1]

model = LogisticRegression(lr=0.1, epochs=2000)
model.fit(X, y)

print("Probabilities:", model.predict_proba([0.3, 0.9]))
print("Predictions:", model.predict([0.3, 0.9], threshold=0.5))
model.plot_loss()
