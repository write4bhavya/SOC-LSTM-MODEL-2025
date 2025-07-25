## 📈 Stock Price Prediction using LSTM – Summer of Code Project

Welcome to our Summer of Code project! Our goal is to build an LSTM-based model for predicting stock prices using historical financial data.

---

## ✅ Project Roadmap

### 📘 Week 1: Python Basics
Learned fundamentals of Python:
- Variables, data types, loops, functions
- Conditional statements and I/O operations
- Hands-on practice with simple programs

---

### 🛠️ Week 2: Python Libraries for Data Science
Explored essential libraries:
- **NumPy** – for numerical computing  
- **Pandas** – for data manipulation and analysis  
- **Matplotlib** & **Seaborn** – for data visualization  
- **Scikit-learn** – for machine learning tools  

Assignments:
- Integrated concepts from Week 1 and Week 2 in practical exercises.

---

### 🧠 Week 3: Introduction to Machine Learning + PyTorch Basics
What We Covered:
- What is Machine Learning?
- Core ML Math: Linear algebra, probability, and statistics
- Supervised Learning:
  - Linear Regression (Simple and MLR)
  - Logistic Regression (Binary Classification)
- Overfitting and Regularization (L2, cross-validation)
- Unsupervised Learning:  
  - K-Means Clustering  
  - K-Nearest Neighbors (KNN)

**Plus: Introduction to PyTorch**
- Tensors and operations
- Neural network layers and loss functions
- Autograd and backpropagation in PyTorch

---

### 🧬 Week 4: Introduction to Deep Learning – CNN & RNN
Learned the fundamentals of deep learning:
- **Neural Networks** – forward and backward propagation
- **Convolutional Neural Networks (CNNs)**:
  - Filters, pooling, and convolution operations
  - Applications in image tasks
- **Recurrent Neural Networks (RNNs)**:
  - Sequential modeling
  - Limitations of vanilla RNNs (vanishing gradients)
  - Introduction to LSTM and GRU

---

### 🔁 Week 5: Understanding LSTM Networks
Dived deep into LSTM architecture for time series prediction:
- Cell structure: input, forget, and output gates
- Sequence modeling with memory
- Input reshaping and tensor dimensions for LSTM
- Built LSTM networks in PyTorch:
  - Defined models with `nn.LSTM` and `nn.Linear`
  - Trained on sequential data (e.g., sine waves)

---

### 📊 Week 6: LSTM for Stock Price Prediction
Final implementation week – applying everything built so far:
- Used `yfinance` to fetch historical stock data
- Computed technical indicators:
  - MACD, RSI, EMA, SMA
- Preprocessed data and normalized it
- Generated LSTM training sequences
- Trained an LSTM model using:
  - MSE Loss  
  - Adam Optimizer  
  - CUDA acceleration for GPU training
- Forecasted future stock prices for 15 days
- Visualized:
  - Technical indicators
  - Actual vs Predicted vs Forecasted prices
- Evaluated performance using **R² score**

---

## 💻 Financial Data with yfinance
- What is the `yfinance` API
- How to:
  - Import the library
  - Download stock data using Python
  - Format and prepare data for modeling

---

## 🔧 Tools & Technologies
- **Python 3.x**
- **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**
- **Scikit-learn**
- **PyTorch**
- **yfinance**
- **Jupyter Notebooks / VS Code**
- **CUDA / GPU acceleration** (for model training)

