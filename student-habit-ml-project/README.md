# Student Habit Performance Machine Learning Project

## Overview
This project aims to analyze the relationship between student habits and their academic performance using machine learning techniques. The dataset contains various metrics related to student habits and their corresponding performance outcomes.

## Project Structure
```
student-habit-ml-project
├── data
│   └── student_habits_performance.csv
├── notebooks
│   └── data_analysis.ipynb
├── src
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
├── requirements.txt
└── README.md
```

## Dataset
The dataset used for this project is located in the `data` directory. It includes information on student habits and performance metrics.

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd student-habit-ml-project
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
- To perform exploratory data analysis, open the Jupyter notebook located in the `notebooks` directory:
  ```
  jupyter notebook notebooks/data_analysis.ipynb
  ```

- For data preprocessing, run the `data_preprocessing.py` script in the `src` directory.

- To train the machine learning model, execute the `model_training.py` script.

- To evaluate the model's performance, use the `evaluation.py` script.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License.