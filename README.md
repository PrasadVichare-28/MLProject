# MLProject: Student Test Score Prediction

## Overview
This project is designed to predict students' test scores based on various influencing factors using machine learning. By analyzing data on student performance, the project aims to uncover key patterns and insights, enabling effective interventions to improve academic outcomes. 

Key highlights include:
- **Structured project design** using a modular coding approach.
- **Logger, Exception Handling** By creating modules for logging and exception handling 
- **Pipeline creation** to streamline processes such as data preprocessing, feature engineering, and model training.
- **Containerization** with Docker for efficient and portable deployment.
- **Hosting on AWS** using Elastic Container Registry (ECR) and EC2 for scalability and reliability.
- **CI/CD pipeline** using Github actions and workflow

## Dataset
- **Source**: [Student Performance in Mathematics Dataset](https://www.kaggle.com/datasets/rkiattisak/student-performance-in-mathematics/data)
- This dataset contains information on the performance of high school students in mathematics, including their grades and demographic information. The data was collected from three high schools in the United States.
- Columns
- Gender: The gender of the student (male/female)
- Race/ethnicity: The student's racial or ethnic background (Asian, African-American, Hispanic, etc.)
- Parental level of education: The highest level of education attained by the student's parent(s) or guardian(s)
- Lunch: Whether the student receives free or reduced-price lunch (yes/no)
- Test preparation course: Whether the student completed a test preparation course (yes/no)
- Math score: The student's score on a standardized mathematics test
- Reading score: The student's score on a standardized reading test
- Writing score: The student's score on a standardized writing test


## Features of the Project
1. **Data Preprocessing**:
   - Handled missing data and outliers to ensure data quality.
   - Encoded categorical variables for model compatibility.
   - Normalized numerical features to maintain consistency.

2. **Feature Engineering**:
   - Identified and selected key features impacting student performance.
   - Created derived metrics to enhance model predictions.

3. **Model Development**:
   - Implemented regression and classification models to predict test scores.
   - Model used: Linear Regression, Gradient Boosting,K-Neighbors Regressor, Decision Tree, Random Forest Regressor, XGBRegressor, CatBoosting Regressor, AdaBoost Regressor
   - Performed hyperparameter tuning to improve model performance.
   - Evaluated models using metrics such as Mean Squared Error (MSE) and R-squared.
   - Saved trained model into pkl file 

4. **Project Structure**:
   - Adopted a modular approach with distinct folders for data, source code, pipelines, artifacts and tests.
   - Each pipeline handles a specific phase of the project for better organization and reusability.

5. **Deployment**:
   - Containerized the application using Docker for consistent deployment environments.
   - **My docker Image**: [Docker image of my project](https://hub.docker.com/r/prasad2896/studentscorepredictor)
   - Hosted the Dockerized application on AWS using ECR for image storage and EC2 for runtime execution.
   - Developed CI/CD pipeline using Github actions and workflows.

## Project Structure
```plaintext
MLProject
── artifacts
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── data.csv
│   ├── test.csv
│   └── train.csv
── Logs
│     
├── data
│   ├── raw_data.csv
│   └── processed_data.csv
├── notebooks
│   ├── EDA_data_analysis.ipynb
│   └── model_training.ipynb
├── src
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   └── model_trainer.py
├── pipelines
│   ├── perdict_pipeline.py
│   └── train_pipeline.py
├── docker
│   ├── Dockerfile
│   └── requirements.txt
├── Components
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Docker
- AWS account with ECR and EC2 setup

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/PrasadVichare-28/MLProject.git
   ```
2. Navigate to the project directory:
   ```bash
   cd MLProject
   ```
3. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project
#### Local Execution
1. Run the data pipeline to preprocess, inject the dataset and train model:
   ```bash
   python src/components/data_ingestion.py
   ```
2. To run appliaction after generation pkl files for model:
   ```bash
   python application.py
   ```


#### Dockerized Execution
1. Build the Docker image:
   ```bash
   docker build -t mlproject:latest .
   ```
2. Run the Docker container:
   ```bash
   docker run -p 5000:5000 mlproject:latest
   ```



## Results
- **Predictive accuracy**: Achieved an R-squared value of **87%** with **Linear Regression**, indicating the model’s effectiveness.


## Technologies Used
- **Programming Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, flask
- **Deployment Tools**: Docker, AWS (ECR, EC2), Github actions



## Contact
- **Author**: Prasad Vichare
- **LinkedIn**: [Prasad Vichare](https://www.linkedin.com/in/prasad-vichare)

