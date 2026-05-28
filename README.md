# Robust Machine Learning Pipeline with Logging & Error Handling

A production-grade, single-file Iris classification pipeline built with Python and `scikit-learn`. Every stage of the machine learning lifecycle—from data loading to validation, preprocessing, training, and evaluation—is wrapped in robust exception handling and comprehensive logging. 

Instead of crashing on corrupted data or runtime anomalies, the pipeline fails gracefully, logs distinct `INFO` or `ERROR` tracebacks to both console and a file, and terminates systematically.

## Features

* **End-to-End ML Workflow:** Implements data loading, structured schema validation, data splitting, `DecisionTreeClassifier` training, and model evaluation.
* **Dual-Channel Logging:** Automatically pipes formatted logs (`%(asctime)s - %(levelname)s - %(message)s`) simultaneously to the console (`stdout`) and a local `ml_pipeline.log` file.
* **Defensive Data Validation:** Assures data integrity by checking input types, verifying non-emptiness, and scanning for unexpected missing values (`NaN`s) before training.
* **Graceful Degradation:** Built-in fault tolerance ensures that upstream pipeline failures trigger informative errors and clean state-exits rather than breaking downstream dependencies.
* **Built-in Failure Simulation:** Contains a demonstration mode that intentionally injects anomalies into the dataset to showcase real-time validation catches and error-logging.

---

## Pipeline Architecture

The pipeline executes through a sequence of modular, isolated steps:

1. **Configure Logging:** Initializes handlers for console output and local log files.
2. **Load Data:** Imports the canonical Iris dataset via `sklearn`.
3. **Validate Data:** Audits the structural integrity of the input DataFrame.
4. **Preprocess & Split:** Handles edge-case missing values and splits the data into stratified Train/Test arrays.
5. **Train Model:** Fits a Decision Tree Classifier on the training partition.
6. **Evaluate & Predict:** Benchmarks accuracy performance metrics on the unseen test partition.

---

## Getting Started

### Prerequisites

* Python 3.8 or higher installed on your system.

### Installation

1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
