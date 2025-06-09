# Fraud Detection Dashboard

A Streamlit-based dashboard for monitoring and analyzing potential fraudulent activities in a behavior-based authentication system.

## Features

- Real-time fraud alerts monitoring
- Fraud pattern analysis
- Investigation tools
- Statistical analysis and visualization
- Case study management
- Temporal analysis of fraud patterns

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

To run the Streamlit dashboard:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Integration with Behavior Monitor

This fraud detection module is designed to work with a behavior monitoring system. To integrate with your behavior monitor:

1. Create a `behavior_monitor.py` file with your behavior monitoring implementation
2. Update the `FraudDetection` class to consume behavior data from your monitor
3. Implement the necessary data exchange methods between the two systems

## Note

Currently, the system uses simulated data for demonstration purposes. Replace the simulated data methods with actual data from your behavior monitoring system when integrating. 