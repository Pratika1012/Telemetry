# import streamlit as st
# import pandas as pd
# import numpy as np
# from io import StringIO
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression

# # CSS Styling for Red and White Theme
# st.markdown("""
#     <style>
#         /* General Styling */
#         body {
#             background-color: #ffffff;  /* White background */
#             color: #000000;            /* Black text for contrast */
#         }

#         /* Header Styling */
#         .header {
#             text-align: center;
#             margin-top: 20px;
#         }

#         .title {
#             font-size: 28px;
#             font-weight: bold;
#             color: red;  /* Thermo Fisher Red */
#             align-items: center;
#         }

#         /* Input/Output Boxes */
#         .box {
#             border: 3px solid #e60012;
#             border-radius: 20px;
#             padding: 20px;
#             margin-bottom: 30px;
#             background-color: #ffffff;
#             box-shadow: 0 0 10px rgba(0,0,0,0.05);
#         }

#         .output-box {
#             border: 2px solid #e60012;
#             border-radius: 15px;
#             padding: 15px;
#             margin-top: 20px;
#             background-color: #ffffff;
#         }

#         .output-box ul {
#             list-style-type: disc;
#             padding-left: 20px;
#             text-align: left;
#             color: #000000;
#         }

#         /* Text and Input Styling */
#         .stTextArea textarea, .stTextInput input {
#             color: #000000;
#             background-color: #ffffff;
#             border: 1px solid #e60012;
#             border-radius: 5px;
#         }

#         /* Button Styling */
#         .stButton > button {
#             background-color: #e60012;  /* Red */
#             color: #ffffff;             /* White text */
#             width: 200px;
#             height: 40px;
#             border: 2px solid #ffffff;
#             border-radius: 25px;
#             cursor: pointer;
#         }

#         .stButton > button:hover {
#             background-color: #ffffff;
#             color: #e60012;
#             border: 2px solid #e60012;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Header
# st.markdown("<h1 class='title'><center>Telemetry Analysis</h1>", unsafe_allow_html=True)

# # File Uploader for Text File
# uploaded_file = st.file_uploader("📂 Upload your Telemetry Data file", type=["puc"])

# if uploaded_file is not None:
#     st.success("✅ File uploaded successfully!")
    
#     # Read the uploaded text file
#     raw_data = uploaded_file.read().decode("utf-8")

#     # Display file details
#     st.write("**Filename:**", uploaded_file.name)
#     st.write("**File size (bytes):**", uploaded_file.size)

#     # --- Data Processing ---
#     lines = raw_data.split('\n')
#     columns = [
#         'Date/Time', 'RTD', 'TC1', 'TC2', 'TC3', 'TC4', 'TC6', 'TC7', 'TC9', 'TC10',
#         'Setpoint', 'Line Input', 'PUC State', 'User Offset',
#         'Warm Warning Setpoint', 'Cold Warning Setpoint', 'Stage 1 RPM', 'Stage 2 RPM',
#         'Total Valve Steps', 'Condenser Fan RPM', 'Algorithm Flags', 
#         'Algorithm State', 'BUS RTD', 'Valve Steps', 'S1 Pressure',
#         'TC8', 'Superheat', 'S2 Temperature', 'TSat'
#     ]
    
#     # Filter and clean data, removing unwanted lines
#     data_lines = [line for line in lines if line and not line.startswith('PUC_VER')]
    
#     # Parse the lines to create a DataFrame
#     data_str = "\n".join(data_lines)
#     df = pd.read_csv(StringIO(data_str), header=None)
#     df.columns = columns[:df.shape[1]]

#     # Convert Date/Time column to datetime
#     df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')

#     # Dropping columns with zero mean values
#     zero_mean_cols = [
#         col for col in df.columns 
#         if col != 'Date/Time' and pd.to_numeric(df[col], errors='coerce').mean() == 0]
#     df = df.drop(columns=zero_mean_cols)

#     # --- Compute Trend Using Linear Regression ---
#     def compute_slope(series):
#         y = series.values
#         x = np.arange(len(y)).reshape(-1, 1)
#         if len(y) < 2 or np.all(np.isnan(y)):
#             return np.nan
#         model = LinearRegression().fit(x, y)
#         return model.coef_[0]  # slope

#     # Ensure 'TC1' and 'TC10' are numeric before applying rolling
#     df['TC1'] = pd.to_numeric(df['TC1'], errors='coerce')
#     df['TC10'] = pd.to_numeric(df['TC10'], errors='coerce')

#     # Compute Trend Using Linear Regression
#     df['TC1_trend'] = df['TC1'].rolling(60).apply(compute_slope, raw=False)
#     df['TC10_trend'] = df['TC10'].rolling(60).apply(compute_slope, raw=False)
#     df['PUC_state_mean'] = df['PUC State'].rolling(60).mean()

#     # Flag sustained trend condition
#     df['Trend_Flag'] = (df['TC1_trend'] < 0) & (df['TC10_trend'] > 0) & (df['PUC_state_mean'] == 1)

#     # Identify sustained sequences (3+ consecutive flags)
#     def flag_sustained(df, col='Trend_Flag', min_consecutive=3):
#         sustained_flags = [False] * len(df)
#         count = 0
#         for i, val in enumerate(df[col]):
#             if val:
#                 count += 1
#                 if count >= min_consecutive:
#                     for j in range(i - count + 1, i + 1):
#                         sustained_flags[j] = True
#             else:
#                 count = 0
#         return sustained_flags

#     df['Sustained_Issue'] = flag_sustained(df)

#     # Create Issue_Detected column based on sustained issues
#     df['Issue_Detected'] = df['Sustained_Issue'].astype(int)  # Convert Boolean to Integer

#     # Train Model
#     features = ['TC1_trend', 'TC10_trend', 'PUC_state_mean']
#     df_model = df[features + ['Issue_Detected']].dropna()
    
#     X = df_model[features]
#     y = df_model['Issue_Detected']
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#     clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     clf.fit(X_train, y_train)

#     y_pred_test = clf.predict(X_test)

#     accuracy = clf.score(X_test, y_test) * 100
#     st.write(f"Model Accuracy: {accuracy:.2f}%")

#     # ------------------ STEP 4: Train Model ------------------
#     features = ['TC1_trend', 'TC10_trend', 'PUC_state_mean']

#     print(df.shape)
#     # Drop rows with NaNs in either X or y
#     df_model = df[features + ['Issue_Detected']].dropna()
#     print(df_model.shape)

#     X = df_model[features]
#     y = df_model['Issue_Detected']

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#     # Model training
#     clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     clf.fit(X_train, y_train)

#     # Predictions
#     y_pred_test = clf.predict(X_test)
#     y_pred_train = clf.predict(X_train)

#     # === Evaluation ===
#     print("=== Train Classification Report ===")
#     print(classification_report(y_train, y_pred_train))

#     print("\n=== Test Classification Report ===")
#     print(classification_report(y_test, y_pred_test))

#     print("\n=== Confusion Matrix (Test) ===")
#     st.write(confusion_matrix(y_test, y_pred_test))

#     # ------------------ STEP 5: GenAI-style Summary Generation ------------------
#     total_rows = len(df)
#     print(total_rows)
#     total_issues = df['Issue_Detected'].sum()
#     print(total_issues)
#     accuracy = clf.score(X_test, y_test) * 100
#     print(accuracy,'%')

#     # Plot feature importance
#     importances = clf.feature_importances_
#     sorted_indices = np.argsort(importances)
#     sorted_features = [features[i] for i in sorted_indices]
#     sorted_importances = importances[sorted_indices]

#     plt.figure(figsize=(8, 5))
#     bars = plt.barh(sorted_features, sorted_importances, color='skyblue')
#     plt.xlabel("Feature Importance")
#     plt.title("Random Forest Feature Importance")
#     plt.tight_layout()

#     # Annotate bars with importance values
#     for bar in bars:
#         width = bar.get_width()
#         plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', va='center')

#     st.pyplot()

#     # Generate GenAI-style Summary
#     total_rows = len(df)
#     total_issues = df['Sustained_Issue'].sum()

#     summary = f"""
#     📊 **GenAI Summary: Telemetry-Based Preventive Maintenance Analysis**

#     Observation:

#     A total of {total_rows} telemetry time points were analyzed.
#     The system detected {total_issues} potential issue(s) where:
#         - TC1 was decreasing,
#         - TC10 was increasing,
#         - PUC State remained in condition 1.
#     A Random Forest Classifier trained on trend features achieved an accuracy of **{accuracy:.2f}%** on the test dataset.
#     Feature importance indicates that **TC1_trend** and **TC10_trend** are strong indicators of issue detection.

#     ✅ Recommended Action: 
#     For identified windows, initiate preventive checks on sensors and control logic that affect TC1 and TC10 behavior during PUC state 1.
#     """
#     st.markdown(f"""
#     <div class='output-box'>
#         <h4>🔍 Recommendation</h4>
#         <ul>
#             <li>{summary}</li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)

# else:
#     st.info("Please upload a text file to get started.")





import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# CSS Styling for Red and White Theme
st.markdown("""
    <style>
        /* General Styling */
        body {
            background-color: #ffffff;  /* White background */
            color: #000000;            /* Black text for contrast */
        }

        /* Header Styling */
        .header {
            text-align: center;
            margin-top: 20px;
        }

        .title {
            font-size: 28px;
            font-weight: bold;
            color: #FF0000;  /* Thermo Fisher Red */
            align-items: center;
        }

        /* Input/Output Boxes */
        .box {
            border: 3px solid #e60012;
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 30px;
            background-color: #ffffff;
            box-shadow: 0 0 10px rgba(0,0,0,0.05);
        }

        .output-box {
            border: 2px solid #e60012;
            border-radius: 15px;
            padding: 15px;
            margin-top: 20px;
            background-color: #ffffff;
        }

        .output-box ul {
            list-style-type: disc;
            padding-left: 20px;
            text-align: left;
            color: #000000;
        }

        /* Text and Input Styling */
        .stTextArea textarea, .stTextInput input {
            color: #000000;
            background-color: #ffffff;
            border: 1px solid #e60012;
            border-radius: 5px;
        }

        /* Button Styling */
        .stButton > button {
            background-color: #e60012;  /* Red */
            color: #ffffff;             /* White text */
            width: 200px;
            height: 40px;
            border: 2px solid #ffffff;
            border-radius: 25px;
            cursor: pointer;
        }

        .stButton > button:hover {
            background-color: #ffffff;
            color: #e60012;
            border: 2px solid #e60012;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='title'><center>Telemetry Analysis</h1>", unsafe_allow_html=True)

# File Uploader for Text File
uploaded_file = st.file_uploader("📂 Upload your Telemetry Data file", type=["puc"])

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")
    
    # Read the uploaded text file
    raw_data = uploaded_file.read().decode("utf-8")

    # Display file details
    st.write("**Filename:**", uploaded_file.name)
    st.write("**File size (bytes):**", uploaded_file.size)

    # --- Data Processing ---
    lines = raw_data.split('\n')
    columns = [
        'Date/Time', 'RTD', 'TC1', 'TC2', 'TC3', 'TC4', 'TC6', 'TC7', 'TC9', 'TC10',
        'Setpoint', 'Line Input', 'PUC State', 'User Offset',
        'Warm Warning Setpoint', 'Cold Warning Setpoint', 'Stage 1 RPM', 'Stage 2 RPM',
        'Total Valve Steps', 'Condenser Fan RPM', 'Algorithm Flags', 
        'Algorithm State', 'BUS RTD', 'Valve Steps', 'S1 Pressure',
        'TC8', 'Superheat', 'S2 Temperature', 'TSat'
    ]
    
    # Filter and clean data, removing unwanted lines
    data_lines = [line for line in lines if line and not line.startswith('PUC_VER')]
    
    # Parse the lines to create a DataFrame
    data_str = "\n".join(data_lines)
    df = pd.read_csv(StringIO(data_str), header=None)
    df.columns = columns[:df.shape[1]]

    # Convert Date/Time column to datetime
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')

    # Dropping columns with zero mean values
    zero_mean_cols = [
        col for col in df.columns 
        if col != 'Date/Time' and pd.to_numeric(df[col], errors='coerce').mean() == 0]
    df = df.drop(columns=zero_mean_cols)

    # --- Compute Trend Using Linear Regression ---
    def compute_slope(series):
        y = series.values
        x = np.arange(len(y)).reshape(-1, 1)
        if len(y) < 2 or np.all(np.isnan(y)):
            return np.nan
        model = LinearRegression().fit(x, y)
        return model.coef_[0]  # slope

    # Ensure 'TC1' and 'TC10' are numeric before applying rolling
    df['TC1'] = pd.to_numeric(df['TC1'], errors='coerce')
    df['TC10'] = pd.to_numeric(df['TC10'], errors='coerce')

    # Compute Trend Using Linear Regression
    df['TC1_trend'] = df['TC1'].rolling(60).apply(compute_slope, raw=False)
    df['TC10_trend'] = df['TC10'].rolling(60).apply(compute_slope, raw=False)
    df['PUC_state_mean'] = df['PUC State'].rolling(60).mean()

    # Flag sustained trend condition
    df['Trend_Flag'] = (df['TC1_trend'] < 0) & (df['TC10_trend'] > 0) & (df['PUC_state_mean'] == 1)

    # Identify sustained sequences (3+ consecutive flags)
    def flag_sustained(df, col='Trend_Flag', min_consecutive=3):
        sustained_flags = [False] * len(df)
        count = 0
        for i, val in enumerate(df[col]):
            if val:
                count += 1
                if count >= min_consecutive:
                    for j in range(i - count + 1, i + 1):
                        sustained_flags[j] = True
            else:
                count = 0
        return sustained_flags

    df['Sustained_Issue'] = flag_sustained(df)

    # Create Issue_Detected column based on sustained issues
    df['Issue_Detected'] = df['Sustained_Issue'].astype(int)  # Convert Boolean to Integer

    # Train Model
    features = ['TC1_trend', 'TC10_trend', 'PUC_state_mean']
    df_model = df[features + ['Issue_Detected']].dropna()
    
    X = df_model[features]
    y = df_model['Issue_Detected']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred_test = clf.predict(X_test)

    accuracy = clf.score(X_test, y_test) * 100
    st.write(f"Model Accuracy: {accuracy:.2f}%")

    # ------------------ STEP 4: Train Model ------------------
    features = ['TC1_trend', 'TC10_trend', 'PUC_state_mean']

    print(df.shape)
    # Drop rows with NaNs in either X or y
    df_model = df[features + ['Issue_Detected']].dropna()
    print(df_model.shape)

    X = df_model[features]
    y = df_model['Issue_Detected']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model training
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    # === Evaluation ===
    print("=== Train Classification Report ===")
    print(classification_report(y_train, y_pred_train))

    print("\n=== Test Classification Report ===")
    print(classification_report(y_test, y_pred_test))

    print("\n=== Confusion Matrix (Test) ===")
    st.write(confusion_matrix(y_test, y_pred_test))

    # ------------------ STEP 5: GenAI-style Summary Generation ------------------
    total_rows = len(df)
    print(total_rows)
    total_issues = df['Issue_Detected'].sum()
    print(total_issues)
    accuracy = clf.score(X_test, y_test) * 100
    print(accuracy,'%')

    # Plot feature importance
    

# Assuming clf and features are defined earlier (e.g., from a trained Random Forest model)
    importances = clf.feature_importances_
    sorted_indices = np.argsort(importances)
    sorted_features = [features[i] for i in sorted_indices]
    sorted_importances = importances[sorted_indices]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the data
    bars = ax.barh(sorted_features, sorted_importances, color='skyblue')
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest Feature Importance")
    plt.tight_layout()

    # Annotate bars with importance values
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', va='center')

    # Display the figure in Streamlit
    st.pyplot(fig)

    # Generate GenAI-style Summary
    total_rows = len(df)
    total_issues = df['Sustained_Issue'].sum()

    summary = f"""
    📊 **GenAI Summary: Telemetry-Based Preventive Maintenance Analysis**

    Observation:

    A total of {total_rows} telemetry time points were analyzed.
    The system detected {total_issues} potential issue(s) where:
        - TC1 was decreasing,
        - TC10 was increasing,
        - PUC State remained in condition 1.
    A Random Forest Classifier trained on trend features achieved an accuracy of **{accuracy:.2f}%** on the test dataset.
    Feature importance indicates that **TC1_trend** and **TC10_trend** are strong indicators of issue detection.

    ✅ Recommended Action: 
    For identified windows, initiate preventive checks on sensors and control logic that affect TC1 and TC10 behavior during PUC state 1.
    """
    st.markdown(f"""
    <div class='output-box'>
        <h4>🔍 Recommendation</h4>
        <ul>
            <li>{summary}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Please upload a text file to get started.")

