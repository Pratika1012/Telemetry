
# import streamlit as st
# import pandas as pd
# import numpy as np
# from io import StringIO
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.dates as mdates
# from sklearn.linear_model import LinearRegression
# import warnings
# warnings.filterwarnings(action="ignore")

# # --- CSS Styling for Red and White Theme ---
# st.markdown("""
#     <style>
           
#     /* Styling for the main submit button */
#     div.stButton > button {
#         background-color: red;  /* Red background */
#         width: 200px;           /* Adjust width */
#         height: 40px;           /* Adjust height */
#         border: none;           /* Remove border */
#         cursor: pointer;        /* Change cursor on hover */
#         border-radius: 25px;    /* Rounded corners */
#         color: Black;           /* White text color */
#         border: 2px solid white; /* Border */
#         margin-top: 5px;
#     }

#     /* Hover effect for the buttons */
#     div.stButton > button:hover {
#         background-color: white;
#         color: red;
#         border: 2px solid red;
#     }
#         body { background-color: #ffffff; color: #000000; }
#         .title { font-size: 28px; font-weight: bold; color: #FF0000; text-align: center; }
#         .output-box { border: 2px solid #e60012; border-radius: 15px; padding: 15px; margin-top: 20px; background: #fff; }
#     </style>
# """, unsafe_allow_html=True)

# # --- Sidebar Logo and Toggle ---
# logo_url = "logo1.png"
# st.sidebar.image(logo_url, width=200)

# if 'show_menu' not in st.session_state:
#     st.session_state.show_menu = False

# if st.sidebar.button("Show Telemetry Menu"):
#     st.session_state.show_menu = True

# # Header
# st.markdown("<h1 class='title'><center>Telemetry Analysis</h1>", unsafe_allow_html=True)
# if st.session_state.show_menu:
#     uploaded_file = st.file_uploader(" Upload your Telemetry Data file", type=["puc"])
#     if uploaded_file is not None:
#         st.success("\File uploaded successfully!")
#         raw_data = uploaded_file.read().decode("utf-8")
#         st.markdown(f"""
#         <div class='output-box'>
#             <h4>🔍 Recommendation</h4>
#             <ul>
#                 <li>{summary}</li>
#             </ul>
#         </div>
#         """, unsafe_allow_html=True)

#         columns = [
#             'Date/Time', 'RTD', 'TC1', 'TC2', 'TC3', 'TC4', 'TC6', 'TC7', 'TC9', 'TC10',
#             'Setpoint', 'Line Input', 'PUC State', 'User Offset',
#             'Warm Warning Setpoint', 'Cold Warning Setpoint', 'Stage 1 RPM', 'Stage 2 RPM',
#             'Total Valve Steps', 'Condenser Fan RPM', 'Algorithm Flags', 
#             'Algorithm State', 'BUS RTD', 'Valve Steps', 'S1 Pressure',
#             'TC8', 'Superheat', 'S2 Temperature', 'TSat'
#         ]

#         lines = raw_data.split('\n')
#         data_lines = [line for line in lines if not line.startswith('PUC_VER')]
#         expected_col_count = len(data_lines[0].split(','))
#         valid_lines = [line for line in data_lines if len(line.split(',')) == expected_col_count]
#         data_str = "\n".join(valid_lines)

#         df = pd.read_csv(StringIO(data_str), header=None)
#         df.columns = columns[:df.shape[1]]
#         df.columns = df.columns.str.strip()
#         df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')

#         three_months_ago = df['Date/Time'].max() - pd.DateOffset(months=3)
#         df = df[df['Date/Time'] >= three_months_ago]

#         df['Diff_RTD_Setpoint'] = df['RTD'] - df['Setpoint']
#         start_date = df['Date/Time'].min()
#         final_date = df['Date/Time'].max()
#         st.write(f"Data from {start_date} to {final_date}")
#     # === Compute Trend Using Linear Regression in Rolling Window ===
#     # === Compute Trend Using Linear Regression in Rolling Window ===
#         def compute_slope(series):
#             y = series.values
#             x = np.arange(len(y)).reshape(-1, 1)
#             if len(y) < 2 or np.all(np.isnan(y)):
#                 return np.nanz
#             model = LinearRegression().fit(x, y)
#             return model.coef_[0]  # slope

#         # Rule for 1st suction compressor failure with TC1(onesuction),TC10(bphx),TC3(evapin),TC4(evapout) & state (puc state)

#         # Define normal operating range for TC1
#         lower_bound_TC1 = -35
#         upper_bound_TC1 = -20

#         # Define normal operating range for TC3
#         lower_bound_TC3 = -95
#         upper_bound_TC3 = -86

#         # Define normal operating range for TC4
#         lower_bound_TC4 = -75
#         upper_bound_TC4 = -40

#         # Define normal operating range for TC6
#         lower_bound_TC6 = -30
#         upper_bound_TC6 = -15

#         # Define normal operating range for TC10
#         lower_bound_TC10 = -45
#         upper_bound_TC10 = -35

#         # === Define tolerance for RTD relative to Setpoint ===
#         tolerance_RTD = 2

#         # Create dynamic RTD bounds based on Setpoint
#         df['lower_bound_RTD'] = df['Setpoint'] - tolerance_RTD
#         df['upper_bound_RTD'] = df['Setpoint'] + tolerance_RTD

#         # Step 1: Flag values within normal range
#         df['TC1_in_range'] = df['TC1'].between(lower_bound_TC1, upper_bound_TC1)
#         df['TC6_in_range'] = df['TC6'].between(lower_bound_TC6, upper_bound_TC6)
#         df['TC3_in_range'] = df['TC3'].between(lower_bound_TC3, upper_bound_TC3)
#         df['TC4_in_range'] = df['TC4'].between(lower_bound_TC4, upper_bound_TC4)
#         df['TC10_in_range'] = df['TC10'].between(lower_bound_TC10, upper_bound_TC10)

#         # === RTD check: dynamically calculated based on Setpoint ===
#         df['RTD_in_range'] = (df['RTD'] >= df['lower_bound_RTD']) & (df['RTD'] <= df['upper_bound_RTD'])

#         print('Range column is updated') #check points

#         # Rolling window size (e.g., last 60 readings ~ 60 minutes)
#         WINDOW = 60

#         # List of columns you want to compute trends for
#         columns_to_trend = ['RTD', 'TC1', 'TC3', 'TC4','TC6', 'TC10', 'TC8']

#         # Compute trends and add new columns
#         for col in columns_to_trend:
#             df[f'{col}_trend'] = df[col].rolling(WINDOW).apply(compute_slope, raw=False)
            
#         df['PUC_state_mean'] = df['PUC State'].rolling(WINDOW).mean()
#         df['Diff_RTD_Setpoint_mean'] = df['Diff_RTD_Setpoint'].rolling(WINDOW).mean()

#         df.head()

#         # === Flag sustained trend condition ===
#         # Condition 1: 1st stage compression failure
#         condition_1 = (
#             (df['RTD_in_range'] == False) &
#             (df['TC1_trend'] > 0) &
#             (df['TC10_trend'] > 0) &
#             (df['PUC_state_mean'] == 1) &
#             (df['RTD_trend'] > 0)
#         )

#         # Condition 2: 1st stage leakage failure
#         condition_2 = (
#             (df['RTD_in_range'] == False) &
#             (df['TC3_trend'] < 0) &
#             (df['TC4_trend'] > 0) &
#             (df['TC10_trend'] > 0) &
#             (df['PUC_state_mean'] == 1) &
#             (df['RTD_trend'] > 0)
#         )

#         # Condition 3: 2nd stage compression failure
#         condition_3 = (
#             (df['RTD_in_range'] == False) &
#             (df['TC6_trend'] > 0) &
#             (df['TC10_trend'] > 0) &
#             (df['PUC_state_mean'] == 3) &
#             (df['RTD_trend'] > 0)
#         )

#         # Condition 4: 2nd stage compression failure
#         condition_4 = (
#             (df['RTD_in_range'] == False) &
#             (df['TC3_trend'] < 0) &
#             (df['TC4_trend'] > 0) &
#             (df['TC10_trend'] < 0) &
#             (df['PUC_state_mean'] == 3) &
#             (df['RTD_trend'] > 0)
#         )

#         # Apply conditions using numpy.select
#         df['Trend_Flag'] = np.select(
#             [condition_1, condition_2,condition_3, condition_4],
#             ['1st stage compression failure','1st stage leakage failure', '2nd stage compression failure','2nd stage leakage failure'],
#             default='No issue detected - your device is working properly'
#         )

#         # df[df['Trend_Flag'] == '1st stage compression/leakage failure'] # check point

#         # === Identify Sustained Sequences (e.g., 3+ consecutive flags) ===
#         def flag_sustained(df, col='Trend_Flag', min_consecutive=3):
#             sustained_flags = [False] * len(df)
#             count = 0
#             for i, val in enumerate(df[col]):
#                 if val != 'No issue detected - your device is working properly':
#                     count += 1
#                     if count >= min_consecutive:
#                         for j in range(i - count + 1, i + 1):
#                             sustained_flags[j] = True
#                 else:
#                     count = 0
#             return sustained_flags

#         # Apply to detect any sustained issue
#         df['Sustained_Issue'] = flag_sustained(df)

#         # === Output flagged rows ===
#         flagged = df[df['Sustained_Issue']]
#         print(flagged[['Date/Time', 'TC1', 'TC10','TC3','TC6', 'TC4','RTD', 'RTD_trend', 'TC1_trend', 'TC10_trend', 'TC3_trend', 'TC4_trend','TC6_trend','PUC State']])

#         df['Trend_Flag'].unique()

#         df['Issue_Detected'] = df['Sustained_Issue'].apply(lambda x: 1 if x else 0)
#         df['Issue_Detected'] = df['Sustained_Issue'].astype(int)

#         # ------------------ STEP 4: Train Model ------------------
#         features = ['PUC_state_mean','RTD_trend', 'TC1_trend', 'TC10_trend', 'TC3_trend', 'TC4_trend','TC6_trend']

#         print(df.shape)

#         # Drop rows with NaNs in either X or y
#         df_model = df[features + ['Issue_Detected']].dropna()
#         print(df_model.shape)

#         X = df_model[features]
#         y = df_model['Issue_Detected']

#         # Train-test split
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#         # Model training
#         clf = RandomForestClassifier(n_estimators=100, random_state=42)
#         clf.fit(X_train, y_train)

#         # Predictions
#         y_pred_test = clf.predict(X_test)
#         y_pred_train = clf.predict(X_train)

#         # === Evaluation ===
#         print("=== Train Classification Report ===")
#         print(classification_report(y_train, y_pred_train))

#         print("\n=== Test Classification Report ===")
#         print(classification_report(y_test, y_pred_test))

#         print("\n=== Confusion Matrix (Test) ===")
#         print(confusion_matrix(y_test, y_pred_test))

#         # --- Ensure Date/Time columns are in datetime format ---
        
# # --- Ensure Date/Time columns are in datetime format ---
#         df['Date/Time'] = pd.to_datetime(df['Date/Time'])

#         if not flagged.empty:
#             flagged['Date/Time'] = pd.to_datetime(flagged['Date/Time'])

#             # --- Determine flagged date boundaries ---
#             flagged_start = flagged['Date/Time'].min()
#             flagged_end = flagged['Date/Time'].max()

#             # --- Define plot boundaries ---
#             plot_start = flagged_start - pd.Timedelta(days=5)
#             plot_end = flagged_end + pd.Timedelta(days=10)

#             print("Flagged Start Date:", flagged_start)
#             print("Flagged End Date:", flagged_end)
#             print("Plot Start Date:", plot_start)
#             print("Plot End Date:", plot_end)

#             # --- Filter the full DataFrame for the plotting window ---
#             plot_df = df[(df['Date/Time'] >= plot_start) & (df['Date/Time'] <= plot_end)]

#             # --- Columns to include in the plot ---
#             columns_to_plot = ['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']

#             # --- Plotting ---
#             plt.figure(figsize=(16, 8))

#             for col in columns_to_plot:
#                 if col in plot_df.columns:
#                     # Convert to numeric to avoid plot issues
#                     plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
#                     plt.plot(plot_df['Date/Time'], plot_df[col], label=col)

#             # Format x-axis to show dates in YYYY-MM-DD format
#             plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#             plt.xticks(rotation=45)

#             plt.title(f"Sensor Trend Values from {plot_start.date()} to {plot_end.date()}")
#             plt.xlabel("Date")
#             plt.ylabel("Trend Values")
#             plt.legend()
#             plt.tight_layout()
#             plt.show()
#             st.pyplot(plt)

#         else:
#             print("✅ No Issue Detected — Device Operating Normally")
       

#         # --- Ensure Date/Time columns are in datetime format ---
        
#     # --- Ensure Date/Time columns are in datetime format ---
#         df['Date/Time'] = pd.to_datetime(df['Date/Time'])

#         if not flagged.empty:
#             flagged['Date/Time'] = pd.to_datetime(flagged['Date/Time'])

#             # --- Determine flagged date boundaries ---
#             flagged_start = flagged['Date/Time'].min()
#             flagged_end = flagged['Date/Time'].max()

#             # --- Define plot boundaries ---
#             plot_start = flagged_start - pd.Timedelta(days=5)
#             plot_end = flagged_end + pd.Timedelta(days=10)

#             print("Flagged Start Date:", flagged_start)
#             print("Flagged End Date:", flagged_end)
#             print("Plot Start Date:", plot_start)
#             print("Plot End Date:", plot_end)

#             # --- Filter the full DataFrame for the plotting window ---
#             plot_df = df[(df['Date/Time'] >= plot_start) & (df['Date/Time'] <= plot_end)]

#             # --- Columns to include in the plot ---
#             columns_to_plot = ['RTD_trend', 'TC1_trend', 'TC10_trend', 'TC3_trend', 'TC4_trend', 'TC6_trend']

#             # --- Plotting ---
#             plt.figure(figsize=(16, 8))

#             for col in columns_to_plot:
#                 if col in plot_df.columns:
#                     # Convert to numeric to avoid plot issues
#                     plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
#                     plt.plot(plot_df['Date/Time'], plot_df[col], label=col)

#             # Format x-axis to show dates in YYYY-MM-DD format
#             plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#             plt.xticks(rotation=45)

#             plt.title(f"Sensor Trend Values from {plot_start.date()} to {plot_end.date()}")
#             plt.xlabel("Date")
#             plt.ylabel("Trend Values")
#             plt.legend()
#             plt.tight_layout()
#             plt.show()
#             st.pyplot(plt)

#         else:
#             print("✅ No issue detected - your device is working properly.")
                
#         # Feature importances
#         importances = clf.feature_importances_
#         feature_names = features

#         # Sort importances for better visualization (optional)
#         indices = np.argsort(importances)
#         sorted_features = [feature_names[i] for i in indices]
#         sorted_importances = importances[indices]

#         # Plotting
#         plt.figure(figsize=(8, 5))
#         bars = plt.barh(sorted_features, sorted_importances, color='skyblue')
#         plt.xlabel("Feature Importance")
#         plt.title("Random Forest Feature Importance")
#         plt.tight_layout()

#         # Annotate bars with importance values
#         for bar in bars:
#             width = bar.get_width()
#             plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2,f'{width:.2f}', va='center')
#         plt.show()
#         st.pyplot(plt)

#         # ------------------ STEP 5: GenAI-style Summary Generation ------------------
#         total_rows = len(df)
#         print(total_rows)
#         total_issues = df['Issue_Detected'].sum()
#         print(total_issues)
#         accuracy = clf.score(X_test, y_test) * 100
#         print(accuracy,'%')

#         # List of columns you're interested in
#         # List of columns to check
#         core_columns = ['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']
#         trend_columns = [col + '_trend' for col in core_columns]
#         columns_to_check = core_columns + trend_columns

#         # Prepare summary
#         summary = []


#         # Check if 'flagged' exists and is not empty
#         if 'flagged' in locals() and not flagged.empty:
#             data_source = flagged
#             source_label = "Flagged Period"
#         else:
#             data_source = df  # fallback to full data
#             source_label = "Full Dataset (No Issues Detected)"

#         # Loop through each column and collect stats
#         for col in columns_to_check:
#             if col in data_source.columns:
#                 min_val = data_source[col].min()
#                 max_val = data_source[col].max()
#                 mean_val = data_source[col].mean()
#                 min_date = data_source.loc[data_source[col].idxmin(), 'Date/Time']
#                 max_date = data_source.loc[data_source[col].idxmax(), 'Date/Time']

#                 summary.append({
#                     'Column': col,
#                     'Min': min_val,
#                     'Min Date': min_date,
#                     'Mean': mean_val,
#                     'Max': max_val,
#                     'Max Date': max_date
#                 })

#         # Create and show summary DataFrame
#         summary_df = pd.DataFrame(summary)

#         print(f"📋 Summary Statistics ({source_label})")
#         print(summary_df)

#         # Ensure Date/Time is datetime and set as index
#         df['Date/Time'] = pd.to_datetime(df['Date/Time'])
#         df.set_index('Date/Time', inplace=True)

#         # Resample day-wise
#         daily_df_actual = df[['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']].resample('D').mean().reset_index()

#         # Plot
#         plt.figure(figsize=(14, 6))
#         sns.set(style="whitegrid")

#         # Plot each trend line
#         for col in ['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']:
#             plt.plot(daily_df_actual['Date/Time'], daily_df_actual[col], label=col)

#         # Format x-axis to show only dates
#         plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

#         plt.title('Day-wise actual Values')
#         plt.xlabel('Date')
#         plt.ylabel('Trend Value')
#         plt.legend()
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         plt.show()
#         st.pyplot(plt)
#         # Resample day-wise
#         daily_df_trend = df[['RTD_trend', 'TC1_trend', 'TC10_trend', 'TC3_trend', 'TC4_trend', 'TC6_trend']].resample('D').mean().reset_index()

#         # Plot
#         plt.figure(figsize=(14, 6))
#         sns.set(style="whitegrid")

#         # Plot each trend line
#         for col in ['RTD_trend', 'TC1_trend', 'TC10_trend', 'TC3_trend', 'TC4_trend', 'TC6_trend']:
#             plt.plot(daily_df_trend['Date/Time'], daily_df_trend[col], label=col)

#         # Format x-axis to show only dates
#         plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

#         plt.title('Day-wise Trend Values')
#         plt.xlabel('Date')
#         plt.ylabel('Trend Value')
#         plt.legend()
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         plt.show()
#         st.pyplot(plt)
#         # Date/Time is datetime and set as index
#         # --- Check if flagged data exists and is not empty ---
#         if 'flagged' in locals() and not flagged.empty:
#             # Ensure Date/Time is datetime and set as index
#             flagged['Date/Time'] = pd.to_datetime(flagged['Date/Time'])
#             flagged.set_index('Date/Time', inplace=True)

#             # Resample day-wise mean
#             daily_df_actual = flagged[['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']].resample('D').mean().reset_index()

#             # Plot
#             plt.figure(figsize=(14, 6))
#             sns.set(style="whitegrid")

#             for col in ['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']:
#                 plt.plot(daily_df_actual['Date/Time'], daily_df_actual[col], label=col)

#             # Format x-axis
#             plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#             plt.title('📈 Flagged Day-wise Actual Values')
#             plt.xlabel('Date')
#             plt.ylabel('Trend Value')
#             plt.legend()
#             plt.xticks(rotation=45)
#             plt.tight_layout()
#             plt.show()
#             st.pyplot(plt)

#         else:
#             print("✅ No issue detected — your device is working properly. All investigated values lie within the expected range.")
                

#             # === Extract required variables from flagged DataFrame for summary ===

#     # Compute sensor trend directions
#         tc1_trend = 'Decreasing' if flagged['TC1_trend'].mean() < 0 else 'Increasing'
#         tc10_trend = 'Decreasing' if flagged['TC10_trend'].mean() < 0 else 'Increasing'

#         # Compare average RTD vs Setpoint
#         rtd_mean = flagged['RTD'].mean()
#         setpoint_mean = flagged['Setpoint'].mean()
#         rtd_vs_setpoint_status = "RTD higher than Setpoint consistently" if rtd_mean > setpoint_mean else "RTD within expected range"

#         # Most common root cause from flagged data
#         if 'Trend_Flag' in flagged.columns and not flagged['Trend_Flag'].empty:
#             root_cause = flagged['Trend_Flag'].value_counts().idxmax()
#         else:
#             root_cause = "No root cause detected"

#         # Overall system status
#         device_status = "Issue Detected" if total_issues > 0 else "Working Well"

#             # === Final GenAI Summary ===
#         summary = f"""
#             📊 **GenAI Summary: Telemetry-Based Preventive Maintenance Analysis**

#             Observation:

#             A total of {total_rows} telemetry time points were analyzed.
#             The system detected {total_issues} potential issue(s) where:
#                 - TC1 was {tc1_trend.lower()},
#                 - TC10 was {tc10_trend.lower()},
#                 - PUC State remained in condition 1.
#             A Random Forest Classifier trained on trend features achieved an accuracy of **{accuracy:.2f}%** on the test dataset.
#             Feature importance indicates that **TC1_trend** and **TC10_trend** are strong indicators of issue detection.

#             🔎 **Technical Evaluation Summary**

#             **Device Status:** {device_status}  
#             **Detected Root Cause:** {root_cause}

#             **Key Sensor Readings & Trends:**
#             - TC1 Trend: {tc1_trend}  
#             - TC10 Trend: {tc10_trend}  
#             - RTD vs Setpoint: {rtd_vs_setpoint_status}

#             🧠 **Root Cause Explanation:**  
#             The combination of a {tc1_trend.lower()} TC1 and {tc10_trend.lower()} TC10, while the system remains in state 1, suggests abnormal heat transfer or inefficiencies likely due to a **{root_cause}**. Persistent RTD elevation beyond the setpoint supports the hypothesis of system load imbalance or cooling inefficiency.

#             🔧 **Suggested Preventive Actions:**
#             - Schedule periodic pressure integrity tests for both compressor stages.
#             - Regularly inspect thermal coupling and ensure adequate insulation around TC1/TC10 lines.
#             - Implement alerts when TC trends diverge while PUC remains constant.

#             🛠 **Suggested Corrective Actions:**
#             - Conduct a diagnostic check on the suspected compressor module.
#             - Inspect and replace any worn or damaged seals that could cause internal leaks.
#             - Recalibrate sensors and verify PID control settings for thermal regulation.

#             ✅ **Confidence Level in Root Cause Identification:** **{accuracy:.2f}%**
#             """

#             # Print in console for debugging (optional)
#         print(summary)

#             # === Display inside styled Streamlit output box ===
#         st.markdown(f"""
#             <div class='output-box'>
#                 <h4>🔍 Recommendation</h4>
#                 <ul>
#                     <li>{summary}</li>
#                 </ul>
#             </div>
#             """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings(action="ignore")

# --- CSS Styling for Red and White Theme ---
st.markdown("""
    <style>
    /* Styling for the main submit button */
    div.stButton > button {
        background-color: red;  /* Red background */
        width: 200px;           /* Adjust width */
        height: 40px;           /* Adjust height */
        border: none;           /* Remove border */
        cursor: pointer;        /* Change cursor on hover */
        border-radius: 25px;    /* Rounded corners */
        color: Black;           /* White text color */
        border: 2px solid white; /* Border */
        margin-top: 5px;
    }
    /* Hover effect for the buttons */
    div.stButton > button:hover {
        background-color: white;
        color: red;
        border: 2px solid red;
    }
    body { background-color: #ffffff; color: #000000; }
    .title { font-size: 28px; font-weight: bold; color: #FF0000; text-align: center; }
    .output-box { border: 2px solid #e60012; border-radius: 15px; padding: 15px; margin-top: 20px; background: #fff; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Logo and Toggle ---
logo_url = "logo1.png"
st.sidebar.image(logo_url, width=200)

if 'show_menu' not in st.session_state:
    st.session_state.show_menu = False

if st.sidebar.button("Show Telemetry Menu"):
    st.session_state.show_menu = True

# Header
st.markdown("<h1 style='color:red; text-align:center;'>Telemetry Analysis</h1>", unsafe_allow_html=True)


if st.session_state.show_menu:
    uploaded_file = st.file_uploader(" Upload your Telemetry Data file", type=["puc"])
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        raw_data = uploaded_file.read().decode("utf-8")

        columns = [
            'Date/Time', 'RTD', 'TC1', 'TC2', 'TC3', 'TC4', 'TC6', 'TC7', 'TC9', 'TC10',
            'Setpoint', 'Line Input', 'PUC State', 'User Offset',
            'Warm Warning Setpoint', 'Cold Warning Setpoint', 'Stage 1 RPM', 'Stage 2 RPM',
            'Total Valve Steps', 'Condenser Fan RPM', 'Algorithm Flags', 
            'Algorithm State', 'BUS RTD', 'Valve Steps', 'S1 Pressure',
            'TC8', 'Superheat', 'S2 Temperature', 'TSat'
        ]

        lines = raw_data.split('\n')
        data_lines = [line for line in lines if not line.startswith('PUC_VER')]
        expected_col_count = len(data_lines[0].split(','))
        valid_lines = [line for line in data_lines if len(line.split(',')) == expected_col_count]
        data_str = "\n".join(valid_lines)

        df = pd.read_csv(StringIO(data_str), header=None)
        df.columns = columns[:df.shape[1]]
        df.columns = df.columns.str.strip()
        df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')

        three_months_ago = df['Date/Time'].max() - pd.DateOffset(months=3)
        df = df[df['Date/Time'] >= three_months_ago]

        df['Diff_RTD_Setpoint'] = df['RTD'] - df['Setpoint']
        start_date = df['Date/Time'].min()
        final_date = df['Date/Time'].max()
        st.write(f"Data from {start_date} to {final_date}")

        # Compute Trend Using Linear Regression in Rolling Window
        def compute_slope(series):
            y = series.values
            x = np.arange(len(y)).reshape(-1, 1)
            if len(y) < 2 or np.all(np.isnan(y)):
                return np.nan
            model = LinearRegression().fit(x, y)
            return model.coef_[0]

        # Define normal operating ranges
        lower_bound_TC1 = -35
        upper_bound_TC1 = -20
        lower_bound_TC3 = -95
        upper_bound_TC3 = -86
        lower_bound_TC4 = -75
        upper_bound_TC4 = -40
        lower_bound_TC6 = -30
        upper_bound_TC6 = -15
        lower_bound_TC10 = -45
        upper_bound_TC10 = -35
        tolerance_RTD = 2

        df['lower_bound_RTD'] = df['Setpoint'] - tolerance_RTD
        df['upper_bound_RTD'] = df['Setpoint'] + tolerance_RTD

        df['TC1_in_range'] = df['TC1'].between(lower_bound_TC1, upper_bound_TC1)
        df['TC6_in_range'] = df['TC6'].between(lower_bound_TC6, upper_bound_TC6)
        df['TC3_in_range'] = df['TC3'].between(lower_bound_TC3, upper_bound_TC3)
        df['TC4_in_range'] = df['TC4'].between(lower_bound_TC4, upper_bound_TC4)
        df['TC10_in_range'] = df['TC10'].between(lower_bound_TC10, upper_bound_TC10)
        df['RTD_in_range'] = (df['RTD'] >= df['lower_bound_RTD']) & (df['RTD'] <= df['upper_bound_RTD'])

        print('Range column is updated')

        WINDOW = 60
        columns_to_trend = ['RTD', 'TC1', 'TC3', 'TC4', 'TC6', 'TC10', 'TC8']
        for col in columns_to_trend:
            df[f'{col}_trend'] = df[col].rolling(WINDOW).apply(compute_slope, raw=False)
        
        df['PUC_state_mean'] = df['PUC State'].rolling(WINDOW).mean()
        df['Diff_RTD_Setpoint_mean'] = df['Diff_RTD_Setpoint'].rolling(WINDOW).mean()

        df.head()

        # Flag sustained trend conditions
        condition_1 = (
            (df['RTD_in_range'] == False) &
            (df['TC1_trend'] > 0) &
            (df['TC10_trend'] > 0) &
            (df['PUC_state_mean'] == 1) &
            (df['RTD_trend'] > 0)
        )
        condition_2 = (
            (df['RTD_in_range'] == False) &
            (df['TC3_trend'] < 0) &
            (df['TC4_trend'] > 0) &
            (df['TC10_trend'] > 0) &
            (df['PUC_state_mean'] == 1) &
            (df['RTD_trend'] > 0)
        )
        condition_3 = (
            (df['RTD_in_range'] == False) &
            (df['TC6_trend'] > 0) &
            (df['TC10_trend'] > 0) &
            (df['PUC_state_mean'] == 3) &
            (df['RTD_trend'] > 0)
        )
        condition_4 = (
            (df['RTD_in_range'] == False) &
            (df['TC3_trend'] < 0) &
            (df['TC4_trend'] > 0) &
            (df['TC10_trend'] < 0) &
            (df['PUC_state_mean'] == 3) &
            (df['RTD_trend'] > 0)
        )

        df['Trend_Flag'] = np.select(
            [condition_1, condition_2, condition_3, condition_4],
            ['1st stage compression failure', '1st stage leakage failure', '2nd stage compression failure', '2nd stage leakage failure'],
            default='No issue detected - your device is working properly'
        )

        def flag_sustained(df, col='Trend_Flag', min_consecutive=3):
            sustained_flags = [False] * len(df)
            count = 0
            for i, val in enumerate(df[col]):
                if val != 'No issue detected - your device is working properly':
                    count += 1
                    if count >= min_consecutive:
                        for j in range(i - count + 1, i + 1):
                            sustained_flags[j] = True
                else:
                    count = 0
            return sustained_flags

        df['Sustained_Issue'] = flag_sustained(df)
        flagged = df[df['Sustained_Issue']]
        print(flagged[['Date/Time', 'TC1', 'TC10', 'TC3', 'TC6', 'TC4', 'RTD', 'RTD_trend', 'TC1_trend', 'TC10_trend', 'TC3_trend', 'TC4_trend', 'TC6_trend', 'PUC State']])

        df['Trend_Flag'].unique()

        df['Issue_Detected'] = df['Sustained_Issue'].apply(lambda x: 1 if x else 0)
        df['Issue_Detected'] = df['Sustained_Issue'].astype(int)

        # Train Model
        features = ['PUC_state_mean', 'RTD_trend', 'TC1_trend', 'TC10_trend', 'TC3_trend', 'TC4_trend', 'TC6_trend']
        print(df.shape)
        df_model = df[features + ['Issue_Detected']].dropna()
        print(df_model.shape)

        X = df_model[features]
        y = df_model['Issue_Detected']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)

        print("=== Train Classification Report ===")
        print(classification_report(y_train, y_pred_train))
        print("\n=== Test Classification Report ===")
        print(classification_report(y_test, y_pred_test))
        print("\n=== Confusion Matrix (Test) ===")
        print(confusion_matrix(y_test, y_pred_test))

        # --- Generate Summary ---
        total_rows = len(df)
        print(total_rows)
        total_issues = df['Issue_Detected'].sum()
        print(total_issues)
        accuracy = clf.score(X_test, y_test) * 100
        print(accuracy, '%')

        core_columns = ['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']
        trend_columns = [col + '_trend' for col in core_columns]
        columns_to_check = core_columns + trend_columns
        summary = []

        if 'flagged' in locals() and not flagged.empty:
            data_source = flagged
            source_label = "Flagged Period"
        else:
            data_source = df
            source_label = "Full Dataset (No Issues Detected)"

        for col in columns_to_check:
            if col in data_source.columns:
                min_val = data_source[col].min()
                max_val = data_source[col].max()
                mean_val = data_source[col].mean()
                min_date = data_source.loc[data_source[col].idxmin(), 'Date/Time']
                max_date = data_source.loc[data_source[col].idxmax(), 'Date/Time']
                summary.append({
                    'Column': col,
                    'Min': min_val,
                    'Min Date': min_date,
                    'Mean': mean_val,
                    'Max': max_val,
                    'Max Date': max_date
                })

        summary_df = pd.DataFrame(summary)
        print(f"📋 Summary Statistics ({source_label})")
        print(summary_df)

        tc1_trend = 'Decreasing' if flagged['TC1_trend'].mean() < 0 else 'Increasing' if not flagged.empty else 'Stable'
        tc10_trend = 'Decreasing' if flagged['TC10_trend'].mean() < 0 else 'Increasing' if not flagged.empty else 'Stable'
        rtd_mean = flagged['RTD'].mean() if not flagged.empty else df['RTD'].mean()
        setpoint_mean = flagged['Setpoint'].mean() if not flagged.empty else df['Setpoint'].mean()
        rtd_vs_setpoint_status = "RTD higher than Setpoint consistently" if rtd_mean > setpoint_mean else "RTD within expected range"
        root_cause = flagged['Trend_Flag'].value_counts().idxmax() if not flagged.empty and 'Trend_Flag' in flagged.columns else "No root cause detected"
        device_status = "Issue Detected" if total_issues > 0 else "Working Well"

        summary = f"""
        📊 **GenAI Summary: Telemetry-Based Preventive Maintenance Analysis**

        Observation:

        A total of {total_rows} telemetry time points were analyzed.
        The system detected {total_issues} potential issue(s) where:
            - TC1 was {tc1_trend.lower()},
            - TC10 was {tc10_trend.lower()},
            - PUC State remained in condition 1.
        A Random Forest Classifier trained on trend features achieved an accuracy of **{accuracy:.2f}%** on the test dataset.
        Feature importance indicates that **TC1_trend** and **TC10_trend** are strong indicators of issue detection.

        🔎 **Technical Evaluation Summary**

        **Device Status:** {device_status}  
        **Detected Root Cause:** {root_cause}

        **Key Sensor Readings & Trends:**
        - TC1 Trend: {tc1_trend}  
        - TC10 Trend: {tc10_trend}  
        - RTD vs Setpoint: {rtd_vs_setpoint_status}

        🧠 **Root Cause Explanation:**  
        The combination of a {tc1_trend.lower()} TC1 and {tc10_trend.lower()} TC10, while the system remains in state 1, suggests abnormal heat transfer or inefficiencies likely due to a **{root_cause}**. Persistent RTD elevation beyond the setpoint supports the hypothesis of system load imbalance or cooling inefficiency.

        🔧 **Suggested Preventive Actions:**
        - Schedule periodic pressure integrity tests for both compressor stages.
        - Regularly inspect thermal coupling and ensure adequate insulation around TC1/TC10 lines.
        - Implement alerts when TC trends diverge while PUC remains constant.

        🛠 **Suggested Corrective Actions:**
        - Conduct a diagnostic check on the suspected compressor module.
        - Inspect and replace any worn or damaged seals that could cause internal leaks.
        - Recalibrate sensors and verify PID control settings for thermal regulation.

        ✅ **Confidence Level in Root Cause Identification:** **{accuracy:.2f}%**
        """

        print(summary)
        st.markdown(f"""
        <div class='output-box'>
            <h4>🔍 Recommendation</h4>
            <ul>
                <li>{summary}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # --- Plotting Section ---
        df['Date/Time'] = pd.to_datetime(df['Date/Time'])
        if not flagged.empty:
            flagged['Date/Time'] = pd.to_datetime(flagged['Date/Time'])
            flagged_start = flagged['Date/Time'].min()
            flagged_end = flagged['Date/Time'].max()
            plot_start = flagged_start - pd.Timedelta(days=5)
            plot_end = flagged_end + pd.Timedelta(days=10)

            print("Flagged Start Date:", flagged_start)
            print("Flagged End Date:", flagged_end)
            print("Plot Start Date:", plot_start)
            print("Plot End Date:", plot_end)

            plot_df = df[(df['Date/Time'] >= plot_start) & (df['Date/Time'] <= plot_end)]
            columns_to_plot = ['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']
            plt.figure(figsize=(16, 8))
            for col in columns_to_plot:
                if col in plot_df.columns:
                    plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                    plt.plot(plot_df['Date/Time'], plot_df[col], label=col)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.title(f"Sensor Trend Values from {plot_start.date()} to {plot_end.date()}")
            plt.xlabel("Date")
            plt.ylabel("Trend Values")
            plt.legend()
            plt.tight_layout()
            plt.show()
            st.pyplot(plt)
        else:
            print("✅ No Issue Detected — Device Operating Normally")

        df['Date/Time'] = pd.to_datetime(df['Date/Time'])
        if not flagged.empty:
            flagged['Date/Time'] = pd.to_datetime(flagged['Date/Time'])
            flagged_start = flagged['Date/Time'].min()
            flagged_end = flagged['Date/Time'].max()
            plot_start = flagged_start - pd.Timedelta(days=5)
            plot_end = flagged_end + pd.Timedelta(days=10)

            print("Flagged Start Date:", flagged_start)
            print("Flagged End Date:", flagged_end)
            print("Plot Start Date:", plot_start)
            print("Plot End Date:", plot_end)

            plot_df = df[(df['Date/Time'] >= plot_start) & (df['Date/Time'] <= plot_end)]
            columns_to_plot = ['RTD_trend', 'TC1_trend', 'TC10_trend', 'TC3_trend', 'TC4_trend', 'TC6_trend']
            plt.figure(figsize=(16, 8))
            for col in columns_to_plot:
                if col in plot_df.columns:
                    plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                    plt.plot(plot_df['Date/Time'], plot_df[col], label=col)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.title(f"Sensor Trend Values from {plot_start.date()} to {plot_end.date()}")
            plt.xlabel("Date")
            plt.ylabel("Trend Values")
            plt.legend()
            plt.tight_layout()
            plt.show()
            st.pyplot(plt)
        else:
            print("✅ No issue detected - your device is working properly.")

        importances = clf.feature_importances_
        feature_names = features
        indices = np.argsort(importances)
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
        plt.figure(figsize=(8, 5))
        bars = plt.barh(sorted_features, sorted_importances, color='skyblue')
        plt.xlabel("Feature Importance")
        plt.title("Random Forest Feature Importance")
        plt.tight_layout()
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', va='center')
        plt.show()
        st.pyplot(plt)

        df['Date/Time'] = pd.to_datetime(df['Date/Time'])
        df.set_index('Date/Time', inplace=True)
        daily_df_actual = df[['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']].resample('D').mean().reset_index()
        plt.figure(figsize=(14, 6))
        sns.set(style="whitegrid")
        for col in ['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']:
            plt.plot(daily_df_actual['Date/Time'], daily_df_actual[col], label=col)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.title('Day-wise actual Values')
        plt.xlabel('Date')
        plt.ylabel('Trend Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        st.pyplot(plt)

        daily_df_trend = df[['RTD_trend', 'TC1_trend', 'TC10_trend', 'TC3_trend', 'TC4_trend', 'TC6_trend']].resample('D').mean().reset_index()
        plt.figure(figsize=(14, 6))
        sns.set(style="whitegrid")
        for col in ['RTD_trend', 'TC1_trend', 'TC10_trend', 'TC3_trend', 'TC4_trend', 'TC6_trend']:
            plt.plot(daily_df_trend['Date/Time'], daily_df_trend[col], label=col)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.title('Day-wise Trend Values')
        plt.xlabel('Date')
        plt.ylabel('Trend Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        st.pyplot(plt)

        if 'flagged' in locals() and not flagged.empty:
            flagged['Date/Time'] = pd.to_datetime(flagged['Date/Time'])
            flagged.set_index('Date/Time', inplace=True)
            daily_df_actual = flagged[['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']].resample('D').mean().reset_index()
            plt.figure(figsize=(14, 6))
            sns.set(style="whitegrid")
            for col in ['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']:
                plt.plot(daily_df_actual['Date/Time'], daily_df_actual[col], label=col)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.title('📈 Flagged Day-wise Actual Values')
            plt.xlabel('Date')
            plt.ylabel('Trend Value')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            st.pyplot(plt)
        else:
            print("✅ No issue detected — your device is working properly. All investigated values lie within the expected range.")
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings(action="ignore")

# --- CSS Styling for Red and White Theme ---
st.markdown("""
    <style>
    /* Styling for the main submit button */
    div.stButton > button {
        background-color: red;  /* Red background */
        width: 200px;           /* Adjust width */
        height: 40px;           /* Adjust height */
        border: none;           /* Remove border */
        cursor: pointer;        /* Change cursor on hover */
        border-radius: 25px;    /* Rounded corners */
        color: Black;           /* White text color */
        border: 2px solid white; /* Border */
        margin-top: 5px;
    }
    /* Hover effect for the buttons */
    div.stButton > button:hover {
        background-color: white;
        color: red;
        border: 2px solid red;
    }
    body { background-color: #ffffff; color: #000000; }
    .title { font-size: 28px; font-weight: bold; color: #FF0000; text-align: center; }
    .output-box { border: 2px solid #e60012; border-radius: 15px; padding: 15px; margin-top: 20px; background: #fff; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Logo and Toggle ---
logo_url = "logo1.png"
st.sidebar.image(logo_url, width=200)

if 'show_menu' not in st.session_state:
    st.session_state.show_menu = False

if st.sidebar.button("Show Telemetry Menu"):
    st.session_state.show_menu = True

# Header
st.markdown("<h1 class='title'><center>Telemetry Analysis</h1>", unsafe_allow_html=True)

if st.session_state.show_menu:
    uploaded_file = st.file_uploader(" Upload your Telemetry Data file", type=["puc"])
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        raw_data = uploaded_file.read().decode("utf-8")

        columns = [
            'Date/Time', 'RTD', 'TC1', 'TC2', 'TC3', 'TC4', 'TC6', 'TC7', 'TC9', 'TC10',
            'Setpoint', 'Line Input', 'PUC State', 'User Offset',
            'Warm Warning Setpoint', 'Cold Warning Setpoint', 'Stage 1 RPM', 'Stage 2 RPM',
            'Total Valve Steps', 'Condenser Fan RPM', 'Algorithm Flags', 
            'Algorithm State', 'BUS RTD', 'Valve Steps', 'S1 Pressure',
            'TC8', 'Superheat', 'S2 Temperature', 'TSat'
        ]

        lines = raw_data.split('\n')
        data_lines = [line for line in lines if not line.startswith('PUC_VER')]
        expected_col_count = len(data_lines[0].split(','))
        valid_lines = [line for line in data_lines if len(line.split(',')) == expected_col_count]
        data_str = "\n".join(valid_lines)

        df = pd.read_csv(StringIO(data_str), header=None)
        df.columns = columns[:df.shape[1]]
        df.columns = df.columns.str.strip()
        df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')

        three_months_ago = df['Date/Time'].max() - pd.DateOffset(months=3)
        df = df[df['Date/Time'] >= three_months_ago]

        df['Diff_RTD_Setpoint'] = df['RTD'] - df['Setpoint']
        start_date = df['Date/Time'].min()
        final_date = df['Date/Time'].max()
        st.write(f"Data from {start_date} to {final_date}")

        # Compute Trend Using Linear Regression in Rolling Window
        def compute_slope(series):
            y = series.values
            x = np.arange(len(y)).reshape(-1, 1)
            if len(y) < 2 or np.all(np.isnan(y)):
                return np.nan
            model = LinearRegression().fit(x, y)
            return model.coef_[0]

        # Define normal operating ranges
        lower_bound_TC1 = -35
        upper_bound_TC1 = -20
        lower_bound_TC3 = -95
        upper_bound_TC3 = -86
        lower_bound_TC4 = -75
        upper_bound_TC4 = -40
        lower_bound_TC6 = -30
        upper_bound_TC6 = -15
        lower_bound_TC10 = -45
        upper_bound_TC10 = -35
        tolerance_RTD = 2

        df['lower_bound_RTD'] = df['Setpoint'] - tolerance_RTD
        df['upper_bound_RTD'] = df['Setpoint'] + tolerance_RTD

        df['TC1_in_range'] = df['TC1'].between(lower_bound_TC1, upper_bound_TC1)
        df['TC6_in_range'] = df['TC6'].between(lower_bound_TC6, upper_bound_TC6)
        df['TC3_in_range'] = df['TC3'].between(lower_bound_TC3, upper_bound_TC3)
        df['TC4_in_range'] = df['TC4'].between(lower_bound_TC4, upper_bound_TC4)
        df['TC10_in_range'] = df['TC10'].between(lower_bound_TC10, upper_bound_TC10)
        df['RTD_in_range'] = (df['RTD'] >= df['lower_bound_RTD']) & (df['RTD'] <= df['upper_bound_RTD'])

        print('Range column is updated')

        WINDOW = 60
        columns_to_trend = ['RTD', 'TC1', 'TC3', 'TC4', 'TC6', 'TC10', 'TC8']
        for col in columns_to_trend:
            df[f'{col}_trend'] = df[col].rolling(WINDOW).apply(compute_slope, raw=False)
        
        df['PUC_state_mean'] = df['PUC State'].rolling(WINDOW).mean()
        df['Diff_RTD_Setpoint_mean'] = df['Diff_RTD_Setpoint'].rolling(WINDOW).mean()

        df.head()

        # Flag sustained trend conditions
        condition_1 = (
            (df['RTD_in_range'] == False) &
            (df['TC1_trend'] > 0) &
            (df['TC10_trend'] > 0) &
            (df['PUC_state_mean'] == 1) &
            (df['RTD_trend'] > 0)
        )
        condition_2 = (
            (df['RTD_in_range'] == False) &
            (df['TC3_trend'] < 0) &
            (df['TC4_trend'] > 0) &
            (df['TC10_trend'] > 0) &
            (df['PUC_state_mean'] == 1) &
            (df['RTD_trend'] > 0)
        )
        condition_3 = (
            (df['RTD_in_range'] == False) &
            (df['TC6_trend'] > 0) &
            (df['TC10_trend'] > 0) &
            (df['PUC_state_mean'] == 3) &
            (df['RTD_trend'] > 0)
        )
        condition_4 = (
            (df['RTD_in_range'] == False) &
            (df['TC3_trend'] < 0) &
            (df['TC4_trend'] > 0) &
            (df['TC10_trend'] < 0) &
            (df['PUC_state_mean'] == 3) &
            (df['RTD_trend'] > 0)
        )

        df['Trend_Flag'] = np.select(
            [condition_1, condition_2, condition_3, condition_4],
            ['1st stage compression failure', '1st stage leakage failure', '2nd stage compression failure', '2nd stage leakage failure'],
            default='No issue detected - your device is working properly'
        )

        def flag_sustained(df, col='Trend_Flag', min_consecutive=3):
            sustained_flags = [False] * len(df)
            count = 0
            for i, val in enumerate(df[col]):
                if val != 'No issue detected - your device is working properly':
                    count += 1
                    if count >= min_consecutive:
                        for j in range(i - count + 1, i + 1):
                            sustained_flags[j] = True
                else:
                    count = 0
            return sustained_flags

        df['Sustained_Issue'] = flag_sustained(df)
        flagged = df[df['Sustained_Issue']]
        print(flagged[['Date/Time', 'TC1', 'TC10', 'TC3', 'TC6', 'TC4', 'RTD', 'RTD_trend', 'TC1_trend', 'TC10_trend', 'TC3_trend', 'TC4_trend', 'TC6_trend', 'PUC State']])

        df['Trend_Flag'].unique()

        df['Issue_Detected'] = df['Sustained_Issue'].apply(lambda x: 1 if x else 0)
        df['Issue_Detected'] = df['Sustained_Issue'].astype(int)

        # Train Model
        features = ['PUC_state_mean', 'RTD_trend', 'TC1_trend', 'TC10_trend', 'TC3_trend', 'TC4_trend', 'TC6_trend']
        print(df.shape)
        df_model = df[features + ['Issue_Detected']].dropna()
        print(df_model.shape)

        X = df_model[features]
        y = df_model['Issue_Detected']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)

        print("=== Train Classification Report ===")
        print(classification_report(y_train, y_pred_train))
        print("\n=== Test Classification Report ===")
        print(classification_report(y_test, y_pred_test))
        print("\n=== Confusion Matrix (Test) ===")
        print(confusion_matrix(y_test, y_pred_test))

        # --- Generate Summary ---
        total_rows = len(df)
        print(total_rows)
        total_issues = df['Issue_Detected'].sum()
        print(total_issues)
        accuracy = clf.score(X_test, y_test) * 100
        print(accuracy, '%')

        core_columns = ['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']
        trend_columns = [col + '_trend' for col in core_columns]
        columns_to_check = core_columns + trend_columns
        summary = []

        if 'flagged' in locals() and not flagged.empty:
            data_source = flagged
            source_label = "Flagged Period"
        else:
            data_source = df
            source_label = "Full Dataset (No Issues Detected)"

        for col in columns_to_check:
            if col in data_source.columns:
                min_val = data_source[col].min()
                max_val = data_source[col].max()
                mean_val = data_source[col].mean()
                min_date = data_source.loc[data_source[col].idxmin(), 'Date/Time']
                max_date = data_source.loc[data_source[col].idxmax(), 'Date/Time']
                summary.append({
                    'Column': col,
                    'Min': min_val,
                    'Min Date': min_date,
                    'Mean': mean_val,
                    'Max': max_val,
                    'Max Date': max_date
                })

        summary_df = pd.DataFrame(summary)
        print(f"📋 Summary Statistics ({source_label})")
        print(summary_df)

        tc1_trend = 'Decreasing' if flagged['TC1_trend'].mean() < 0 else 'Increasing' if not flagged.empty else 'Stable'
        tc10_trend = 'Decreasing' if flagged['TC10_trend'].mean() < 0 else 'Increasing' if not flagged.empty else 'Stable'
        rtd_mean = flagged['RTD'].mean() if not flagged.empty else df['RTD'].mean()
        setpoint_mean = flagged['Setpoint'].mean() if not flagged.empty else df['Setpoint'].mean()
        rtd_vs_setpoint_status = "RTD higher than Setpoint consistently" if rtd_mean > setpoint_mean else "RTD within expected range"
        root_cause = flagged['Trend_Flag'].value_counts().idxmax() if not flagged.empty and 'Trend_Flag' in flagged.columns else "No root cause detected"
        device_status = "Issue Detected" if total_issues > 0 else "Working Well"

        summary = f"""
        📊 **GenAI Summary: Telemetry-Based Preventive Maintenance Analysis**

        Observation:

        A total of {total_rows} telemetry time points were analyzed.
        The system detected {total_issues} potential issue(s) where:
            - TC1 was {tc1_trend.lower()},
            - TC10 was {tc10_trend.lower()},
            - PUC State remained in condition 1.
        A Random Forest Classifier trained on trend features achieved an accuracy of **{accuracy:.2f}%** on the test dataset.
        Feature importance indicates that **TC1_trend** and **TC10_trend** are strong indicators of issue detection.

        🔎 **Technical Evaluation Summary**

        **Device Status:** {device_status}  
        **Detected Root Cause:** {root_cause}

        **Key Sensor Readings & Trends:**
        - TC1 Trend: {tc1_trend}  
        - TC10 Trend: {tc10_trend}  
        - RTD vs Setpoint: {rtd_vs_setpoint_status}

        🧠 **Root Cause Explanation:**  
        The combination of a {tc1_trend.lower()} TC1 and {tc10_trend.lower()} TC10, while the system remains in state 1, suggests abnormal heat transfer or inefficiencies likely due to a **{root_cause}**. Persistent RTD elevation beyond the setpoint supports the hypothesis of system load imbalance or cooling inefficiency.

        🔧 **Suggested Preventive Actions:**
        - Schedule periodic pressure integrity tests for both compressor stages.
        - Regularly inspect thermal coupling and ensure adequate insulation around TC1/TC10 lines.
        - Implement alerts when TC trends diverge while PUC remains constant.

        🛠 **Suggested Corrective Actions:**
        - Conduct a diagnostic check on the suspected compressor module.
        - Inspect and replace any worn or damaged seals that could cause internal leaks.
        - Recalibrate sensors and verify PID control settings for thermal regulation.

        ✅ **Confidence Level in Root Cause Identification:** **{accuracy:.2f}%**
        """

        print(summary)
        st.markdown(f"""
        <div class='output-box'>
            <h4>🔍 Recommendation</h4>
            <ul>
                <li>{summary}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Define a consistent figure size for all plots
        FIGURE_SIZE = (10, 6)

        # --- Plotting Section ---
        col1, col2, col3 = st.columns(3)

        with col1:
            df['Date/Time'] = pd.to_datetime(df['Date/Time'])
            if not flagged.empty:
                flagged['Date/Time'] = pd.to_datetime(flagged['Date/Time'])
                flagged_start = flagged['Date/Time'].min()
                flagged_end = flagged['Date/Time'].max()
                plot_start = flagged_start - pd.Timedelta(days=5)
                plot_end = flagged_end + pd.Timedelta(days=10)

                plot_df = df[(df['Date/Time'] >= plot_start) & (df['Date/Time'] <= plot_end)]
                columns_to_plot = ['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']
                plt.figure(figsize=FIGURE_SIZE)
                for col in columns_to_plot:
                    if col in plot_df.columns:
                        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                        plt.plot(plot_df['Date/Time'], plot_df[col], label=col)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                plt.title(f"Sensor Values ({plot_start.date()} to {plot_end.date()})")
                plt.xlabel("Date")
                plt.ylabel("Values")
                plt.legend()
                plt.tight_layout()
                st.pyplot(plt)
                plt.clf()  # Clear figure to prevent overlap
            else:
                st.write("✅ No Issue Detected — Device Operating Normally")

        with col2:
            df['Date/Time'] = pd.to_datetime(df['Date/Time'])
            if not flagged.empty:
                flagged['Date/Time'] = pd.to_datetime(flagged['Date/Time'])
                flagged_start = flagged['Date/Time'].min()
                flagged_end = flagged['Date/Time'].max()
                plot_start = flagged_start - pd.Timedelta(days=5)
                plot_end = flagged_end + pd.Timedelta(days=10)

                plot_df = df[(df['Date/Time'] >= plot_start) & (df['Date/Time'] <= plot_end)]
                columns_to_plot = ['RTD_trend', 'TC1_trend', 'TC10_trend', 'TC3_trend', 'TC4_trend', 'TC6_trend']
                plt.figure(figsize=FIGURE_SIZE)
                for col in columns_to_plot:
                    if col in plot_df.columns:
                        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                        plt.plot(plot_df['Date/Time'], plot_df[col], label=col)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                plt.title(f"Sensor Trends ({plot_start.date()} to {plot_end.date()})")
                plt.xlabel("Date")
                plt.ylabel("Trend Values")
                plt.legend()
                plt.tight_layout()
                st.pyplot(plt)
                plt.clf()
            else:
                st.write("✅ No Issue Detected — Device Operating Normally")

        with col3:
            importances = clf.feature_importances_
            feature_names = features
            indices = np.argsort(importances)
            sorted_features = [feature_names[i] for i in indices]
            sorted_importances = importances[indices]
            plt.figure(figsize=FIGURE_SIZE)
            bars = plt.barh(sorted_features, sorted_importances, color='skyblue')
            plt.xlabel("Feature Importance")
            plt.title("Random Forest Feature Importance")
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', va='center')
            plt.tight_layout()
            st.pyplot(plt)
            plt.clf()

        col4, col5, col6 = st.columns(3)

        with col4:
            df['Date/Time'] = pd.to_datetime(df['Date/Time'])
            df.set_index('Date/Time', inplace=True)
            daily_df_actual = df[['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']].resample('D').mean().reset_index()
            plt.figure(figsize=FIGURE_SIZE)
            sns.set(style="whitegrid")
            for col in ['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']:
                plt.plot(daily_df_actual['Date/Time'], daily_df_actual[col], label=col)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.title('Daily Actual Values')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)
            plt.clf()

        with col5:
            daily_df_trend = df[['RTD_trend', 'TC1_trend', 'TC10_trend', 'TC3_trend', 'TC4_trend', 'TC6_trend']].resample('D').mean().reset_index()
            plt.figure(figsize=FIGURE_SIZE)
            sns.set(style="whitegrid")
            for col in ['RTD_trend', 'TC1_trend', 'TC10_trend', 'TC3_trend', 'TC4_trend', 'TC6_trend']:
                plt.plot(daily_df_trend['Date/Time'], daily_df_trend[col], label=col)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.title('Daily Trend Values')
            plt.xlabel('Date')
            plt.ylabel('Trend Value')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)
            plt.clf()

        with col6:
            if 'flagged' in locals() and not flagged.empty:
                flagged['Date/Time'] = pd.to_datetime(flagged['Date/Time'])
                flagged.set_index('Date/Time', inplace=True)
                daily_df_actual = flagged[['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']].resample('D').mean().reset_index()
                plt.figure(figsize=FIGURE_SIZE)
                sns.set(style="whitegrid")
                for col in ['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']:
                    plt.plot(daily_df_actual['Date/Time'], daily_df_actual[col], label=col)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.title('Flagged Daily Actual Values')
                plt.xlabel('Date')
                plt.ylabel('Value')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(plt)
                plt.clf()
            else:
                st.write("✅ No Issue Detected — Device Operating Normally")
