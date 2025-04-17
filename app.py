from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Fix Matplotlib GUI issue
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, zscore, iqr
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from scipy.fft import fft
from pykalman import KalmanFilter
from fpdf import FPDF
import re
from scipy.stats import shapiro, skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import entropy
import mimetypes
import pickle  # already installed with pandas
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# safe_filename
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
TEMP_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("temp", exist_ok=True)  # Create temp folder if it doesn't exist

# 📌 PDF Helper
def render_table_to_pdf(pdf, df, title, max_width=190):
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, title, ln=True)
    pdf.set_font("Arial", "", 8)
    col_width = max_width / len(df.columns)

    # Header
    for col in df.columns:
        pdf.cell(col_width, 8, str(col)[:20], border=1, align="C")
    pdf.ln()

    # Rows
    for row in df.itertuples(index=False):
        for value in row:
            value_str = str(value)
            if len(value_str) > 30:
                value_str = value_str[:27] + "..."
            pdf.cell(col_width, 8, value_str, border=1, align="C")
        pdf.ln()
    pdf.ln(10)

def calculate_psi(expected, actual, buckets=10):
    """ Population Stability Index (PSI) Calculation """
    expected_perc = np.histogram(expected, bins=buckets)[0] / len(expected)
    actual_perc = np.histogram(actual, bins=buckets)[0] / len(actual)
    
    psi_values = (expected_perc - actual_perc) * np.log(expected_perc / actual_perc)
    psi_values = np.nan_to_num(psi_values)  # Handle divide-by-zero errors
    
    return np.sum(psi_values)

def plot_anomalies(df, time_col, value_col, anomaly_col, save_path='static/anomalies.png'):
    plt.figure(figsize=(12, 6))
    
    # ✅ If time_col is not a string, treat it as index
    if isinstance(time_col, str):
        x_vals = df[time_col]
        anomaly_x = df[df[anomaly_col] == 1][time_col]
    else:
        x_vals = df.index
        anomaly_x = df[df[anomaly_col] == 1].index

    plt.plot(x_vals, df[value_col], label='Value', color='blue')
    plt.scatter(
        anomaly_x,
        df[df[anomaly_col] == 1][value_col],
        color='red',
        label='Anomalies',
        zorder=5
    )
    plt.title('Time Series with Anomalies')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


@app.route("/", methods=["GET", "POST"])
def upload_file():
    global stats_table, results_df, stability_df, numeric_cols, cols_per_fig
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            if os.path.getsize(file_path) == 0:
                return "Error: Uploaded file is empty.", 400
            
            ext = os.path.splitext(filename)[1].lower()
            
            try:
                if ext in [".xlsx", ".xls"]:
                    df = pd.read_excel(file_path)
                    
                elif ext == ".csv":
                    df = pd.read_csv(file_path, encoding='utf-8', index_col=False)
                    
                else:
                    return "Error: Unsupported file type. Please upload a CSV or Excel file.", 400
                
            except Exception as e:
                return f"Error while reading the file: {str(e)}", 400

            # Load Data
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df.replace(["", "NULL", "NA", "N/A", "?", "-"], np.nan, inplace=True)

            # **Statistics Table**
            stats_table = df[numeric_cols].describe().transpose()
            stats_table["Available Row"] = df[numeric_cols].notnull().sum()
            stats_table["Missing Rows"] = df[numeric_cols].isnull().sum()
            stats_table["Duplicates"] = df[numeric_cols].apply(lambda x: x.duplicated().sum()) 

            # Step 1: Detect frequency using the first column (assumed time)
            time_col = df.columns[0]  # First column is assumed to be time
            inferred_freq = "Unknown"
            try:
                df[time_col] = pd.to_datetime(df[time_col])
                df = df.sort_values(by=time_col)
                time_diffs = df[time_col].diff().dropna()
                most_common_diff = time_diffs.mode()[0]

                freq_seconds = most_common_diff.total_seconds()

            except Exception as e:
                inferred_freq = "Unknown"

            # Step 3: Add Frequency row to stats_table (as a new summary row)
            # Add 'Frequency' as a new column (not a row!)
            stats_table["Frequency"] = freq_seconds
 
            stats_table = stats_table.rename(columns={"std": "Standard Deviation", "50%": "Median"})

            stats_table = stats_table[[
                "Available Row", "Missing Rows", "Duplicates", "Frequency" , "mean", "Standard Deviation", "min", "25%", "Median", "75%", "max"
            ]]
            stats_table_html = stats_table.to_html(classes="table table-bordered")

            # **Plots**
            num_cols = len(numeric_cols)
            cols_per_fig = 6  

            plt.close('all')  # Ensure no overlapping plots

            # Histograms
            for i in range(0, num_cols, cols_per_fig):
                subset_cols = numeric_cols[i:i + cols_per_fig]
                fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
                axes = axes.flatten()
                
                for j, col in enumerate(subset_cols):
                    axes[j].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor="black")
                    axes[j].set_title(col)
                
                for j in range(len(subset_cols), len(axes)):
                    fig.delaxes(axes[j])  

                plt.tight_layout()
                plt.savefig(f"static/histogram_{i}.png")
                plt.close()

    
                    
            
            # Function to clean filename
            def safe_filename(name):
                return re.sub(r'[^a-zA-Z0-9_-]', '_', name)
            
            # **Time Series Plots (Stacked Subplots)**
            num_plots = len(numeric_cols)
            fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(12, num_plots * 2), sharex=True)
            
            # Fix for single subplot case
            if num_plots == 1:
                axes = [axes]
                
            for i, col in enumerate(numeric_cols):
                axes[i].plot(df.index, df[col], color="blue", alpha=0.7)
                axes[i].set_title(col, fontsize=10)
                axes[i].tick_params(axis='y', labelsize=8)
                
            axes[-1].set_xlabel("Time")
            plt.tight_layout()
            plt.savefig("static/timeseries_stacked.png", bbox_inches="tight")
            plt.close()


            # ==================== STACKED ANOMALY PLOTS ====================
            fig, axes = plt.subplots(nrows=len(numeric_cols), ncols=1, figsize=(12, len(numeric_cols) * 3), sharex=True)

            # Handle single subplot
            if len(numeric_cols) == 1:
                axes = [axes]

            for idx, col in enumerate(numeric_cols):
                df['anomaly_flag'] = 0  # Reset for each column

                if df[col].dropna().shape[0] > 10:
                    clean_df = df[[col]].dropna()
                    model = IsolationForest(contamination=0.05, random_state=42)
                    preds = model.fit_predict(clean_df)
                    df.loc[clean_df.index, 'anomaly_flag'] = (preds == -1).astype(int)

                    x_vals = df.index
                    y_vals = df[col]
                    anomaly_x = df[df['anomaly_flag'] == 1].index
                    anomaly_y = df.loc[df['anomaly_flag'] == 1, col]

                    axes[idx].plot(x_vals, y_vals, label='Value', color='blue')
                    axes[idx].scatter(anomaly_x, anomaly_y, color='red', label='Anomalies', zorder=5)
                    axes[idx].set_title(f"Anomalies in {col}", fontsize=10)
                    axes[idx].tick_params(axis='y', labelsize=8)

            axes[-1].set_xlabel("Time")
            plt.tight_layout()
            plt.savefig("static/anomalies_stacked.png", bbox_inches="tight")
            plt.close()
            
            # **Correlation Heatmap (Extra Large & Readable)**
            plt.figure(figsize=(30, 24))  # Extra large for readability
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=1, annot_kws={"size": 10})
            
            # Improve label readability
            plt.xticks(rotation=45, ha="right", fontsize=14)  # Rotate X labels slightly
            plt.yticks(fontsize=14)  # Increase font size for Y labels
            plt.title("Correlation Heatmap", fontsize=20, fontweight="bold")
            
            # Save the heatmap
            plt.savefig("static/correlation_heatmap_large.png", bbox_inches="tight")
            plt.close()


            # **Sensor Health Check**
            results_df = pd.DataFrame(index=numeric_cols, columns=[
                "Missing Values %", "Duplicates %", "Outliers", "Isolation Forest Anomaly",
                "One-Class SVM Anomaly", "Kalman Filter Anomaly", "Cp", "Cpk", 
                "K-S Statistic", "K-S p-value", "Skewness", "Kurtosis", 
                "Distribution Type", "Suggestions", "Reason"
                ])


            def calculate_cp_cpk(series):
                """ Calculate Process Capability Indices (Cp & Cpk) """
                if series.isnull().all():  
                    return "Not Enough Data", "Not Enough Data"

                sigma = series.std()
                if sigma == 0 or np.isnan(sigma):  
                    return "Not Enough Data", "Not Enough Data"

                LSL, USL = series.min(), series.max()
                if LSL == USL:  
                    return "Not Enough Data", "Not Enough Data"

                Cp = (USL - LSL) / (6 * sigma)
                Cpk = min((USL - series.mean()) / (3 * sigma), (series.mean() - LSL) / (3 * sigma))

                return round(Cp, 2), round(Cpk, 2)

            for col in numeric_cols:
                
                reason = []

                # Missing Values
                missing_percentage = df[col].isnull().mean() * 100
                results_df.loc[col, "Missing Values %"] = round(missing_percentage, 2)
                if missing_percentage > 10:
                    
                    reason.append(f"High missing values: {missing_percentage:.2f}%")

                # Duplicates
                duplicate_percentage = df[col].duplicated().mean() * 100
                results_df.loc[col, "Duplicates %"] = round(duplicate_percentage, 2)
                if duplicate_percentage > 95:
                    
                    reason.append("Too many duplicate values (sensor might be stuck).")

                # Outliers
                iqr_value = iqr(df[col].dropna())
                mean_value = df[col].mean()
                std_dev = df[col].std()

                # IQR Method
                if iqr_value == 0:
                    outlier_count_iqr = 0
                else:
                    outlier_count_iqr = ((df[col].dropna() < df[col].quantile(0.25) - 1.5 * iqr_value) |
                                         (df[col].dropna() > df[col].quantile(0.75) + 1.5 * iqr_value)).sum()
                    
                # 5 SD Method
                if std_dev == 0 or np.isnan(std_dev):
                    outlier_count_sd = 0
                else:
                    outlier_count_sd = ((df[col].dropna() < mean_value - 5 * std_dev) |
                                        (df[col].dropna() > mean_value + 5 * std_dev)).sum()
                    
                # Take the maximum outlier count from both methods

                outlier_count = max(outlier_count_iqr, outlier_count_sd)
                results_df.loc[col, "Outliers"] = outlier_count

                # Dynamic thresholding

                threshold = max(10, int(0.01 * len(df)))
                if outlier_count > threshold:
                    reason.append(f"Too many outliers detected: {outlier_count} (Threshold: {threshold}, using IQR & 5 SD)")
            
                # **New: Compute Distribution Type**
                col_data = df[col].dropna()
                if len(col_data) > 10:
                    if col_data.nunique() > 1:  # Ensure there are at least two unique values
                          skewness = skew(col_data)
                          kurt = kurtosis(col_data)
                    else:
                          skewness, kurt = "Not Enough Data", "Not Enough Data"
                          
                    results_df.loc[col, "Skewness"] = round(skewness, 3) if isinstance(skewness, float) else skewness
                    results_df.loc[col, "Kurtosis"] = round(kurt, 3) if isinstance(kurt, float) else kurt

                    if col_data.nunique() > 1:
                          shapiro_stat, shapiro_p = shapiro(col_data)
                    else:
                          shapiro_p = "Not Enough Data"  
                        


                    # Check if we have enough numerical data before classification
                    if isinstance(shapiro_p, float) and isinstance(skewness, float) and isinstance(kurt, float):
                        if shapiro_p > 0.05 and abs(skewness) < 0.5:
                            dist_type = "Normal Distribution"
                        elif skewness > 1:
                            dist_type = "Right-Skewed"
                        elif skewness < -1:
                            dist_type = "Left-Skewed"
                        elif kurt > 3:
                            dist_type = "Heavy-Tailed (Leptokurtic)"
                        elif kurt < 3:
                            dist_type = "Light-Tailed (Platykurtic)"
                        else:
                            dist_type = "Unknown Distribution"
                    else:
                        dist_type = "Not Enough Data"

                    # Ensure skewness is a number before rounding
                    results_df.loc[col, "Skewness"] = round(skewness, 3) if isinstance(skewness, (int, float)) else "Not Enough Data"
                    results_df.loc[col, "Kurtosis"] = round(kurt, 3) if isinstance(kurt, (int, float)) else "Not Enough Data"
                    # Ensure shapiro_p is a valid number before rounding
                    # results_df.loc[col, "Shapiro P-Value"] = round(shapiro_p, 5) if isinstance(shapiro_p, (int, float)) else "Not Enough Data"

                    results_df.loc[col, "Distribution Type"] = dist_type
                else:
                    results_df.loc[col, "Skewness"] = "Not Enough Data"
                    results_df.loc[col, "Kurtosis"] = "Not Enough Data"
                    results_df.loc[col, "Shapiro P-Value"] = "Not Enough Data"
                    results_df.loc[col, "Distribution Type"] = "Not Enough Data"

                    col_values = df[col].dropna().values.reshape(-1, 1)  # Ensure column is properly shaped
                    
                    # Dynamic Isolation Forest contamination
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr_value = q3 - q1
                    std_dev = df[col].std()
                    
                    # Avoid division by zero or NaN
                    if np.isnan(std_dev) or std_dev == 0:
                          contamination_rate = 0.01  # Set a default value when std is invalid
                    else:
                          contamination_rate = min(0.05, (iqr_value / std_dev) * 0.01)
                          
                    contamination_rate = max(0.01, contamination_rate)  # Ensure it stays within range

                    
                # Default values to prevent errors
                anomaly_count = "Not Enough Data"
                svm_anomalies_count = "Not Enough Data"


                
                col_values = df[col].dropna().values.reshape(-1, 1)  # Ensure valid input shape
                
                # ✅ Isolation Forest Anomaly Detectio
                
                if len(col_values) > 10:
                    model = IsolationForest(contamination=0.05, random_state=42)
                    anomaly_labels = model.fit_predict(col_values)
                    anomaly_count = (anomaly_labels == -1).sum()
                    
                # ✅ One-Class SVM Anomaly Detection
                if len(col_values) > 10:
                    nu_value = max(0.01, min(0.05, 1.0 / np.sqrt(len(col_values))))  # Dynamic nu value
                    svm_model = OneClassSVM(nu=nu_value, kernel="rbf")
                    svm_labels = svm_model.fit_predict(col_values)
                    svm_anomalies_count = (svm_labels == -1).sum()
                    
                # ✅ Store results in DataFrame
                results_df.loc[col, "Isolation Forest Anomaly"] = anomaly_count
                results_df.loc[col, "One-Class SVM Anomaly"] = svm_anomalies_count
                
                # ✅ Add reason if anomalies are too high
                # 
                if isinstance(anomaly_count, int) and anomaly_count > max(5, int(0.01 * len(df))):
                    reason.append(f"Isolation Forest detected anomalies: {anomaly_count}")
                    
                if isinstance(svm_anomalies_count, int) and svm_anomalies_count > max(5, int(0.01 * len(df))):
                    reason.append(f"One-Class SVM detected anomalies: {svm_anomalies_count}")

                results_df.loc[col, "Reason"] = "; ".join(reason) if reason else "No significant issues detected."

                # Drift Detection
                # already inside `for col in numeric_cols:` loop
                # # so directly use
                first_half = df.iloc[:len(df)//2]
                second_half = df.iloc[len(df)//2:]
                
                first_half_clean = first_half[col].dropna()
                second_half_clean = second_half[col].dropna()
                
                
                if first_half_clean.count() > 10 and second_half_clean.count() > 10:
                     first_half_normalized = zscore(first_half_clean)
                     second_half_normalized = zscore(second_half_clean)
                     
                     
                     stat, p = ks_2samp(first_half_normalized, second_half_normalized)
                     results_df.loc[col, "K-S Statistic"] = round(stat, 5)
                     results_df.loc[col, "K-S p-value"] = round(p, 5)
                     
                else:
                     results_df.loc[col, "K-S Statistic"] = None
                     results_df.loc[col, "K-S p-value"] = None

                try:
                    first_half = df.iloc[:len(df)//2]
                    second_half = df.iloc[len(df)//2:]
                    first_half_clean = first_half[col].dropna()
                    second_half_clean = second_half[col].dropna()
                    
                    if first_half_clean.count() > 10 and second_half_clean.count() > 10:
                        first_half_normalized = zscore(first_half_clean)
                        second_half_normalized = zscore(second_half_clean)
                        
                        stat, p = ks_2samp(first_half_normalized, second_half_normalized)
                        
                        results_df.loc[col, "K-S Statistic"] = round(stat, 5)
                        
                        results_df.loc[col, "K-S p-value"] = round(p, 5)
                        
                    else:
                        results_df.loc[col, "K-S Statistic"] = None
                        results_df.loc[col, "K-S p-value"] = None
                        
                except Exception as e:
                    results_df.loc[col, "K-S Statistic"] = None
                    results_df.loc[col, "K-S p-value"] = None



                # ✅ Improved Kalman Filter Anomaly Detection
                try:
                    valid_col_values = df[col].dropna()
                    
                    # ✅ Ensure minimum number of valid data points
                    if len(valid_col_values) >= 10:  
                        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
                        kf = kf.em(valid_col_values.values.reshape(-1, 1), n_iter=5)  # Ensure proper shape
                        state_means, _ = kf.filter(valid_col_values.values.reshape(-1, 1))
                        
                        # ✅ Compute residuals safely
                        # 
                        residuals = valid_col_values.values - state_means.flatten()
                        
                        # ✅ Count anomalies based on 3-sigma rule
                        kalman_anomaly = (np.abs(residuals) > 3 * np.std(residuals)).sum()
                        
                        results_df.loc[col, "Kalman Filter Anomaly"] = kalman_anomaly if kalman_anomaly > 0 else "No Anomalies"
                        
                        if kalman_anomaly > 5:
                            reason.append(f"Kalman filter detected anomalies: {kalman_anomaly}")
                    else:
                        results_df.loc[col, "Kalman Filter Anomaly"] = "Not Enough Data"
                            
                except Exception as e:
                    results_df.loc[col, "Kalman Filter Anomaly"] = f"Error: {str(e)}"


                # Cp and Cpk Calculation
                Cp, Cpk = calculate_cp_cpk(df[col])
                results_df.loc[col, "Cp"] = round(Cp, 3) if isinstance(Cp, (int, float)) else "Not Enough Data"
                results_df.loc[col, "Cpk"] = round(Cpk, 3) if isinstance(Cpk, (int, float)) else "Not Enough Data"
                if Cpk != "Not Enough Data" and Cpk < 1.33:
                    reason.append(f"Low Cpk value: {Cpk} (Process capability issue)")

                # Immediately mark "Unhealthy" if missing values > 96% or duplicates > 90%
                if missing_percentage > 96 or duplicate_percentage > 99:
                    results_df.loc[col, "Suggestions"] = "❌ Unhealthy"
                    
                elif (
                    missing_percentage <= 10 or  # Acceptable missing values
                    duplicate_percentage <= 95 or  # Not too many duplicates
                    outlier_count <= max(10, int(0.01 * len(df))) or  # Not excessive outliers
                    (isinstance(anomaly_count, int) and anomaly_count <= max(5, int(0.01 * len(df)))) or  # Isolation Forest anomaly check
                    (isinstance(svm_anomalies_count, int) and svm_anomalies_count <= max(5, int(0.01 * len(df)))) or  # SVM anomaly check
                    # results_df.loc[col, "K-S Drift Test"] == "✅ No Drift" or  # No significant data drift
                    (isinstance(kalman_anomaly, int) and kalman_anomaly <= 5) or  # Kalman filter anomaly check
                    (isinstance(Cpk, (int, float)) and Cpk >= 1.33)  # Sufficient process capability
                    ):
                    
                    results_df.loc[col, "Suggestions"] = "✅ Healthy"
                    
                else:
                    results_df.loc[col, "Suggestions"] = "❌ Unhealthy"




                results_df.loc[col, "Reason"] = "; ".join(reason) if reason else "No significant issues detected."



            # ======= Missing Timestamp Tend Table =======
            missing_timestamps_summary = []

            # Determine frequency string compatible with pandas
            freq_map = {
                "Secondly": "S",
                "Minutely": "T",
                "Hourly": "H",
                "Daily": "D",
                "Weekly": "W"
            }
            freq_str = freq_map.get(inferred_freq, None)

            if freq_str:
                # Ensure datetime and set index
                df = df.set_index(time_col)
                df.index = pd.to_datetime(df.index)

                # Create complete time index
                full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq_str)

                # Reindex to find missing
                df_reindexed = df.reindex(full_index)

                for col in numeric_cols:
                    missing_times = df_reindexed[df_reindexed[col].isna()].index
                    if not missing_times.empty:
                        missing_list_str = ", ".join(missing_times.strftime("%Y-%m-%d %H:%M:%S").tolist())
                    else:
                        missing_list_str = "No Missing Timestamps"
                    
                    missing_timestamps_summary.append({
                        "Column": col,
                        "Missing Timestamps": missing_list_str
                    })

                missing_time_df = pd.DataFrame(missing_timestamps_summary)
            else:
                missing_time_df = pd.DataFrame([{"Column": "N/A", "Missing Timestamps": "Unsupported or Unknown Frequency"}])

            missing_time_html = missing_time_df.to_html(classes="table table-bordered", index=False)



            # ========== CLUSTERING WITH AUTO k SELECTION ==========

            # Drop rows with missing numeric values for clustering
            cluster_df = df[numeric_cols].dropna()

            if len(cluster_df) >= 10:  # Ensure enough data for clustering
                # Standardize the features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(cluster_df)

                # Try multiple k values
                inertia = []
                silhouette = []
                K_range = range(2, min(10, len(cluster_df)))  # Safe upper bound on k

                for k in K_range:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    labels = kmeans.fit_predict(X_scaled)
                    inertia.append(kmeans.inertia_)
                    silhouette.append(silhouette_score(X_scaled, labels))

                # Choose k with best silhouette score
                best_k = K_range[silhouette.index(max(silhouette))]

                # 🔽 ADD THIS BLOCK HERE
                plt.figure(figsize=(12, 5))

                # Inertia Plot (Elbow)
                plt.subplot(1, 2, 1)
                plt.plot(K_range, inertia, marker='o')
                plt.title("Elbow Method (Inertia)")
                plt.xlabel("Number of clusters (k)")
                plt.ylabel("Inertia")

                # Silhouette Score Plot
                plt.subplot(1, 2, 2)
                plt.plot(K_range, silhouette, marker='o', color='green')
                plt.title("Silhouette Score")
                plt.xlabel("Number of clusters (k)")
                plt.ylabel("Score")

                plt.tight_layout()
                plt.savefig("static/k_selection.png")
                plt.close()


                # Final clustering with best k
                final_kmeans = KMeans(n_clusters=best_k, random_state=42)
                final_labels = final_kmeans.fit_predict(X_scaled)

                df.loc[cluster_df.index, 'Cluster_Label'] = final_labels

                # Cluster summary
                cluster_summary = df.groupby("Cluster_Label")[numeric_cols].mean().round(2)
                cluster_summary_html = cluster_summary.to_html(classes="table table-bordered")

                # Optional: plot PCA for cluster visualization
                from sklearn.decomposition import PCA

                pca = PCA(n_components=2)
                pca_components = pca.fit_transform(X_scaled)

                pca_df = pd.DataFrame(pca_components, columns=["PC1", "PC2"])
                pca_df["Cluster"] = final_labels

                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", palette="Set1")
                plt.title(f"K-Means Clustering (k = {best_k})")
                plt.savefig("static/cluster_plot.png")
                plt.close()

            else:
                cluster_summary_html = "<p>Not enough data for clustering.</p>"




            # ========================== Stability Analysis ========================== # 
           
            
            target_col = numeric_cols[-1] if len(numeric_cols) > 1 else None
            if target_col:
                X = df[numeric_cols].drop(columns=[target_col]).dropna()
                y = df[target_col].dropna()
                # Ensure no NaN rows
                data = pd.concat([X, y], axis=1).dropna()
                X = data.iloc[:, :-1]  # Features (All columns except last)
                y = data.iloc[:, -1]   # Target (Last column)
                
                # Check if X and y have the same number of row
                # 
                print(f"X shape: {X.shape}, y shape: {y.shape}")
                # Now safely split
                # 
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                
                mae_scores, mse_scores = [], []
                for i in range(10):
                    model = RandomForestRegressor(n_estimators=100, random_state=i)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mae_scores.append(mean_absolute_error(y_test, y_pred))
                    mse_scores.append(mean_squared_error(y_test, y_pred))
                
                stability_result_df = pd.DataFrame([ {
                    "Mean MAE": np.mean(mae_scores),
                    "MAE Variance": np.var(mae_scores),
                    "Mean MSE": np.mean(mse_scores),
                    "MSE Variance": np.var(mse_scores)
                }])
                
            else:
                stability_result_df = pd.DataFrame()
            
            
            # ======================== Stability Analysis Ends ========================= #

            # ========================== Feature Stability Analysis ========================== #
            stability_columns = [
                "Feature Name", "CV", "PSI", "KS Test", "KL Divergence", "Outliers (%)", "Rolling Mean Stability", "PCA Influence", "Stability Decision"]

            stability_results = []

            for col in numeric_cols:
                col_data = df[col].dropna()

                # **1️⃣ Compute Coefficient of Variation (CV)**
                cv = np.std(col_data) / np.mean(col_data) if np.mean(col_data) != 0 else 0

                # **2️⃣ Compute PSI (Compare First Half vs Second Half)**
                mid_index = len(col_data) // 2
                psi_value = calculate_psi(col_data[:mid_index], col_data[mid_index:]) if len(col_data) > 20 else np.nan

                # **3️⃣ Kolmogorov-Smirnov (KS) Test**
                ks_stat, ks_p = ks_2samp(col_data[:mid_index], col_data[mid_index:]) if len(col_data) > 20 else (np.nan, np.nan)

                # **4️⃣ KL Divergence (Information Loss between Two Halves)**
                hist_1, _ = np.histogram(col_data[:mid_index], bins=10, density=True)
                hist_2, _ = np.histogram(col_data[mid_index:], bins=10, density=True)
                kl_div = entropy(hist_1, hist_2) if len(col_data) > 20 else np.nan

                # **5️⃣ Outliers Percentage (Z-score method)**
                z_scores = np.abs(zscore(col_data)) if len(col_data) > 10 else np.array([])
                outliers_pct = (np.sum(z_scores > 3) / len(col_data)) * 100 if len(col_data) > 10 else np.nan

                # **6️⃣ Rolling Mean Stability (Check Time-based Variability)**
                rolling_mean = col_data.rolling(window=10).mean()
                rolling_std = col_data.rolling(window=10).std()
                stability_score = np.std(rolling_mean.dropna()) / np.std(col_data) if len(rolling_mean.dropna()) > 10 else np.nan
                rolling_stability = "Stable" if stability_score < 0.5 else "Unstable"

                # **7️⃣ PCA Influence (Feature Variance Contribution in PCA)**
                if len(numeric_cols) > 1 and len(col_data) > 10:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=1)
                    transformed = pca.fit_transform(df[numeric_cols].dropna())
                    pca_variance = pca.explained_variance_ratio_[0]
                    pca_stability = "Stable" if pca_variance > 0.05 else "Unstable"
                else:
                    pca_stability = "Not Computed"

                # **8️⃣ Final Stability Decision**
                unstable_metrics = sum([
                    cv > 0.5,  # High variation
                    psi_value > 0.25 if not np.isnan(psi_value) else False,  # High PSI = Drift
                    ks_stat > 0.1 if not np.isnan(ks_stat) else False,  # KS test shows significant difference
                    kl_div > 0.1 if not np.isnan(kl_div) else False,  # KL Divergence high
                    outliers_pct > 10 if not np.isnan(outliers_pct) else False,  # Too many outliers
                    rolling_stability == "Unstable",
                    pca_stability == "Unstable"
                ])
    
                stability_decision = "Stable" if unstable_metrics <= 3 else "Unstable"

                # Store results
                stability_results.append([
                    col, round(cv, 3), round(psi_value, 3) if not np.isnan(psi_value) else "N/A",
                    round(ks_stat, 3) if not np.isnan(ks_stat) else "N/A",
                    round(kl_div, 3) if not np.isnan(kl_div) else "N/A",
                    round(outliers_pct, 2) if not np.isnan(outliers_pct) else "N/A",
                    rolling_stability, pca_stability, stability_decision
                ])

        # Create DataFrame
                feature_stability_df = pd.DataFrame(stability_results, columns=stability_columns)




            # Save DataFrames to temporary files
            stats_table.to_pickle("temp/stats_table.pkl")
            results_df.to_pickle("temp/results_df.pkl")
            feature_stability_df.to_pickle("temp/stability_df.pkl")
            
            # Convert to HTML for rendering
            stability_result_html = stability_result_df.to_html(classes="table table-bordered")
            feature_stability_html = feature_stability_df.to_html(classes="table table-bordered", index=False)
            # Save numeric column info for PDF route
            with open("temp/numeric_cols.pkl", "wb") as f:
                pickle.dump((numeric_cols, cols_per_fig), f)






            return render_template(
                "report.html",
                stats_table=stats_table_html,
                results_table=results_df.to_html(classes="table table-bordered"),
                num_cols=num_cols,
                cols_per_fig=cols_per_fig,
                timeseries_plot="static/timeseries_stacked.png",
                anomaly_plot="static/anomalies_stacked.png",
                correlation_plot="static/correlation_heatmap_large.png",
                stability_result=stability_result_html,
                stability_analysis=feature_stability_html,
                missing_time_table=missing_time_html,
                cluster_summary=cluster_summary_html,
                cluster_plot="static/cluster_plot.png")


    return render_template("index.html")



@app.route("/download_pdf")
def download_pdf():
    try:
        stats_table = pd.read_pickle(os.path.join(TEMP_FOLDER, "stats_table.pkl"))
    except Exception as e:
        return f"Error: {str(e)}", 400

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Sensor Data Statistics Report", ln=True, align="C")
    pdf.ln(10)

    render_table_to_pdf(pdf, stats_table, "📊 Statistics Summary")

    pdf_path = os.path.join("static", "sensor_report.pdf")
    pdf.output(pdf_path)

    return send_file(pdf_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)