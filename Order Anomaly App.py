
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest
from plotly.subplots import make_subplots
import xgboost as xgb

# --- Croston’s Method ---
def croston(ts, alpha=0.1, n_forecast=12):
    demand = np.array(ts)
    n = len(demand)
    if not np.any(demand > 0):
        return np.zeros(n_forecast)
    forecast = np.zeros(n + n_forecast)
    first_occurrence = np.argmax(demand > 0)
    a = demand[first_occurrence]
    p, q = 1, 1
    forecast[first_occurrence] = a / p
    
    for t in range(first_occurrence + 1, n):
        if demand[t] > 0:
            a = alpha * demand[t] + (1 - alpha) * a
            p = alpha * q + (1 - alpha) * p
            q = 1
        else:
            q += 1
        forecast[t] = a / p
    
    # Bias correction (SBA)
    forecast[n:] = (a / p) * (1 - alpha/2)
    return forecast[-n_forecast:]

# --- Streamlit UI ---
st.title("Demand Forecasting & Inventory DIOH Calculator")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    rawData = pd.read_csv(uploaded_file)

    # Ensure OrderWeek is datetime
    rawData["OrderWeek"] = pd.to_datetime(rawData["OrderWeek"])

    # --- Pivot ---
    data = rawData.pivot_table(
        index="Key", 
        columns="OrderWeek", 
        values="sum_OrderQuantityBase", 
        fill_value=0    
    )

    # --- Feature Engineering ---
    total_periods = data.shape[1]
    non_zero_counts = (data > 0).sum(axis=1)
    adi = total_periods / non_zero_counts
    mean = data.mean(axis=1)
    std = data.std(axis=1)
    cv2 = (std/mean) ** 2
    SeasonalIndex = data.std(axis=1) / data.mean(axis=1)

    df = pd.DataFrame({
        "Key": data.index,
        "ADI": adi.values,
        "SeasonalIndex": SeasonalIndex,
        "CV2": cv2
    }).reset_index(drop=True)

    # --- Demand classification ---
    conditions = [
        (df["ADI"] < 1.32) & (df["CV2"] < 0.49),
        (df["ADI"] >= 1.32) & (df["CV2"] < 0.49),
        (df["ADI"] < 1.32) & (df["CV2"] >= 0.49),
        (df["ADI"] >= 1.32) & (df["CV2"] >= 0.49)
    ]
    choices = ["Smooth","Intermittent","Erratic","Lumpy"]
    df["Demand type"] = np.select(conditions, choices, default="Unclassified")

    # --- Map ZB NAME and Material ---
    zb_map = rawData[["Key", "ZB NAME", "MaterialNumberHarmonized"]].drop_duplicates()
    DemandDF = df.merge(zb_map, on="Key", how="left")

    # --- Merge with pivoted data ---
    data_reset = data.reset_index()
    merged = pd.merge(data_reset, DemandDF, on="Key", how="inner")

    # --- Sample 10 per demand type ---
    sampled_data = (
        merged.groupby("Demand type", group_keys=False)
        .apply(lambda x: x.sample(min(10, len(x)), random_state=42))
        .reset_index(drop=True)
    )
    data = sampled_data.copy()

    # --- Forecasting ---
    results = []
    for _, row in data.iterrows():
        key = row["Key"]
        demand_type = row["Demand type"]
        series = row.drop(["Key","ADI","SeasonalIndex","CV2","Demand type",
                           "ZB NAME","MaterialNumberHarmonized"]).values.astype(float)
        weeks = [c for c in data.columns if c not in ["Key","ADI","SeasonalIndex","CV2","Demand type","ZB NAME","MaterialNumberHarmonized"]]

        # Train/test split
        train_size = 44
        test_size = 12
        if len(series) < train_size + test_size:
            continue
        train, test = series[:train_size], series[train_size:train_size+test_size]

        if demand_type == "Smooth":
            model = ExponentialSmoothing(train, trend="add", seasonal=None)
            fit = model.fit()
            forecast = fit.forecast(test_size)

        elif demand_type == "Lumpy":
            forecast = croston(train, alpha=0.1, n_forecast=test_size)

        elif demand_type == "Erratic":
            ts = pd.DataFrame({"y": train})
            ts["lag1"] = ts["y"].shift(1)
            ts["week"] = np.arange(len(ts)) % 52
            ts = ts.dropna()
            if len(ts) > 10:
                X = ts[["lag1", "week"]]
                y = ts["y"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                model = xgb.XGBRegressor(objective="reg:squarederror")
                model.fit(X_train, y_train)
                last_lag = ts["y"].iloc[-1]
                future_weeks = [(last_lag, (len(ts)+i) % 52) for i in range(1, test_size+1)]
                X_future = pd.DataFrame(future_weeks, columns=["lag1", "week"])
                forecast = model.predict(X_future)
            else:
                forecast = np.repeat(train.mean(), test_size)

        else:
            forecast = np.repeat(train.mean(), test_size)

        # RMSE
        rmse = np.sqrt(mean_squared_error(test, forecast))

        results.append({
            "Key": key,
            "ZB NAME": row["ZB NAME"],
            "MaterialNumberHarmonized": row["MaterialNumberHarmonized"],
            "Demand type": demand_type,
            "Forecast": forecast,
            "RMSE": rmse
        })

    forecast_df = pd.DataFrame(results)
    # --- Meta columns --

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(["Forecasting", "Inventory & DIOH", "Raw Data"])

    # --- Tab 1: Forecasting ---
    with tab1:
        st.subheader("Forecasting Analysis")

        # --- User selections ---
        selected_zb = st.selectbox("Select Distributor NAME", forecast_df["ZB NAME"].dropna().unique())
        filtered = forecast_df[forecast_df["ZB NAME"] == selected_zb]

        selected_material = st.selectbox(
            "Select Material Number",
            filtered["MaterialNumberHarmonized"].dropna().unique()
        )

        row = filtered[filtered["MaterialNumberHarmonized"] == selected_material].iloc[0]
        demand_type = row["Demand type"]
        forecast = row["Forecast"]
 

        # --- Meta and week columns ---
        meta_cols = ["Key","ADI","SeasonalIndex","CV2","Demand type","ZB NAME","MaterialNumberHarmonized"]
        week_cols = [c for c in data.columns if c not in meta_cols]

        # --- Actual demand series ---
        actual = data.loc[data["Key"] == row["Key"], week_cols].values.flatten().astype(float)
        weeks_actual = pd.to_datetime(week_cols)

        # --- Train/test split ---
        train_size = 44
        test_size = 11
        train = actual[:train_size]
        test = actual[train_size:train_size+test_size]
        weeks_forecast = weeks_actual[train_size:train_size+test_size]

        # --- Mean and thresholds ---
        mean_val = actual.mean()
        std_val = actual.std()
        upper_threshold = mean_val + 1.65 * std_val
        lower_threshold = mean_val - 1.65 * std_val

        # --- Feature columns (probabilistic randomness) ---
        prob_promo = np.where(actual > upper_threshold, 0.7, 0.2)
        prob_holiday = np.where(actual > upper_threshold, 0.5, 0.1)
        promotional_event = (np.random.rand(len(actual)) < prob_promo).astype(int)
        national_holiday = (np.random.rand(len(actual)) < prob_holiday).astype(int)

        # --- Isolation Forest (actual demand) ---
        X_actual = np.column_stack([actual, promotional_event, national_holiday])
        iso = IsolationForest(contamination=0.1, random_state=42)
        iso.fit(X_actual)
        ml_outliers_actual = np.where(iso.predict(X_actual) == -1, 1, 0)

        # --- Forecast features + ML outliers ---
        prob_promo_forecast = np.where(forecast > upper_threshold, 0.7, 0.2)
        prob_holiday_forecast = np.where(forecast > upper_threshold, 0.5, 0.1)
        promotional_event_forecast = (np.random.rand(len(forecast)) < prob_promo_forecast).astype(int)
        national_holiday_forecast = (np.random.rand(len(forecast)) < prob_holiday_forecast).astype(int)

        X_forecast = np.column_stack([forecast, promotional_event_forecast, national_holiday_forecast])
        ml_outliers_forecast = np.where(iso.predict(X_forecast) == -1, 1, 0)
        
        # --- Plotly Figure ---
        fig = go.Figure()

        # Actual Demand line
        fig.add_trace(go.Scatter(
            x=weeks_actual,
            y=actual,
            mode="lines+markers",
            name="Actual Demand",
            line=dict(color="blue"),
            marker=dict(symbol="circle", size=8)
        ))

        # Highlight statistical outliers
        outlier_mask = (actual > upper_threshold) | (actual < lower_threshold)
        fig.add_trace(go.Scatter(
            x=weeks_actual[outlier_mask],
            y=actual[outlier_mask],
            mode="markers",
            name="Statistical Outlier (> ±1.65σ)",
            marker=dict(color="red", size=10, symbol="diamond"),
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=weeks_forecast,
            y=forecast,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="green", dash="dash"),
            marker=dict(symbol="x", size=8)
        ))

        # Reference lines
        fig.add_hline(y=mean_val, line=dict(color="orange", dash="dash"), annotation_text="Mean", annotation_position="top left")
        fig.add_hline(y=upper_threshold, line=dict(color="red", dash="dot"), annotation_text="Mean + 1.65σ", annotation_position="top left")
        fig.add_hline(y=lower_threshold, line=dict(color="red", dash="dot"), annotation_text="Mean - 1.65σ", annotation_position="bottom left")

        # Layout customization
        fig.update_layout(
            title=f"ZB: {row['ZB NAME']} | Material: {row['MaterialNumberHarmonized']} | Demand Type: {row['Demand type']}",
            xaxis_title="Week",
            yaxis_title="Order Quantity",
            legend=dict(title="Legend", orientation="h", y=-0.2),
            template="plotly_white",
            hovermode="x unified"
        )

        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # --- Weekly-level table for selected ZB and Material ---
        mean_val = actual.mean()
        std_val = actual.std()
        upper_threshold = mean_val + 1.65 * std_val
        lower_threshold = mean_val - 1.65 * std_val

        weekly_rows = []
        for i in range(len(weeks_actual)):
            actual_val = actual[i]
            week = weeks_actual[i]

            # Forecast alignment (only for test weeks)
            if i >= train_size and (i - train_size) < len(forecast):
                forecast_val = forecast[i - train_size]
                if actual_val != 0:
                    accuracy = abs(actual_val - forecast_val) / actual_val * 100
                else:
                    accuracy = np.nan
            else:
                forecast_val = np.nan
                accuracy = np.nan

            # Feature flags
            promo_flag = promotional_event[i]
            holiday_flag = national_holiday[i]

            # --- Outlier detection ---
            # Statistical outlier (actual vs thresholds)
            stat_outlier = int((actual_val > upper_threshold) or (actual_val < lower_threshold))

            # ML outlier (Isolation Forest on actual demand)
            ml_outlier_actual = ml_outliers_actual[i] if i < len(ml_outliers_actual) else np.nan

            # Forecast ML outlier (Isolation Forest on forecast demand)
            if i >= train_size and (i - train_size) < len(ml_outliers_forecast):
                ml_outlier_forecast = ml_outliers_forecast[i - train_size]
            else:
                ml_outlier_forecast = np.nan

            weekly_rows.append({
                "Order Week": week.strftime("%Y-%m-%d"),
                "ZB Name": row["ZB NAME"],
                "Material": row["MaterialNumberHarmonized"],
                "Demand Actual": actual_val,
                "Demand Forecast": forecast_val,
                "Forecast Accuracy (%)": round(accuracy, 2) if not np.isnan(accuracy) else np.nan,
                "Promotional Event": promo_flag,
                "National Holiday": holiday_flag,
                "Statistical Outlier": stat_outlier,
                "ML Outlier": ml_outlier_actual
            })

        weekly_df = pd.DataFrame(weekly_rows)

        st.subheader("Weekly Forecast Accuracy and Outlier Detection")
        st.dataframe(weekly_df)


    # --- Tab 2: Inventory & DIOH ---
    with tab2:
        st.subheader("Inventory & DIOH Analysis")

        # --- Metadata & Weeks ---
        meta_cols = ["Key","ADI","SeasonalIndex","CV2","Demand type","ZB NAME","MaterialNumberHarmonized"]
        week_cols = [c for c in data.columns if c not in meta_cols]

        # --- Slicers ---
        inventory_data = pd.DataFrame({
            "ZB NAME": forecast_df["ZB NAME"],
            "MaterialNumberHarmonized": forecast_df["MaterialNumberHarmonized"],
            "TargetDIOH": np.random.choice([10, 20, 30], size=len(forecast_df))
        })

        selected_zb = st.selectbox(
            "Select Distributor",
            inventory_data["ZB NAME"].unique(),
            key="inventory_zb"
        )
        filtered_inv = inventory_data[inventory_data["ZB NAME"] == selected_zb]

        selected_material = st.selectbox(
            "Select Material Number",
            filtered_inv["MaterialNumberHarmonized"].unique(),
            key="inventory_material"
        )

        row_inv = filtered_inv[filtered_inv["MaterialNumberHarmonized"] == selected_material].iloc[0]

        # --- Actual Demand Series ---
        actual_full = data.loc[data["MaterialNumberHarmonized"] == selected_material, week_cols].values.flatten()

        # --- Week slicer with start & end ---
        max_weeks = len(actual_full)
        selected_range = st.slider(
            "Select week range",
            min_value=1,
            max_value=max_weeks,
            value=(44, max_weeks),  
            step=1
        )

        # Slice demand series based on user selection
        start_week, end_week = selected_range
        actual = actual_full[start_week-1:end_week]
        week_labels = week_cols[start_week-1:end_week]   # use actual week/date labels

        # --- Production Simulation ---
        production_qty = actual + (np.random.choice([-1, 1], size=len(actual)) * (0.1 * actual)).astype(int)

        # --- Inventory & Shipment Simulation ---
        ending_inventory, new_inventory, shipment_qty_adj = [], [], []
        first_demand = actual[0] if len(actual) > 0 else 0

        if first_demand > 0:
            prev_inventory = int(first_demand + np.random.choice([-1, 1]) * (0.5 * first_demand))
        else:
            prev_inventory = np.random.randint(1000, 2000)  # smaller default

        for i in range(len(actual)):
            ending_inventory.append(prev_inventory)

            # Shipment qty = min(demand, Ending Inventory + Production qty)
            shipment_qty = min(actual[i], prev_inventory + production_qty[i])
            shipment_qty_adj.append(shipment_qty)

            # New inventory = Ending inventory + Production − Shipments
            new_inv = prev_inventory + production_qty[i] - shipment_qty
            new_inventory.append(new_inv)

            prev_inventory = new_inv

        # --- DIOH Calculations ---
        dioh_current = [(ei / (s/7)) if s > 0 else np.nan for ei, s in zip(ending_inventory, shipment_qty_adj)]
        dioh_post_order = [(ni / (s/7)) if s > 0 else np.nan for ni, s in zip(new_inventory, shipment_qty_adj)]

        flags_current = [dc < row_inv["TargetDIOH"] if not np.isnan(dc) else False for dc in dioh_current]
        flags_post = [dp < row_inv["TargetDIOH"] if not np.isnan(dp) else False for dp in dioh_post_order]

        # New columns
        dioh_gap = [row_inv["TargetDIOH"] - dp if not np.isnan(dp) else np.nan for dp in dioh_post_order]
        dioh_push_flag = [1 if (not np.isnan(dp) and dp > row_inv["TargetDIOH"]) else 0 for dp in dioh_post_order]


        # --- Chart using Plotly with make_subplots ---

        # Create figure with secondary y-axis enabled
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add Bar charts (Inventory Units)
        fig.add_trace(
            go.Bar(
                x=week_labels,
                y=ending_inventory,
                name="Ending Inventory",
                marker_color="blue",
                opacity=0.7
            ),
            secondary_y=False, # Use primary Y axis
        )

        fig.add_trace(
            go.Bar(
                x=week_labels,
                y=new_inventory,
                name="New Inventory",
                marker_color="green",
                opacity=0.7
            ),
            secondary_y=False, # Use primary Y axis
        )

        # Add Line charts (DIOH) on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=week_labels,
                y=dioh_current,
                mode="lines+markers",
                name="DIOH Current",
                marker=dict(color="purple", symbol="circle")
            ),
            secondary_y=True, # Use secondary Y axis
        )

        fig.add_trace(
            go.Scatter(
                x=week_labels,
                y=dioh_post_order,
                mode="lines+markers",
                name="DIOH Post Order",
                marker=dict(color="brown", symbol="x")
            ),
            secondary_y=True, # Use secondary Y axis
        )

        # Add Target DIOH Horizontal Line
        # We use fig.update_layout shapes for horizontal lines when using make_subplots
        fig.add_hline(
            y=row_inv["TargetDIOH"],
            line_dash="dot",
            annotation_text=f"Target DIOH ({row_inv['TargetDIOH']} days)",
            annotation_position="top right",
            annotation_font_color="red",
            line_color="red",
            secondary_y=True,
        )


        # Highlight outliers (using scatter plots with larger markers)
        current_outliers_x = [week_labels[i] for i, flag in enumerate(flags_current) if flag]
        current_outliers_y = [dioh_current[i] for i, flag in enumerate(flags_current) if flag]
        if current_outliers_x:
            fig.add_trace(
                go.Scatter(
                    x=current_outliers_x,
                    y=current_outliers_y,
                    mode='markers',
                    name='Current Outlier',
                    marker=dict(color='red', size=12, symbol='circle-open'),
                    showlegend=True
                ),
                secondary_y=True
            )

        post_outliers_x = [week_labels[i] for i, flag in enumerate(flags_post) if flag]
        post_outliers_y = [dioh_post_order[i] for i, flag in enumerate(flags_post) if flag]
        if post_outliers_x:
            fig.add_trace(
                go.Scatter(
                    x=post_outliers_x,
                    y=post_outliers_y,
                    mode='markers',
                    name='Post Order Outlier',
                    marker=dict(color='orange', size=12, symbol='circle-open'),
                    showlegend=True
                ),
                secondary_y=True
            )


        # Update layout for aesthetics and labels
        fig.update_layout(
            title_text=f"ZB: {row_inv['ZB NAME']} | Material: {row_inv['MaterialNumberHarmonized']}",
            xaxis_title="Week",
            barmode="group",  # Groups the bars together
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=50, t=100, b=50),
            hovermode="x unified",
        )

        # Set axis titles using update_yaxes and secondary_y parameter
        fig.update_yaxes(title_text="Inventory Units", secondary_y=False)
        fig.update_yaxes(title_text="DIOH (days)", secondary_y=True)


        # Render in Streamlit
        st.plotly_chart(fig)


        # --- Weekly Summary Table ---
        summary_rows = []
        for i in range(len(actual)):
            summary_rows.append({
                "Week": week_labels[i],
                "Demand": int(actual[i]),
                "Production Qty": int(production_qty[i]),
                "Shipment Qty": int(shipment_qty_adj[i]),
                "Ending Inventory": int(ending_inventory[i]),
                "New Inventory": int(new_inventory[i]),
                "DIOH Current": int(dioh_current[i]) if not np.isnan(dioh_current[i]) else np.nan,
                "DIOH Post Order": int(round(dioh_post_order[i])) if not np.isnan(dioh_post_order[i]) else np.nan,
                "Target DIOH": int(row_inv["TargetDIOH"]),
                "DIOH Gap": int(round(dioh_gap[i])) if not np.isnan(dioh_gap[i]) else np.nan,
                "DIOH Push Flag": int(dioh_push_flag[i])    ,
                "Flag Current": "Below Target" if flags_current[i] else "OK",
                "Flag Post Order": "Below Target" if flags_post[i] else "OK"
            })

        summary_df = pd.DataFrame(summary_rows)


        st.subheader("Weekly Inventory & DIOH Summary")

        # --- Flag filter ---
        flag_filter = st.selectbox("Filter by Flag", ["All", "Flag Current Below Target", "Flag Post Order Below Target"])
        if flag_filter == "Flag Current Below Target":
            summary_df = summary_df[summary_df["Flag Current"] == "Below Target"]
        elif flag_filter == "Flag Post Order Below Target":
            summary_df = summary_df[summary_df["Flag Post Order"] == "Below Target"]

        # --- Highlight highest/lowest DIOH values ---
        def highlight_extremes(s):
            if s.name in ["DIOH Current", "DIOH Post Order"]:
                return ["background-color: lightgreen" if v == s.max() else
                        "background-color: lightcoral" if v == s.min() else "" for v in s]
            return [""] * len(s)

        styled_df = summary_df.style.apply(highlight_extremes, axis=0)

        st.dataframe(styled_df, use_container_width=True)

    # --- Tab 3: Raw Data ---
    with tab3:
        st.subheader("Raw Data Summary")

        # --- Meta and week columns ---
        meta_cols = ["Key","ADI","SeasonalIndex","CV2","Demand type","ZB NAME","MaterialNumberHarmonized"]
        week_cols = [c for c in data.columns if c not in meta_cols]

        raw_summary = []
        for _, row in data.iterrows():
            actual_series = row[week_cols].values.astype(float)
            mean_qty = actual_series.mean()
            std_qty = actual_series.std()

            raw_summary.append({
                "ZB Name": row["ZB NAME"],
                "Material": row["MaterialNumberHarmonized"],
                "Mean Order Qty": round(mean_qty, 2),
                "Standard Deviation of Order Qty": round(std_qty, 2),
                "Demand type": row["Demand type"],
                "ADI": round(row["ADI"], 2),
                "SeasonalIndex": round(row["SeasonalIndex"], 2),
                "CV2": round(row["CV2"], 2)
            })

        raw_df = pd.DataFrame(raw_summary)

        # --- Display table ---
        st.dataframe(raw_df)