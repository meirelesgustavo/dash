#IMPORTING LIBRARIES
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import pickle
from sklearn import  metrics
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import holidays
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

#Importing Figures
from Dashboard import fig5 #Filter method graph
from Dashboard import fig6 #Wrapper method 3 features graph
from Dashboard import fig7 #Wrapper method 2 features graph
from Dashboard import fig8 #Wrapper method 1 features graph
from Dashboard import fig9 #Embedded method
from plotly.graph_objects import Figure

#Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Load data
df_Raw_Data2019 = pd.read_csv("testData_2019_NorthTower.csv")
df_Raw_Data2019['Date'] = pd.to_datetime (df_Raw_Data2019['Date'])
df_Raw_Data2019.set_index("Date", inplace=True)
df_Cleaned_Data = pd.read_csv("CleanedData.csv")
df_Cleaned_Data['Date'] = pd.to_datetime (df_Cleaned_Data['Date']) # create a new column 'data time' of datetime type
df_Cleaned_Data.set_index("Date", inplace=True)

fig1 = px.line(df_Raw_Data2019, x=df_Raw_Data2019.index, y=df_Raw_Data2019.columns[1:10])# Creates a figure with the raw data

#Adding features to the 2019 data
# Create a list of Portuguese holidays
portugal_holidays = holidays.Portugal(years=df_Raw_Data2019.index.year.unique())

# Create a new column "Holiday" (1 if date is a holiday, 0 otherwise)
df_Raw_Data2019["Holiday"] = df_Raw_Data2019.index.map(lambda x: 1 if x in portugal_holidays else 0)

# Extract hour directly from the index
df_Raw_Data2019["Hour"] = df_Raw_Data2019.index.hour

# Shift the "Power [kW]" column by 1 hour to create the "Power - 1" column
df_Raw_Data2019["Power - 1"] = df_Raw_Data2019["North Tower (kWh)"].shift(1)

# Create the "Sin Hour" column using sine transformation
amp=1/24
time=df_Raw_Data2019["Hour"].values
df_Raw_Data2019["Sin hour"] = 10 * np.sin(2 * np.pi * amp * time -8)

# Renaming the columns
df_Raw_Data2019.rename(columns={
    "temp_C": "Temp (°C)",
    "windSpeed_m/s": "Wind Speed (m/s)",
    "windGust_m/s": "Wind Gust (m/s)",
    "pres_mbar": "Pres (mbar)",
    "solarRad_W/m2": "Solar Irradiance (W/m²)",
    "rain_mm/h": "Rain (mm/h)",
    "rain_day": "Rain (Day)"
}, inplace=True)


#SETTING UP THE FORECASTING COMPARISON
df_PredictedData = pd.read_csv("Predictions_2019.csv")
df_PredictedData['Date'] = pd.to_datetime (df_PredictedData['Date']) # create a new column 'data time' of datetime type
df_PredictedData.set_index("Date", inplace=True)

#Loading the Random Forest Model
with open('RF_model.pkl', 'rb') as file:
    RF_model = pickle.load(file)

df2=df_PredictedData.iloc[:,1:5]
X2=df2.values
y2=df_PredictedData["North Tower (kWh)"]
y2_pred_RF = RF_model.predict(X2)

#Loading the Decision Tree Model
with open('DT_regr_model.pkl', 'rb') as file:
    DT_regr_model = pickle.load(file)

y3_pred_DT = DT_regr_model.predict(X2)

#Doing the comparison between Predicted x Actual power
# Create a DataFrame for comparison
comparison_df = pd.DataFrame({
    "Date": df_Raw_Data2019.index,  # Ensure both datasets align by Date
    "Actual Power (kWh)": df_Raw_Data2019["North Tower (kWh)"],  # From Actual Data CSV
    "Predicted Power RF Model (kWh)": y2_pred_RF,  #RF Model's predicted output
    "Predicted Power DT Model (kWh)": y3_pred_DT #DT Model´s predicted output
})

comparison_df.info()

fig2 = px.line(
    comparison_df,
    x=comparison_df.index,
    y=["Actual Power (kWh)",'Predicted Power RF Model (kWh)',"Predicted Power DT Model (kWh)"],
    title="Actual vs Predicted Power Comparison",
    labels={"Power (kWh)": "Power (kWh)", "Date": "Date"},
    template="plotly_white"
)

#Evaluate errors
# Error metrics for RF model
metrics_rf = {
    'Model': 'Random Forest',
    'MAE': metrics.mean_absolute_error(y2, y2_pred_RF),
    'MBE': np.mean(y2 - y2_pred_RF),
    'MSE': metrics.mean_squared_error(y2, y2_pred_RF),
    'RMSE': np.sqrt(metrics.mean_squared_error(y2, y2_pred_RF)),
    'cvRMSE': np.sqrt(metrics.mean_squared_error(y2, y2_pred_RF)) / np.mean(y2),
    'NMBE': np.mean(y2 - y2_pred_RF) / np.mean(y2)
}

# Error metrics for DT model
metrics_dt = {
    'Model': 'Decision Tree',
    'MAE': metrics.mean_absolute_error(y2, y3_pred_DT),
    'MBE': np.mean(y2 - y3_pred_DT),
    'MSE': metrics.mean_squared_error(y2, y3_pred_DT),
    'RMSE': np.sqrt(metrics.mean_squared_error(y2, y3_pred_DT)),
    'cvRMSE': np.sqrt(metrics.mean_squared_error(y2, y3_pred_DT)) / np.mean(y2),
    'NMBE': np.mean(y2 - y3_pred_DT) / np.mean(y2)
}

# Combine both into one DataFrame
#df_metrics = pd.DataFrame([metrics_rf, metrics_dt])

#Function to generate tables
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(html.Tr([html.Th(col) for col in dataframe.columns])),
        html.Tbody([
            html.Tr([html.Td(dataframe.iloc[i][col]) for col in dataframe.columns]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


# CREATING THE APP
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1('IST North Tower Energy Forecast Tool (kWh)'),
    html.P('Representing North Tower Data and Forecasting for November 2019'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Feature Analysis', value='tab-2'),
        #dcc.Tab(label='Forecasting Comparison', value='tab-3'),
        dcc.Tab(label='Train your Model', value='tab-3'),
        dcc.Tab(label="Error Metrics",value="tab-4"),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H4('IST Raw Data 2019'),
            html.Div([
                dcc.Checklist(
                    id='variable-selector',
                    options=[{'label': col, 'value': col} for col in df_Raw_Data2019.columns],
                    value=df_Raw_Data2019.columns[:3].tolist(),
                    labelStyle={'display': 'inline-block', 'margin-right': '15px'},
                    inputStyle={'margin-right': '5px'}
                )
            ], style={'maxHeight': '120px', 'overflowY': 'scroll', 'margin-bottom': '20px'}),

            dcc.Graph(id='yearly-data', figure=fig1)
        ])

    elif tab == 'tab-2':
        return html.Div([
            html.H4('Feature Analysis'),
            html.P("Choose the method you want to analyze:", style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='feature-method-dropdown',
                options=[
                    {'label': 'Filter Method', 'value': 'linear'},
                    {'label': 'Wrapper Method (3 features)', 'value': 'wrap3'},
                    {'label': 'Wrapper Method (2 features)', 'value': 'wrap2'},
                    {'label': 'Wrapper Method (1 feature)', 'value': 'wrap1'},
                    {'label': 'Embedded Method', 'value': 'embed'}
                ],
                value='linear',
                clearable=False,
                style={'width': '80%', 'margin-bottom': '20px'}
            ),
            dcc.Graph(id='feature-graph')
        ])



    elif tab == 'tab-3':
        return html.Div([
            html.H4("Train Model with Selected Features"),
            html.Label(
                "All the models available will be trained with your features selection and you will be able to see the error metrics in the next tab."),
            # Model Selection Dropdown
            html.Label("Select the model you want to visualize:"),
            dcc.Dropdown(
                id='model-selection',
                options=[
                    {'label': 'Random Forest', 'value': 'Random Forest'},
                    {'label': 'Decision Tree', 'value': 'Decision Tree'},
                    {'label': 'Linear Regression', 'value': 'Linear Regression'}
                ],
                value=None,
                placeholder="Select",
                style={"width": "50%"}
            ),

            # Feature Selection Dropdown (BEFORE comparison menu)
            html.Label("Select features to use in the model:"),
            dcc.Dropdown(
                id='feature-selector',
                options=[{'label': col, 'value': col} for col in df_Cleaned_Data.columns if col != "Power [kW]"],
                value=[],  # Nothing selected initially
                multi=True,
                placeholder="Select",  # Placeholder text saying "Select"
                style={"width": "80%"}
            ),

            # Model Comparison Dropdown (AFTER feature selection)
            html.Label("Compare with another model? (Optional)"),
            dcc.Dropdown(
                id='compare-model-selection',
                options=[
                    {'label': 'Random Forest', 'value': 'Random Forest'},
                    {'label': 'Decision Tree', 'value': 'Decision Tree'},
                    {'label': 'Linear Regression', 'value': 'Linear Regression'}
                ],
                value=[],
                multi=True,
                placeholder="Select",
                style={"width": "50%"}
            ),

            # Train Button
            html.Button("Train Model", id='train-model-btn', n_clicks=0, style={"margin-top": "10px"}),
            # Output Graph
            dcc.Graph(id='output-graph'),
            html.Div(id='train-error-output')
        ])


    elif tab == 'tab-4':
        return html.Div([
            html.H4('Error Metrics'),
            html.Div([
                html.Div([
                    html.Label("Choose the model(s) you want to analyze:"),
                    dcc.Dropdown(
                        id='model-dropdown',
                        options=[{'label': model, 'value': model} for model in df_metrics['Model']],
                        value=[],
                        placeholder="Select model",
                        multi = True
                    )
                ], style={'display': 'inline-block', 'width': '45%', 'padding-right': '20px'}),
                html.Div([
                    html.Label("Choose the error metric(s):"),
                    dcc.Dropdown(
                        id='metric-dropdown',
                        options=[{'label': col, 'value': col} for col in df_metrics.columns if col != 'Model'],
                        value=[],
                        placeholder="Select metric(s)",
                        multi=True
                    )
                ], style={'display': 'inline-block', 'width': '30%', 'padding-right': '20px'}),
            ], style={'margin-bottom': '30px'}),
            html.Div(id='error-output')
        ])


# Callback to update graph in tab-1
@app.callback(
    Output('yearly-data', 'figure'),
    Input('variable-selector', 'value')
)
def update_raw_graph(selected_vars):
    if not selected_vars:
        return px.line(title="Please select at least one variable.")

    fig = px.line(
        df_Raw_Data2019,
        x=df_Raw_Data2019.index,
        y=selected_vars,
        labels={"value": "Value", "Date": "Date"},
        template="plotly_white"
    )

    return fig

# Callback to update graph in tab-2
@app.callback(
    Output('feature-graph', 'figure'),
    Input('feature-method-dropdown', 'value')
)
def update_feature_graph(method):
    if method == 'linear':
        fig = fig5
    elif method == 'wrap3':
        fig = fig6
    elif method == 'wrap2':
        fig = fig7
    elif method == 'wrap1':
        fig = fig8
    elif method == 'embed':
        fig = fig9
    else:
        fig = go.Figure()  # Return an empty figure if no method is selected

    return fig


# Callback to update graph in tab-4
from dash import dash_table
@app.callback(
    Output('error-output', 'children'),
    [Input('model-dropdown', 'value'),
     Input('metric-dropdown', 'value')]
)
def display_metrics_table(models, selected_metrics):
    if not models:
        return html.P("Please select at least one model.", style={'color': 'red'})

    if not selected_metrics:
        return html.P("Please select at least one metric.", style={'color': 'red'})

    metrics_data = []
    column_names = ['Metric']  # First column is the metric name

    for model in models:
        if model not in trained_models or "metrics" not in trained_models[model]:
            return html.P(f"Please train the model '{model}' first.", style={'color': 'red'})
        column_names.append(model)  # Add model name to the columns

    # Gather metrics for each selected model
    for metric_name in selected_metrics:
        row = {'Metric': metric_name}

        for model in models:
            if metric_name in trained_models[model]["metrics"]:
                row[model] = f"{trained_models[model]['metrics'][metric_name]:.4f}"
            else:
                row[model] = "N/A"  # If the metric is not available for this model

        metrics_data.append(row)

    # Render table
    return dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in column_names],
        data=metrics_data,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center', 'padding': '5px'},
        style_header={'backgroundColor': '#f1f1f1', 'fontWeight': 'bold'},
        style_as_list_view=True
    )

# Callback to update graph in tab-3
# Store trained models' predictions and metrics
trained_models = {}
df_metrics = pd.DataFrame(columns=['Model', 'MAE', 'MBE', 'MSE', 'RMSE', 'cvRMSE', 'NMBE'])

@app.callback(
    [Output("output-graph", "figure"),
     Output("train-error-output", "children")],
    [Input("train-model-btn", "n_clicks")],
    [State("model-selection", "value"),
     State("feature-selector", "value"),
     State("compare-model-selection", "value")]
)
def train_model(n_clicks, selected_model, selected_features, compare_models):
    global df_metrics  # Declare df_metrics as global to modify it

    if n_clicks == 0 or not selected_features or not selected_model:
        return go.Figure(), html.P("Please select a model and features before training.", style={'color': 'red'})

    if len(selected_features) > 4:
        return go.Figure(), html.P("You can select a maximum of 4 features.", style={'color': 'red'})

    # Prepare training and testing data
    X_train = df_Cleaned_Data[selected_features].dropna()  # Remove NaN values
    y_train = df_Cleaned_Data.loc[X_train.index, "Power [kW]"]  # Ensure y_train matches the cleaned X_train

    X_test = df_Raw_Data2019[selected_features].dropna()  # Remove NaN values
    y_test = df_Raw_Data2019.loc[X_test.index, "North Tower (kWh)"]
    dates = X_test.index

    # Define a function to train models and store metrics
    def train_and_store_model(model_name, model_obj):
        global df_metrics  # Declare df_metrics as global to modify it

        model_obj.fit(X_train, y_train)
        y_pred = model_obj.predict(X_test)

        # Calculate metrics
        metrics_dict = {
            'Model': model_name,
            'MAE': mean_absolute_error(y_test, y_pred),
            'MBE': np.mean(y_test - y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'cvRMSE': np.sqrt(mean_squared_error(y_test, y_pred)) / np.mean(y_test),
            'NMBE': np.mean(y_test - y_pred) / np.mean(y_test)
        }

        # Store model and metrics
        trained_models[model_name] = {
            "model": model_obj,
            "features": selected_features,
            "y_test": y_test,
            "y_pred": y_pred,
            "metrics": metrics_dict
        }

        # Update df_metrics
        if model_name in df_metrics['Model'].values:
            for key, value in metrics_dict.items():
                df_metrics.loc[df_metrics['Model'] == model_name, key] = value
        else:
            df_metrics = pd.concat([df_metrics, pd.DataFrame([metrics_dict])], ignore_index=True)
        return y_pred

    # Train the selected model
    if selected_model == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    elif selected_model == "Decision Tree":
        model = DecisionTreeRegressor(random_state=42)
    elif selected_model == "Linear Regression":
        model = LinearRegression()
    else:
        return go.Figure(), html.P("Invalid model selection.", style={'color': 'red'})

    y_pred = train_and_store_model(selected_model, model)

    # Plot predictions and actual data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_test, name="Actual", mode="lines", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=dates, y=y_pred, name=f"Predicted ({selected_model})", mode="lines", line=dict(color='red')))

    # Compare with other models if selected
    if compare_models:
        if isinstance(compare_models, str):
            compare_models = [compare_models]

        for compare_model in compare_models:
            if compare_model not in trained_models:
                if compare_model == "Random Forest":
                    compare_model_obj = RandomForestRegressor(n_estimators=200, random_state=42)
                elif compare_model == "Decision Tree":
                    compare_model_obj = DecisionTreeRegressor(random_state=42)
                elif compare_model == "Linear Regression":
                    compare_model_obj = LinearRegression()
                else:
                    return fig, html.P(f"Invalid comparison model selection: {compare_model}", style={'color': 'red'})

                compare_y_pred = train_and_store_model(compare_model, compare_model_obj)
            else:
                compare_y_pred = trained_models[compare_model]["y_pred"]

            fig.add_trace(go.Scatter(
                x=dates,
                y=compare_y_pred,
                name=f"Predicted ({compare_model})",
                mode="lines",
                line=dict(dash='dash')
            ))

    fig.update_layout(
        title=f"{selected_model} Prediction vs Actual",
        xaxis_title="Date",
        yaxis_title="Power (kWh)",
        template="plotly_white"
    )

    return fig, html.P("Model training complete.", style={'color': 'green'})

if __name__ == '__main__':
    app.run(debug=True)
