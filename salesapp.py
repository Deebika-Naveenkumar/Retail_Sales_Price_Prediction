import streamlit as st 
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
import pickle
import tensorflow
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU

# streamlit page
st.set_page_config(layout="wide", page_title= "ANN Predictive Modeling of Retail Sales Price Analysis")

# Option menu in Streamlit
with st.sidebar:
    selected = option_menu(
        menu_title = "Menu",
        options = ["Home","Analysis","Insight","Sales Price Prediction"],
        icons = ["house-door","reception-4","globe2","patch-question"],
        menu_icon = "emoji-smile",
        default_index = 0
    )

# Home menu of Streamlit
if selected == "Home":
    st.title(":blue[ANN PREDICTIVE MODELING OF RETAIL SALES]")

    st.subheader(":blue[Overview :]")
    st.write("""
    - This project aims to develop a predictive ANN model to forecast department-wide sales for each store over the next year
    and analyze the impact of markdowns on sales during holiday weeks.
    - This prediction model will be based on past sales data of 45 retail stores (February 2010 to October 2012) and
    its goal is to provide recommendations to optimize markdown strategies and inventory management.
    - Sale prices are influenced by a wide variety of criteria, including holiday/non-holiday weeks, markdown offered, store type,
    size , departments and few other factors.
    """)

    st.subheader(":blue[Skills Takeaway from Project:]")
    st.write("""
             - Time Series Analysis
- Feature Engineering
- Predictive Modeling
- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Deep Learning Algorithms
- AWS Deployment
- Model Evaluation and Validation
- Data Visualization
- Tensorflow
- Python Programming
                 """)
    
    st.subheader(":blue[Domain :]")
    st.write("Retail Analytics")

# Load the data
data_path = "D:\Jupyter\Retail\df.csv"
df = pd.read_csv(data_path)

# Functions for each analysis plot
def display_sales_summary(df):
    # Calculate and display average sales for holiday weeks with/without markdowns
    df_holiday = df[df["IsHoliday"] == True]
    markdown_holiday = df_holiday[df_holiday["Markdown"] != 0]
    no_markdown_holiday = df_holiday[df_holiday["Markdown"] == 0]
    
    avg_sales_markdown_holiday = markdown_holiday["Weekly_Sales"].mean()
    avg_sales_no_markdown_holiday = no_markdown_holiday["Weekly_Sales"].mean()
    
    st.write(f"Avg sale price of Holiday weeks with markdowns: ${avg_sales_markdown_holiday:.2f}")
    st.write(f"Avg sale price of Holiday weeks without markdowns: ${avg_sales_no_markdown_holiday:.2f}")

def plot_scatter_markdown_sales(df, is_holiday=True):
    filtered_data = df[(df["IsHoliday"] == is_holiday) & (df["Markdown"] != 0)]
    title = 'Holiday' if is_holiday else 'Non-Holiday'
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_data, x='Markdown', y='Weekly_Sales', ax=ax)
    ax.set_title(f'Impact of Markdowns on Sales during {title} Weeks')
    st.pyplot(fig)

def plot_festive_sales(df):
    df_festive = df[(df["Month"] > 10) | (df["Month"] == 1)]
    df_festive_with_md = df_festive[df_festive["Markdown"] != 0]
    df_festive_without_md = df_festive[df_festive["Markdown"] == 0]
    
    df1 = df_festive_with_md.groupby("Month")["Weekly_Sales"].mean().reset_index()
    df1["Markdown"] = "Yes"
    df2 = df_festive_without_md.groupby("Month")["Weekly_Sales"].mean().reset_index()
    df2["Markdown"] = "No"
    df3 = pd.concat([df1, df2], axis=0)
    
    fig = px.bar(df3, x="Month", y="Weekly_Sales", color="Markdown", barmode="group")
    fig.update_layout(title="Markdown Impact on Festive Sales", title_x=0.5)
    st.plotly_chart(fig)

def plot_monthly_sales(df):
    df_month = df[["Year", "Month", "Weekly_Sales"]]
    df_month_sale_analysis = df_month.groupby(["Year", "Month"])["Weekly_Sales"].mean().reset_index()
    
    fig = px.line(df_month_sale_analysis, x="Month", y="Weekly_Sales", color="Year")
    fig.update_layout(title="Monthly Sales Analysis by Year", title_x=0.5)
    st.plotly_chart(fig)

def plot_avg_sales_by_type(df):
    df_type = df.groupby("Type")["Weekly_Sales"].mean().reset_index()
    
    fig = px.bar(df_type, x="Type", y="Weekly_Sales")
    fig.update_layout(title="Average Weekly Sales by Store Type", title_x=0.5)
    st.plotly_chart(fig)

def plot_line_features_by_type(df):
    y_vars = ["Size", "Temperature", "Unemployment"]
    fig, axes = plt.subplots(1, 3, figsize=(10,6))
    for i, y_var in enumerate(y_vars):
        sns.lineplot(data=df, x="Type", y=y_var, ax=axes[i])
        axes[i].set_title(f'{y_var} by Store Type')
    st.pyplot(fig)

def plot_department_sales(df):
    df_dept = df.groupby("Dept")["Weekly_Sales"].mean().reset_index().sort_values(by="Weekly_Sales", ascending=False)
    df_dept = df_dept.head(10)
    plt.figure(figsize=(10,6))
    sns.barplot(data=df_dept, x="Dept", y="Weekly_Sales", order=df_dept["Dept"])
    plt.title("Department Sales Analysis")
    st.pyplot(plt)

def plot_sales_by_size(df):
    df_size = df.groupby("Size")["Weekly_Sales"].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(10,6))
    sns.lineplot(data=df_size, x="Size", y="Weekly_Sales", ax=ax)
    ax.set_title("Average Weekly Sales by Store Size")
    st.pyplot(fig)

# Analysis section
if selected == "Analysis":
    st.title("Sales Data Analysis")
    display_sales_summary(df)
    col1, col2 = st.columns(2)

    with col1:
        plot_scatter_markdown_sales(df, is_holiday=True)
        st.write("")
        st.write("")
        plot_festive_sales(df)
        st.write("")
        st.write("")
        plot_monthly_sales(df)
        plot_department_sales(df)
       
    with col2:
        plot_scatter_markdown_sales(df, is_holiday=False)
        st.write("")
        st.write("")
        plot_avg_sales_by_type(df)
        st.write("")
        st.write("")
        plot_sales_by_size(df)
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        plot_line_features_by_type(df)

if selected == "Insight":
    # Define the layout with columns for circular cards
    col1, col2 = st.columns(2)

    # Impact of Markdown on Sales (Holiday vs Non-Holiday)
    with col1:
        st.subheader("Impact of Markdown on Sales")
        fig_holiday_md = go.Figure(go.Indicator(
            mode="gauge+number",
            value=17594,  # Average sales during holiday weeks with markdowns
            title={"text": "Holiday Markdown"},
            gauge={'axis': {'range': [15000, 20000]}, 'bar': {'color': "green"}},
            domain={'x': [0, 1], 'y': [0, 1]}
        ))
        
        fig_holiday_no_md = go.Figure(go.Indicator(
            mode="gauge+number",
            value=16656,  # Average sales during holiday weeks without markdowns
            title={"text": "Holiday Without Markdown"},
            gauge={'axis': {'range': [15000, 20000]}, 'bar': {'color': "blue"}},
            domain={'x': [0, 1], 'y': [0, 1]}
        ))
        
        st.plotly_chart(fig_holiday_md, use_container_width=True)
    
    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.plotly_chart(fig_holiday_no_md, use_container_width=True)

    # Monthly Sales Trends
    st.subheader("Monthly Sales Trends")
    monthly_sales_data = {
        'January': 14126, 'February': 16008, 'March': 15416, 'April': 15650, 'May': 15776,
        'June': 16326, 'July': 15861, 'August': 16062, 'September': 15095,
        'October': 15243, 'November': 17491, 'December': 19355
    }
    fig_trends = go.Figure(go.Bar(
        x=list(monthly_sales_data.keys()),
        y=list(monthly_sales_data.values()),
        marker_color=['#E3F2FD' if month not in ['November', 'December'] else '#1E88E5' for month in monthly_sales_data.keys()]
    ))
    fig_trends.update_layout(title="Sales by Month (Peak in Festive Months)")
    st.plotly_chart(fig_trends, use_container_width=True)

    # Store Type Analysis
    st.subheader("Store Type Analysis")
    # Define circle properties
    circle_data = [
    {"title": "Store Size", "text": "A > B > C", "color": "purple"},
    {"title": "Number of Stores", "text": "22 in A, 17 in B, 6 in C", "color": "purple"},
    {"title": "Temperature", "text": "C > A > B", "color": "purple"},
    {"title": "Unemployment", "text": "C > B > A", "color": "purple"},
]

    # Layout with columns for circles
    col3, col4 = st.columns(2)

    # Function to create a circle figure
    def create_circle(title, text, color):
        fig = go.Figure()

        # Add a circle shape
        fig.add_shape(
            type="circle",
            x0=0, y0=0, x1=25, y1=25,
            xref="x", yref="y",
            fillcolor=color,
            line_color="blue",
        )

        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(range=[0, 30])
        fig.update_yaxes(range=[0, 30])

        
        # Add the title text inside the circle
        fig.add_annotation(
            x=10, y=28,
            text=f"<b>{title}</b>",
            showarrow=False,
            font=dict(size=26),
            xref="x", yref="y"
        )
        
        # Add the main text inside the circle
        fig.add_annotation(
            x=12, y=14,
            text=text,
            showarrow=False,
            font=dict(size=16),
            xref="x", yref="y"
        )

        # Customize layout to center the circle and text
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            width=200, height=200,
            margin=dict(l=0, r=0, t=0, b=0)
        )

        return fig

    # Display circles in columns
    with col3:
        st.write("")
        st.write("")
        st.plotly_chart(create_circle(circle_data[0]["title"], circle_data[0]["text"], circle_data[0]["color"]), use_container_width=True)
        st.write("")
        st.write("")
        st.plotly_chart(create_circle(circle_data[1]["title"], circle_data[1]["text"], circle_data[1]["color"]), use_container_width=True)

    with col4:
        st.write("")
        st.write("")
        st.plotly_chart(create_circle(circle_data[2]["title"], circle_data[2]["text"], circle_data[2]["color"]), use_container_width=True)
        st.write("")
        st.write("")
        st.plotly_chart(create_circle(circle_data[3]["title"], circle_data[3]["text"], circle_data[3]["color"]), use_container_width=True)

# Sales Price Prediction Menu:
if selected == "Sales Price Prediction":
    st.markdown("<h4 style=color:#fbe337>Enter the following details:",unsafe_allow_html=True)
    st.write('')

    with st.form('price_prediction'):
        holiday_value = ["Yes","No"]
        holiday_value_map = {"Yes":1,"No":0}
        Type = ["A","B","C"]
        Type_map = {"A":1,"B":2,"C":3}
        col1, spacer, col2 = st.columns([4, 1, 4])
        # ['Store', 'Dept', 'IsHoliday', 'Type', 'Size',
    #    'Temperature', 'CPI', 'Unemployment', 'Markdown', 'Year', 'Month',
    #    'Day', 'Weekly_Sales_Lag1', 'Weekly_Sales_Lag2',
    #    'Markdown_Holiday_Only', 'Markdown_NonHoliday_Only']

        with col1:
            Store = st.number_input('**Store**',min_value=1, max_value=45)
            Department = st.number_input('**Department**',min_value=1, max_value=99)
            IsHoliday = st.selectbox('**IsHoliday**',holiday_value)
            Type = st.selectbox('**Type**',Type)
            Size = st.selectbox('**Size**',sorted(df['Size'].unique()))
            Temperature = st.number_input('**Temperature**',min_value=-2.0, max_value=101.0)
            CPI = st.number_input('**CPI**',min_value=126.0, max_value=227.0)
            Unemployment = st.number_input('**Unemployment**',min_value=0.0, max_value=15.0)
           
        with col2:
            MarkDown = st.number_input('**Markdown**',min_value=1.0, max_value=160510.0)
            Year = st.number_input('**Year**',min_value=2010, max_value=2024)
            Month = st.number_input('**Month**',min_value=1, max_value=12)
            Day = st.number_input('**Day**',min_value=1, max_value=31)
            Weekly_Sales_Lag1 = st.number_input('**Weekly_Sales_Lag1**',min_value=0.01, max_value=693099.36)
            Weekly_Sales_Lag2 = st.number_input('**Weekly_Sales_Lag2**',min_value=0.01, max_value=693099.36)
            MarkDown_holiday = st.number_input('**MarkDown_Holiday_Only**',min_value=1.0, max_value=160510.0)
            MarkDown_nonholiday = st.number_input('**MarkDown_NonHoliday_Only**',min_value=1.0, max_value=160510.0)
            button = st.form_submit_button(':green[**Predict Sales Price**]',use_container_width=True)

    if button:
        if not all([Store, Department, IsHoliday, Type, Size, Temperature, CPI, Unemployment, MarkDown, Year, Month,
                     Day, Weekly_Sales_Lag1, Weekly_Sales_Lag2, MarkDown_holiday, MarkDown_nonholiday]):
            st.error("Kindly fill the blanks")
        else:
            Holiday_value_encoded = holiday_value_map[IsHoliday]
            Type_encoded = Type_map[Type]
            Unemployment_input = boxcox(Unemployment, lmbda=-0.06995164335275873)
            Markdown_input = boxcox(MarkDown, lmbda=-0.19278533614859414)
            weekly_sales_lag1_input = boxcox(Weekly_Sales_Lag1, lmbda=0.21190295673149415)
            weekly_sales_lag2_input = boxcox(Weekly_Sales_Lag2, lmbda=0.21190295673149415)
            MarkDown_Holiday_Only_input = boxcox(MarkDown_holiday, lmbda=-0.19278533614859414)
            MarkDown_NonHoliday_Only_input = boxcox(MarkDown_nonholiday, lmbda=-0.19278533614859414)

        # Load the scaler and model
        model_loaded = load_model("model.h5", custom_objects={'LeakyReLU': LeakyReLU})
        with open("scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
            
        # user_input
        user_input = np.array([[Store, Department, Holiday_value_encoded, Type_encoded, Size, Temperature, CPI, 
                    Unemployment_input, Markdown_input, Year, Month, Day, weekly_sales_lag1_input, weekly_sales_lag2_input, 
                    MarkDown_Holiday_Only_input, MarkDown_NonHoliday_Only_input]])

        # Scale the input values
        scaled_input = scaler.transform(user_input)
        # Make predictions using the trained model
        predict = model_loaded.predict(scaled_input)
        predicted_sales_original = inv_boxcox(predict, 0.212)
        st.markdown(f'<h4 style=color:#fbe337>The Predicted Retail Sales Price is: {predicted_sales_original[0][0]-1:.2f} </h4>',unsafe_allow_html=True)


        






           
            
            
    

       
        

    
   
   