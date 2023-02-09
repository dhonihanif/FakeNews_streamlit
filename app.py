import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle

def model(csv):
    # Load data
    df = pd.read_csv(csv, encoding="latin-1")
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    # Load model
    model = pickle.load(open("./fakenews/tree.sav", "rb"))
    
    # evaluation
    y_pred = model.predict(X_test)
    training_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_scores = f1_score(y_test, y_pred)

    return df, model, training_score, test_score, precision, recall, f1_scores

(
    df, model, training_score,
    test_score, precision, recall, f1_scores
) = model("./fakenews/FakeNews.csv")
df2 = pd.read_csv("./fakenews/FakeNewsNet.csv", encoding="latin-1")
df3 = pd.read_csv("./fakenews/FakeNews3.csv", encoding="latin-1")
df3.drop(df3.columns[0], axis=1, inplace=True)
df.drop(df.columns[0], axis=1, inplace=True)

option = st.sidebar.selectbox(
    "Silakan pilih:",
    ("Home","Dataframe", "Data Visualization", "Model Building", "Predict")
)

if option == "Home" or option == "":
    st.write("""# Fake News Project""") #menampilkan halaman utama
    st.write()
    st.markdown("**Fake News Project**")
    st.write("This website is about build project about Fake News Predictions")
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://dhonihanif.netlify.app/doni.jpeg", width=200)
    with col2:
        st.write(f"""
        Name : Dhoni Hanif Supriyadi\n
        Birth : 27 November 2001\n
        Degree : Bachelor degree start from 2020 until 2024\n
        Lasted GPA : 3.97 from 4.00\n
        University : University of Bina Sarana Informatika\n
        Field : Information System\n
        Linkedin : http://bit.ly/3x72z9F \n
        Github : https://github.com/dhonihanif \n
        Email : dhonihanif354@gmail.com \n
        Phone : +62895326168335
        """)

elif option == "Dataframe":
    st.write("""## Dataframe""") #menampilkan judul halaman dataframe
    st.write()
    st.markdown("**We read the data and do step of preparation data**")
    st.write(f"\nOriginal data with {df2.shape[0]} row and {df2.shape[1]} columns")
    st.write(df2)
    st.write(f"\nAfter cleaning and encode the category variables become {df.shape[0]} row and {df.shape[1]} columns")
    st.write(df)
    st.write(f"\nAfter normalization with standard scaler become {df3.shape[0]} row and {df3.shape[1]} columns")
    st.write(df3)

elif option == "Data Visualization":
    st.write("""## Data Visualization\n""")
    st.markdown("**Do some visualization or analysis about the data for get the insight of the data**\n")

    with st.expander("Graph real vs fake"):
        st.image("./images/no1.png")
        st.write("""
        As we can see, the data we get is out of balance. Real data totals
76.2% of the total data or around 17 thousand data while fake data amounts to 23.8%.
all data or 5 thousand data.
        """)
    
    with st.expander("Information of variables category"):
        st.image("./images/no2.png")
        st.write("""
        As we can see, the data title has a unique value of around 21K with the top or mode
is Connecting People Through News and the frequency is 20, the data news_url has
unique value of 21k data with top or mode is www.thewrap.com / and frequency
is 11, and data source_domain has a unique value of 2k data with its top or mode
is people.com and the frequency is 1,779.
        """)
    
    with st.expander("Correlation of the data"):
        st.image("./images/no3.png")
        st.write("""
        As we can see, there is data that is low correlated with other data and there is also
data that correlates highly with other data such as news_url correlates very well with
source_domain data. Then, there is also very low correlated data such as title data with
source_domain data.
        """)
    
    with st.expander("Univariate Analysis for categorical variables of Title"):
        st.image("./images/no4.png")
        st.write("""
        Because of much unique value of this variable, we make encode first for easy to read.
        As we can see, the title data has a pretty good spread and isn't there
outliers there. This is the original data object that I have encoded into numeric data.
        """)
    
    with st.expander("Univariate Analysis for categorical variables of the News URL"):
        st.image("./images/no5.png")
        st.write("""
        Because of much unique value of this variable, we make encode first for easy to read.
        As we can see, the distribution of the data is quite good and there are no outlier values. Same
like the previous data, it is a data object that I have encoded with a numeric.
        """)
    
    with st.expander("Univariate Analysis for categorical variables of Source Domain"):
        st.image("./images/no6.png")
        st.write("""
        Because of much unique value of this variable, we make encode first for easy to read.
        As we can see, the data source_domain has a poor distribution because of the data
the most has a fairly large reach with the least data.
However, on the other hand, this data can still be said to be good because there are no outlier data. Same
like the previous data, this data is object data that I have encoded into data
numeric.
        """)
    
    with st.expander("Univariate Analysis for numeric variables of Tweet Num"):
        st.image("./images/no7.png")
        st.write("""
        As we can see, there are a lot of outlier values ​​here. Even though this data is data
numeric and not my encode. However, this data contains a lot of outliers. Knowing things
this, I will check the amount of each data with value_counts() as follows.
        \n""")
        st.image("./images/no8.png")
        st.write("""
        As you can see, it turns out that the spread of the data is not good enough. Most of the data exists
in numbers around 0 – 10, the rest there is only 1 data so we can
say that the distribution of the data is not good enough and there are many outliers or
noise.
        """)
    
    with st.expander("Bivariate Analysis of variable real and variable tweet num"):
        st.image("./images/no9.png")
        st.write("""
        As we can see, fake data has a higher tweet_num value than real data.
        """)
    

elif option == "Model Building":
    st.write("""## Model Building\n""")
    st.markdown("**Build some models to get better predict**\n")
    st.write("We have made some models like Naive Bayes, Decision Tree, and Support Vector Machine. Then, we compare the evaluation of the models like below")
    st.image("./images/no10.png")
    st.write(f"""
    As we can see, the best model so far that we can use is the Decision Tree and
Support Vector Machine. With only 5 data dimensions, it can be suggested to
using Decision Trees. Decision Tree is an interesting model to study because
The tree of the decision tree contains only ifs and else which we can understand easily because
of this model as follows.\n
    The performance of the model Decision Tree is:
    Training Score : {training_score}\n
    Test Score : {test_score}\n
    Precision Score : {precision}\n
    Recall Score : {recall}\n
    f1 Score : {f1_scores}\n
    Then, the confusion matrix like below:
    """)
    st.image("./images/no11.png")
    st.write("Some explanation like if else tree is below")
    st.image("./images/no12.png")

elif option == "Predict":
    st.write("""## Predict\n""")
    st.markdown("**Predict the value of input**\n")
    results = []
    le = LabelEncoder()
    
    for i in df2.columns[:-1]:
        if i == "tweet_num":
            inputt = st.number_input(i)
        else:
            inputt = st.text_input(i)
        
        results.append(inputt)
    
    a = st.button("Predict")
    if a:
        s = df2.columns.tolist()
        for i in results:
            if type(i) == str:
                le.fit(df2[df2.columns[results.index(i)]])
                results[results.index(i)] = le.transform([i])[0]
        scaler = StandardScaler().fit(df.iloc[:, :-1])
        results = scaler.transform(np.array(results).reshape(1, -1))
        target = model.predict(results)
        st.write(target)
