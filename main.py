import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
import pickle # to load a saved model
import csv # to create file for storing user inputs


# load postive-word list and negative-words list
positive_words = []
with open('positive-words.txt', 'r') as f:
    for line in f:
        positive_words.append(line.strip())

negative_words = []
with open('negative-words.txt', encoding = "ISO-8859-1") as f:
    for line in f:
        negative_words.append(line.strip())

app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Feedback']) #two pages

if app_mode == 'Home':
    st.header('Welcome to "Review Classifier" Web App!')
    st.subheader('This app allows you to predict whether a review is "Positive" or "Negative".')
    st.image('review_pic.PNG')
    st.subheader('Please write a review below: ')
    review = st.text_input("Review: ")

    if review:
        # convert review to array to be able to insert in in the model
        review_list = nltk.word_tokenize(review.lower())
        review_length = np.log(len(review_list))
        if 'no' in review_list:
            no_count = 1
        else:
            no_count = 0

        if '!' in review_list:
            exclamation = 1
        else:
            exclamation = 0
        i = 0 # for counting # positive word in the review
        j = 0 # for counting # negative word in the review
        k = 0 # for counting 1st and 2nd pronoun in the review
        for word in review_list:
            if word in positive_words:
                i += 1
            if word in negative_words:
                j += 1
            if word in ['i', 'me', 'my', 'you', 'your']:
                k +=1
        review_num = [i, j, no_count, k, exclamation, review_length]
        review_array = np.array(review_num).reshape(1, -1)
        
        # load the saved model and predict the class of the review
        loaded_model = pickle.load(open('Review_classifier.sav', 'rb'))
        prediction = loaded_model.predict(review_array)[0]

        if prediction == 1:
            predict = "Positive"
        else:
            predict = "Negative"

        st.subheader(f'The review is "{predict}".')

    st.subheader(f'Your feedback is cruicial for us!')
    #feedback = st.radio('The above prediction is: ', ['Correct', 'Wrong'])

    options = ['Correct', 'Wrong']
    # Use a custom placeholder option and add it to the list of options
    placeholder = "Select an option"
    options_with_placeholder = [placeholder] + options
    # Use st.selectbox instead of st.radio
    feedback = st.selectbox("The above prediction is: ", options_with_placeholder, index=0)

    if feedback != placeholder:
        # store user data in csv file
        with open("user_data.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([review, predict, feedback])

if app_mode == 'Feedback':
    st.header('This page reports the feedback from users.')
    review_df = pd.read_csv('user_data.csv', header=None)
    df = review_df.rename(columns={0: 'Review', 1: 'Prediction', 2: 'Feedback'})
    st.write(df)

    count = df['Feedback'].value_counts()

    #st.bar_chart(count)

    fig1, ax1 = plt.subplots()
    ax1.pie(count, labels=['Correct', 'Wrong'], autopct='%1.1f%%')
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)
