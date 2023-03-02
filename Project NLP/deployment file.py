import streamlit as st
import pandas as pd
import datetime as dt
from PIL import Image

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

st.set_page_config("Youtube comment analysis", layout="centered")
st.title("Youtube comments analysis app")

st.info(""" 

**This app** gives the sentimental analysis for a youtube video comments 

""")

st.info(""" 

-----  **PLEASE TURN ON INTERNET CONNECTION** to run the app  ----- 

""")


def time_line_extractor(f):
    import pandas as pd
    import datetime as dt

    f = pd.read_csv("translated_comments.csv")
    f = f.drop(columns=["extracted_comments"], axis=1)
    # df.rename(columns={"translated comments": "translated_comments"}, inplace=True)
    # dt = []
    # for i in df["comments_date"]:
    #     date = i[0:10]
    #     # time = i[-9:-1]
    #     # dt.append(" ".join([date, time]))
    #     dt.append(date)

    # df["dt"] = dt
    # df = df.drop(["comments_date"], axis=1)
    f["comments_date"] = pd.to_datetime(f["comments_date"])
    f["year"] = f["comments_date"].dt.year
    f["month"] = f["comments_date"].dt.month_name()
    f["day"] = f["comments_date"].dt.day
    f["hour"] = f["comments_date"].dt.hour
    f["min"] = f["comments_date"].dt.minute
    f.rename(columns={"comments_date": "date_time"}, inplace=True)
    return f


def create_wordcloud(df):
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color="white")
    df_wc = wc.generate(df["translated_comments"].str.cat(sep=" "))
    return df_wc


def data_clean():
    def clean_text(text):
        text = re.sub(r'@[A-Za-z0-9]+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'RT[\s]+', '', text)
        text = re.sub(r'https?:\/\/\S+', '', text)

        return text

    df["translated_comments"] = df["translated_comments"].apply(clean_text)

    def convert_lower(text):
        return text.lower()

    df["translated_comments"] = df["translated_comments"].apply(convert_lower)

    def remove_special(text):

        x = ''

        for i in text:
            if i.isalnum():
                x = x + i
            else:
                x = x + ' '
        return x

    df["translated_comments"] = df["translated_comments"].apply(remove_special)
    return df


def remove_stopwords(text):
    x = []

    for i in text.split():

        if i not in stopwords.words("english"):
            x.append(i)

    y = x[:]
    x.clear()
    return y


def join_back(list):
    return " ".join(list)


def most_common_words():
    words = []
    for i in df["translated_comments"]:
        for j in i.lower().split():
            words.append(j)

    from collections import Counter
    common_word_df = pd.DataFrame(Counter(words).most_common(20))
    return common_word_df


def month_timeline():
    df["month_num"] = df["date_time"].dt.month
    timeline_month = df.groupby(["year", "month_num", "month"]).count()["translated_comments"].reset_index()

    month_time = []

    for i in range(timeline_month.shape[0]):
        month_time.append(timeline_month["month"][i] + "-" + str(timeline_month["year"][i]))

    timeline_month["month_time"] = month_time

    return timeline_month


def week_timeline():
    df["week"] = df["date_time"].dt.week
    timeline_week = df.groupby(["year", "month_num", "month", "week"]).count()["translated_comments"].reset_index()
    week_time = []

    for i in range(timeline_week.shape[0]):
        week_time.append(timeline_week["month"][i] + "-" + "week no." + str(timeline_week["week"][i]))

    timeline_week["week_time"] = week_time

    return timeline_week


def daily_timeline():
    df["date"] = df["date_time"].dt.date
    timeline_daily = df.groupby(["date"]).count()["translated_comments"].reset_index()
    return timeline_daily


def period():
    period = []
    for i in df[["day_name", "hour"]]["hour"]:
        if i == 23:
            period.append(str(i) + "-" + str("00"))
        elif i == 0:
            period.append(str("00") + "-" + str(i + 1))
        else:
            period.append(str(i) + "-" + str(i + 1))
    df["period"] = period
    heatmap_daily_activity = df.pivot_table(index='day_name', columns="period", values='translated_comments',
                                            aggfunc="count").fillna(0)
    return heatmap_daily_activity


def get_polarity(text):
    return TextBlob(text).sentiment.polarity


def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def get_analysis(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    elif score > 0:
        return "Positive"


def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2, 2), max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec1.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]




with st.form(key="form1"):
    name = st.text_input(label='Search')
    submit = st.form_submit_button("submit")

if submit:
    from youtubesearchpython import VideosSearch

    video_search = VideosSearch(name, limit=10)
    # search_result = video_search.result()["result"][0]["title"]
    video_titles = []
    view_count = []
    video_link = []
    video_id = []

    for i in range(0, 10):
        video_titles.append(video_search.result()["result"][i]["title"])
        view_count.append(video_search.result()["result"][i]["viewCount"]["text"])
        video_link.append(video_search.result()["result"][i]["link"])
        video_id.append(video_search.result()["result"][i]["id"])

    video_search_result_df = pd.DataFrame({"video_titles": video_titles,
                                           "view_count": view_count,
                                           "video_link": video_link,
                                           "video_id": video_id})
    st.header("Displaying the Dataframe containing TOP 10 SEARCH RESULTS with video titles, views and links")
    st.write(video_search_result_df)

    views = []
    for i in video_search_result_df["view_count"]:
        i = int(i.strip(" views").replace(",", ""))
        views.append(i)

    sorted_views_df = pd.DataFrame({"video_titles": video_titles,
                                    "view_count": views,
                                    "video_link": video_link,
                                    "video_id": video_id})

    new_sorted_views_df = sorted_views_df.sort_values(by=["view_count"], ascending=False, ignore_index=True)
    st.header("Displaying the Dataframe containing TOP 10 SEARCH RESULTS with video titles and links SORTED by the maximum number of views")
    st.write(new_sorted_views_df)
    st.info("""

    -----  **PLEASE NOTE** : This app automatically analyses the comments of the video with maximum number of views -----

    """)
    # st.info("""
    #
    # -----  **PLEASE NOTE** : if the search results do not match or you want to analise the video of your choice please paste the link manually below -----
    #
    # """)
    # extracting comments using video id

    st.header("Scrapping the comments please wait.....")
    from youtubesearchpython import *

    video_id_extract = new_sorted_views_df["video_id"][0]
    comments = Comments(video_id_extract)
    while comments.hasMoreComments:
        comments.getNextComments()
    comments_result = comments.comments["result"]

    extracted_comments = []
    comments_date = []

    for i in range(len(comments_result)):
        comments_date.append(comments_result[i]["published"].strip(" (edited)"))
        extracted_comments.append(comments_result[i]["content"])

    comments_df = pd.DataFrame({"extracted_comments": extracted_comments, "comments_date": comments_date})

    import datetime as dt

    today = dt.datetime.today()

    converted_comment_date = []
    #
    for i in comments_df["comments_date"]:
        if "weeks" in i:
            i = int(i.replace(" weeks ago", ""))
            i = str(today - dt.timedelta(days=7 * i))

        elif "week" in i:
            i = int(i.replace(" week ago", ""))
            i = str(today - dt.timedelta(days=7))

        elif "hours" in i:
            i = i.replace(" hours ago", "")
            i = str(today - dt.timedelta(days=0))

        elif "hour" in i:
            i = i.replace(" hour ago", "")
            i = str(today - dt.timedelta(days=0))

        elif "minutes" in i:
            i = i.replace(" minutes ago", "")
            i = str(today - dt.timedelta(days=0))

        elif "minute" in i:
            i = i.replace(" minute ago", "")
            i = str(today - dt.timedelta(days=0))

        elif "seconds" in i:
            i = i.replace(" seconds ago", "")
            i = str(today - dt.timedelta(days=0))

        elif "second" in i:
            i = i.replace(" second ago", "")
            i = str(today - dt.timedelta(days=0))

        elif "days" in i:
            i = int(i.replace(" days ago", ""))
            i = str(today - dt.timedelta(days=i))

        elif "day" in i:
            i = int(i.replace(" day ago", ""))
            i = str(today - dt.timedelta(days=i))

        elif "months" in i:
            i = int(i.replace(" months ago", ""))
            i = str(today - dt.timedelta(days=30 * i))

        elif "month" in i:
            i = int(i.replace(" month ago", ""))
            i = str(today - dt.timedelta(days=30))

        elif "years" in i:
            i = int(i.replace(" years ago", ""))
            i = str(today - dt.timedelta(days=365 * i))

        elif "year" in i:
            i = int(i.replace(" year ago", ""))
            i = str(today - dt.timedelta(days=365))

        converted_comment_date.append(i[0:19])
    #
    new_extracted_comments_df = pd.DataFrame({"extracted_comments": extracted_comments, "comments_date": converted_comment_date})

    st.header("Displaying the dataset containing extracted comments with date(top rows)")
    st.write(new_extracted_comments_df.head())

    # new_extracted_comments_df["translated_comments"] = new_extracted_comments_df["extracted_comments"].copy()

    # st.header("Displaying the dataset of translated comments(<=100) displayed")
    # new_extracted_comments_df.to_csv("translated_comments.csv")
    # st.write(new_extracted_comments_df.head())
    # df = time_line_extractor(new_extracted_comments_df)
    # st.write(new_extracted_comments_df)
    from googletrans import Translator

    translated_comments = []
    for i in new_extracted_comments_df["extracted_comments"]:
        t = Translator()
        translated_comment = t.translate(i, dest="en").text
        translated_comments.append(translated_comment)
    new_extracted_comments_df["translated_comments"] = translated_comments
    #
    new_extracted_comments_df["translated_comments"] = new_extracted_comments_df["extracted_comments"].copy()
    new_extracted_comments_df.to_csv("translated_comments.csv")

    st.header("Displaying the dataset of translated comments(<=100) displayed")
    st.write(new_extracted_comments_df.head())





# st.header("Do you want to paste the link?")
#
# # st.info("""
# #
# # -----  **PLEASE** open up the side bar menu to paste the link -----
# #
# # """)
# with st.form(key="form2"):
#     # paste_link = st.text_input("Paste the link")
#     submit_button2 = st.form_submit_button("yes")
# if submit_button2:
#     with st.form(key="form3"):
#         paste_link = st.text_input("Paste the link")
#         submit_button2 = st.form_submit_button("submit")
#
# else:
#     pass

st.header("Upload the csv file stored in the folder to proceed the analysis")
uploaded_file = st.file_uploader("Choose a csv or excel file", type=["csv", "xlsx"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, engine="python", encoding="utf-8")
    st.subheader("Glimpse to the Dataset.")
    df = time_line_extractor(df)
    df = df.drop("Unnamed: 0", axis = 1)
    st.write(df)

        # build_model(df=df)
        # with st.form(key="form2"):
        #     submit_2 = st.form_submit_button("proceed")
        #
        # if submit_2:


else:
    st.info("Awaiting for the file to be uploaded")


st.header("Click on show analysis button to view analysis")
with st.form(key="form3"):
    submit_3 = st.form_submit_button("Show Analysis")
if submit_3:
    st.header("Total Comments:")
    total_comments = df["translated_comments"].shape[0]
    st.title(total_comments)

    st.header("Total Words:")
    words = []
    for w in df["translated_comments"]:
        words.extend(w.split())
        total_words = len(words)
    st.title(total_words)

    # Data Cleaning
    df = data_clean()
    df["translated_comments"] = df["translated_comments"].apply(remove_stopwords)
    df["translated_comments"] = df["translated_comments"].apply(join_back)

    st.header("Displaying dataset after Data cleaning")

    st.write(df)

    #Wordcloud
    st.header("Word Cloud")
    st.subheader("Displaying the wordcloud:")


    df_wc = create_wordcloud(df)
    fig,ax = plt.subplots()
    ax.imshow(df_wc)
    st.pyplot(fig)



    # Displaying most common words
    st.header("Displaying most common words")

    most_common_words()
    fig,ax = plt.subplots()
    ax.barh(most_common_words()[0], most_common_words()[1])
    plt.xticks(rotation="vertical")
    st.pyplot(fig)

    #displaying number of comments done in terms of month timeline
    st.header("Displaying number of comments done in terms of month timeline")
    timeline_month = month_timeline()
    fig,ax = plt.subplots()
    ax.plot(timeline_month["month_time"], timeline_month["translated_comments"])
    plt.xticks(rotation = 40)
    st.pyplot(fig)

    # displaying number of comments done in terms of week timeline
    st.header("Displaying number of comments done in terms of week timeline")
    timeline_week = week_timeline()

    fig, ax = plt.subplots()

    ax.plot(timeline_week["week"], timeline_week["translated_comments"])
    plt.xticks(rotation=40)
    st.pyplot(fig)

    # displaying number of comments done in terms of daily timeline
    st.header("Displaying number of comments done in terms of daily timeline")
    timeline_daily = daily_timeline()
    fig, ax = plt.subplots()
    ax.plot(timeline_daily["date"], timeline_daily["translated_comments"])
    plt.xticks(rotation=40)
    st.pyplot(fig)

    # displaying most busy days of comment section

    st.header("Displaying most busy days in comment section of a video")
    df["day_name"] = df["date_time"].dt.day_name()
    busy_days = df["day_name"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(busy_days.index, busy_days.values)
    plt.xticks(rotation=40)
    st.pyplot(fig)

    # displaying most busy hours per days in comment section
    # st.header("Displaying heatmap representing busy hours per day")
    # st.subheader("lighter shade represents the most busy section of the day")
    #
    # daily_busy_hours_heatmap = period()
    # fig, ax = plt.subplots()
    # ax = sns.heatmap(daily_busy_hours_heatmap)
    # st.pyplot(fig)

    #Assigning polarity
    df["Polarity"] = df["translated_comments"].apply(get_polarity)
    df["Subjectivity"] = df["translated_comments"].apply(get_subjectivity)
    df["Analysis"] = df["Polarity"].apply(get_analysis)
    cv = CountVectorizer()
    X = cv.fit_transform(df["translated_comments"]).toarray()

    st.header("Displaying Polarity vs subjectivity")
    fig, ax = plt.subplots()
    for i in range(0, df.shape[0]):
        plt.scatter(df["Polarity"][i], df["Subjectivity"][i], color="Blue")
    plt.title("Sentiment analysis")
    plt.xlabel("Polarity")
    plt.ylabel("Subjectivity")
    st.pyplot(fig)
    #
    st.header("Displaying Count plot for number of negative, positive and neutral comments")
    senti_count = df["Analysis"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(senti_count.index, senti_count.values)
    plt.xticks(rotation=40)
    st.pyplot(fig)

    # Displaying Bi_gram of top 20 words
    st.header("Displaying Bi_gram of top 20 words")

    vectorizer = CountVectorizer(max_features=600)  # considering first 600 unique words or features

    x = vectorizer.fit_transform(df["translated_comments"]).toarray()
    top2_words = get_top_n2_words(df["translated_comments"], n=200)  # top 200
    top2_df = pd.DataFrame(top2_words)
    top2_df.columns = ["Bi-gram", "Freq"]
    top2_df.head()

    top20_bigram_df=top2_df.iloc[0:20, :].set_index("Freq")
    fig, ax = plt.subplots()
    ax.bar(top20_bigram_df["Bi-gram"],top20_bigram_df.index)
    plt.xticks(rotation=80)


    st.pyplot(fig)



    # # Wordcloud
    # pos_comments = []
    # for i in range(0, len(df["Analysis"]):
    #     if df["Analysis"][i] == "Positive":
    #         pos_comments.append(df["translated_comments"][i])
    #
    #
    #
    # st.header("Word Cloud of Negative comments")
    # st.subheader("Displaying the wordcloud:")
    #
    # df_wc = word_cloud_1(list_tokenizer(analysis_neg), -1)
    # fig, ax = plt.subplots()
    # ax.imshow(df_wc)
    # st.pyplot(fig)
    #
    # # Wordcloud
    # st.header("Word Cloud of Positive Comments")
    # st.subheader("Displaying the wordcloud:")
    #
    # df_wc = word_cloud_1(list_tokenizer(analysis_pos), -1)
    # fig, ax = plt.subplots()
    # ax.imshow(df_wc)
    # st.pyplot(fig)
    #
    #
    df["Analysis"].replace({"Positive": 1, "Negative": -1, "Neutral": 0}, inplace=True)
    #



    #model building

    y = df["Analysis"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
    from sklearn.metrics import accuracy_score

    clf1 = GaussianNB()
    clf2 = BernoulliNB()
    clf3 = MultinomialNB()

    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    clf3.fit(X_train, y_train)

    st.header("Displaying guassian accuracy")
    y_pred1 = clf1.predict(X_test)
    gaus_acc = round(accuracy_score(y_test, y_pred1), 2)
    st.write(gaus_acc)
    st.header("Displaying bernoulli's accuracy")
    y_pred2 = clf2.predict(X_test)
    bern_acc = round(accuracy_score(y_test, y_pred2), 2)
    st.write(bern_acc)
    st.header("Displaying multinomial accuracy")
    y_pred3 = clf3.predict(X_test)
    multi_acc = round(accuracy_score(y_test, y_pred3), 2)
    st.write(multi_acc)

    df["Analysis"].replace({1:"Positive", -1:"Negative", 0:"Neutral"}, inplace=True)

    if df.Analysis.value_counts()["Positive"] > df.Analysis.value_counts()["Negative"]:
        st.header("Majority comments are POSITIVE")
    if df.Analysis.value_counts()["Negative"] > df.Analysis.value_counts()["Negative"]:
        st.header("Majority comments are NEGATIVE")

    #pie_chart
    st.header("Pie chart representing the percentage of comments belonging to each category")
    labels = "Neutral", "Positive", "Negative"
    sizes = [df["Analysis"].value_counts()[0]/100, df["Analysis"].value_counts()[1]/100, df["Analysis"].value_counts()[2]/100]
    explode = (0, 0.1, 0)
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis("equal")
    st.pyplot(fig1)



