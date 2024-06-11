import alt
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import numpy as np
import time
import pickle

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
def load_data():
    senate_df = pd.read_csv('C:/Users/MA/OneDrive/Documents/1976-2020-senate.csv', encoding='latin-1')
    presidential_df = pd.read_csv('C:/Users/MA/OneDrive/Documents/1976-2020-president.csv', encoding='latin-1')

    senate_df.fillna({"party_detailed": "NA", "candidate": "NA"}, inplace=True)

    senate_df.drop("unofficial", axis=1, inplace=True)

    presidential_df.fillna({"party_detailed": "NA", "candidate": "NA"}, inplace=True)

    presidential_df.drop("notes", axis=1, inplace=True)

    presidential_df.isna().sum()
    return senate_df, presidential_df
senate_df, presidential_df = load_data()

st.sidebar.title("US Elections Analysis")
user_menu = st.sidebar.radio(
    'Select an Option',
    ('Overall Analysis','Senate Analysis', 'Presidential Analysis', 'State-Level Analysis','Candidate Performance','Party Performance','Special vs. Regular Elections','Sentiment Analysis','Prediction','Clustering','About')
)

def party_performance_over_time(data, election_type):
    # Filter out write-in candidates
    data = data[data['writein'] == False]

    # Aggregate data by party and year
    party_performance = data.groupby(['party_simplified', 'year']).sum()['candidatevotes'].reset_index()

    # Plot
    fig = px.line(party_performance, x='year', y='candidatevotes', color='party_simplified',
                  title=f'Party Performance Over Time ({election_type})')
    fig.update_layout(xaxis_title='Year', yaxis_title='Total Votes')
    st.plotly_chart(fig)

def state_level_analysis(data, election_type, selected_year, selected_state):
    # Filter data by selected year and state

    filtered_data = data[(data['year'] == selected_year) & (data['state'] == selected_state)]

    # Aggregate data by party
    party_votes = filtered_data.groupby('party_simplified').sum()['candidatevotes'].reset_index()

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=party_votes, x='party_simplified', y='candidatevotes',palette="viridis")
    plt.title(f'Total Votes by Party in {selected_state} in {selected_year} ({election_type})')
    plt.xlabel('Party')
    plt.ylabel('Total Votes')
    plt.xticks(rotation=45)
    st.pyplot(plt)

if user_menu == "State-Level Analysis":
    # State-Level Analysis for Senate
    st.sidebar.subheader("Senate State-Level Analysis")
    st.header("SENATE STATE LEVEL ANALYSIS")
    total_votes_per_state = senate_df.groupby('state')['totalvotes'].sum().sort_values(ascending=False)

    # Plotting total votes by state
    fig, ax = plt.subplots(figsize=(12, 8))
    total_votes_per_state.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Total Votes by State')
    ax.set_xlabel('State')
    ax.set_ylabel('Total Votes')
    ax.tick_params(axis='x', rotation=90)
    ax.grid(axis='y')
    plt.tight_layout()  # Adjust layout to prevent cutoff labels
    st.pyplot(fig)

    df = senate_df.groupby(['state'])['totalvotes'].sum().reset_index()
    df = df.sort_values(by='totalvotes', ascending=False).head(3)  # Selecting the top 3 states with highest votes

    st.title('Top 3 States with Highest Votes')

    # Creating a Plotly bar chart
    fig = px.bar(df, x='state', y='totalvotes', title='Top 3 States with Highest Votes', text_auto='.2s', height=700)
    fig.update_layout(xaxis_title='State', yaxis_title='Total Votes')

    # Displaying the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Displaying total number of votes for each party
    st.subheader("Total Number of votes for each party:")
    party_votes = senate_df.groupby('party_simplified')['candidatevotes'].sum().sort_values(ascending=False)
    party_votes_sorted = party_votes.sort_index()

    # Display number of votes for each party
    st.write("Number of votes for each party:")
    st.write(party_votes_sorted)

    # Plot a pie chart
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the size of the figure here
    wedges, texts, autotexts = ax.pie(party_votes_sorted, labels=party_votes_sorted.index, autopct='%1.1f%%',
                                      startangle=90, counterclock=False, textprops={'fontsize': 6})

    # Rotate the text vertically
    for text in texts:
        text.set_rotation(90)

    # Rotate the percentage text vertically
    for autotext in autotexts:
        autotext.set_rotation(90)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_aspect('equal')
    ax.set_ylabel('')  # Remove y-axis label

    st.pyplot(fig)
    senate_years = senate_df['year'].unique()
    selected_senate_year = st.sidebar.selectbox("Select Year (Senate):", senate_years)
    senate_states = senate_df['state'].unique()
    selected_senate_state = st.sidebar.selectbox("Select State (Senate):", senate_states)
    state_level_analysis(senate_df, "Senate", selected_senate_year, selected_senate_state)

    # State-Level Analysis for Presidential
    st.sidebar.subheader("Presidential State-Level Analysis")
    st.header("PRESIDENTIAL STATE LEVEL ANALYSIS")

    total_votes_per_state = presidential_df.groupby('state')['totalvotes'].sum().sort_values(ascending=False)

    # Plotting total votes by state
    fig, ax = plt.subplots(figsize=(12, 8))
    total_votes_per_state.plot(kind='bar', color='purple', ax=ax)
    ax.set_title('Total Votes by State')
    ax.set_xlabel('State')
    ax.set_ylabel('Total Votes')
    ax.tick_params(axis='x', rotation=90)
    ax.grid(axis='y')
    plt.tight_layout()  # Adjust layout to prevent cutoff labels
    st.pyplot(fig)

    df = presidential_df.groupby(by=['state', 'party_simplified', 'candidatevotes'])[['totalvotes']].sum().reset_index()
    df = df.sort_values(by=['totalvotes'], ascending=False).head(3)

    # Creating a Streamlit app
    st.title('Top 3 States with Highest Votes')

    # Creating a Plotly bar chart
    fig = px.bar(df, x='state', y='totalvotes', title='Top 3 States with Highest Votes', text_auto='.2s', height=700)
    fig.update_layout(xaxis_title='State', yaxis_title='Total Votes')

    # Displaying the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Total Number of votes for each party:")
    party_votes = presidential_df.groupby('party_simplified')['candidatevotes'].sum().sort_values(ascending=False)
    party_votes_sorted = party_votes.sort_index()
    st.write("Number of votes for each party:")
    st.write(party_votes_sorted)

    # Plot a pie chart
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(party_votes, labels=party_votes.index, autopct='%1.1f%%', startangle=90,textprops={'fontsize': 6})

    # Rotate the text labels vertically
    for text in texts:
        text.set_rotation(90)

    # Rotate the percentage text vertically
    for autotext in autotexts:
        autotext.set_rotation(90)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_aspect('equal')
    ax.set_ylabel('')  # Remove y-axis label

    st.pyplot(fig)

    presidential_years = presidential_df['year'].unique()
    selected_presidential_year = st.sidebar.selectbox("Select Year (Presidential):", presidential_years)
    presidential_states = presidential_df['state'].unique()
    selected_presidential_state = st.sidebar.selectbox("Select State (Presidential):", presidential_states)
    state_level_analysis(presidential_df, "Presidential", selected_presidential_year, selected_presidential_state)

def get_candidates_contested_in_both():
    senate_candidates = set(senate_df['candidate'])
    presidential_candidates = set(presidential_df['candidate'])
    candidates_contested_in_both = senate_candidates.intersection(presidential_candidates)
    return candidates_contested_in_both

def voter_turnout_over_time(data, election_type):
    data['turnout'] = data['totalvotes']
    turnout_data = data.groupby('year').sum()['turnout'].reset_index()
    fig = px.line(turnout_data, x='year', y='turnout', title=f'Voter Turnout Over Time ({election_type})')
    fig.update_layout(xaxis_title='Year', yaxis_title='Total Votes')
    st.plotly_chart(fig)

def winning_margins_analysis(data, election_type):
    data['winning_margin'] = data.groupby('year')['candidatevotes'].diff().abs().fillna(0)
    winning_margin_data = data.groupby('year').max()['winning_margin'].reset_index()
    fig = px.line(winning_margin_data, x='year', y='winning_margin', title=f'Winning Margins Over Time ({election_type})')
    fig.update_layout(xaxis_title='Year', yaxis_title='Votes Difference')
    st.plotly_chart(fig)
if user_menu == "Overall Analysis":
    # Party Performance Over Time for Senate
    party_performance_over_time(senate_df, "Senate")
    # Party Performance Over Time for Presidential
    party_performance_over_time(presidential_df, "Presidential")

if user_menu == "Senate Analysis":
    st.sidebar.header("Senate Analysis")
    years = senate_df['year'].unique()
    selected_year = st.sidebar.selectbox("Select Year", years)
    states = senate_df['state'].unique()
    selected_state = st.sidebar.selectbox("Select State", states)
    senate_data = senate_df[(senate_df['year'] == selected_year) & (senate_df['state'] == selected_state)]

    st.title("Senate Election Analysis")
    st.write(f"Analysis for the year {selected_year} in {selected_state}")

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(senate_data.describe())

    # Vote distribution
    st.subheader("Vote Distribution")
    plt.figure(figsize=(12, 6))
    sns.barplot(x='candidate', y='candidatevotes', data=senate_data, palette='viridis')
    plt.title('Candidate Votes Distribution')
    plt.xlabel('Candidate')
    plt.ylabel('Votes')
    plt.xticks(rotation=90)
    plt.grid(axis='x')
    st.pyplot(plt)

    st.subheader("Vote-wise Analysis")
    total_votes_per_year = senate_df.groupby('year')['totalvotes'].sum().reset_index()
    all_years = pd.DataFrame(
        {'year': range(total_votes_per_year['year'].min(), total_votes_per_year['year'].max() + 1)})
    total_votes_per_year = all_years.merge(total_votes_per_year, on='year', how='left').fillna(0)
    total_votes_per_year['year'] = total_votes_per_year['year'].astype(str)
    fig = px.bar(total_votes_per_year, x='year', y='totalvotes', title='Total Votes Per Year Senate')
    st.plotly_chart(fig)
    # Party-wise analysis
    st.subheader("Party-wise Analysis")
    party_counts = senate_data['party_simplified'].value_counts()
    fig = px.pie(values=party_counts, names=party_counts.index, title='Party Distribution')
    st.plotly_chart(fig)
    voter_turnout_over_time(senate_df, "Senate")
    senate_df['turnout_change'] = senate_df.groupby('state')['totalvotes'].diff()
    state_max_turnout_change = senate_df.groupby('state')['turnout_change'].max().sort_values(
        ascending=False).head(1)
    st.subheader("State with the most significant increase in voter turnout:")
    st.write(state_max_turnout_change)

    st.title('US Senate Election Results by Party')
    data = senate_df.groupby(['year', 'party_simplified'])['party_simplified'].count().unstack()
    attractive_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'
    ]
    # Plotting
    st.bar_chart(data, use_container_width=True, color=attractive_colors)
    winning_margins_analysis(senate_df, "Senate")

elif user_menu == "Presidential Analysis":
    st.sidebar.header("Presidential Analysis")
    years = presidential_df['year'].unique()
    selected_year = st.sidebar.selectbox("Select Year", years)
    states = presidential_df['state'].unique()
    selected_state = st.sidebar.selectbox("Select State", states)
    presidential_data = presidential_df[
        (presidential_df['year'] == selected_year) & (presidential_df['state'] == selected_state)]

    st.title("Presidential Election Analysis")
    st.write(f"Analysis for the year {selected_year} in {selected_state}")

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(presidential_data.describe())

    # Vote distribution
    st.subheader("Vote Distribution")
    plt.figure(figsize=(12, 6))
    sns.barplot(x='candidate', y='candidatevotes', data=presidential_data, palette='viridis')
    plt.title('Candidate Votes Distribution')
    plt.xlabel('Candidate')
    plt.ylabel('Votes')
    plt.xticks(rotation=90)
    plt.grid(axis='x')
    st.pyplot(plt)

    st.subheader("Vote-wise Analysis")
    total_votes_per_year = presidential_df.groupby('year')['totalvotes'].sum().reset_index()
    all_years = pd.DataFrame(
        {'year': range(total_votes_per_year['year'].min(), total_votes_per_year['year'].max() + 1)})
    total_votes_per_year = all_years.merge(total_votes_per_year, on='year', how='left').fillna(0)
    total_votes_per_year['year'] = total_votes_per_year['year'].astype(str)
    fig = px.bar(total_votes_per_year, x='year', y='totalvotes', title='Total Votes Per Year Presidential')
    st.plotly_chart(fig)

    # Party-wise analysis
    st.subheader("Party-wise Analysis")
    party_counts = presidential_data['party_simplified'].value_counts()
    fig = px.pie(values=party_counts, names=party_counts.index, title='Party Distribution')
    st.plotly_chart(fig)
    voter_turnout_over_time(presidential_df, "Presidential")
    presidential_df['turnout_change'] =presidential_df.groupby('state')['totalvotes'].diff()
    state_max_turnout_change = presidential_df.groupby('state')['turnout_change'].max().sort_values(ascending=False).head(1)
    st.subheader("State with the most significant increase in voter turnout:")
    st.write(state_max_turnout_change)

    st.title('US Presidential Election Results by Party')
    data = presidential_df.groupby(['year', 'party_simplified'])['party_simplified'].count().unstack()
    attractive_colors = [

        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#9467bd',  # Purple
        '#17becf'  # Cyan
    ]

    st.bar_chart(data, use_container_width=True, color=attractive_colors)
    winning_margins_analysis(presidential_df, "Presidential")

elif user_menu == "About":
    st.header("About US Elections Analysis")
    st.markdown("This web application analyzes data from US Senate and Presidential elections.")
    st.write("The datasets include information such as candidate names, parties, votes, and more.")
    st.write("You can explore different years, states, and election details.")
    st.write("The analysis includes visualizations and insights into election trends.")

# Clean candidate names by removing commas
senate_df['candidate'] = senate_df['candidate'].str.replace(',', '')
presidential_df['candidate'] = presidential_df['candidate'].str.replace(',', '')

# Clean candidate names by removing commas
senate_df['candidate'] = senate_df['candidate'].str.replace(',', '')
presidential_df['candidate'] = presidential_df['candidate'].str.replace(',', '')

# Function to calculate party performance
def calculate_party_performance(senate_df, presidential_df):
    # Party Performance Analysis for Senate Elections
    senate_party_performance = senate_df.groupby('party_simplified').agg({
        'candidatevotes': 'sum',
        'totalvotes': 'sum'
    }).reset_index()

    senate_party_performance['success_rate_senate'] = (senate_party_performance['candidatevotes'] / senate_party_performance['totalvotes']) * 100

    # Party Performance Analysis for Presidential Elections
    presidential_party_performance = presidential_df.groupby('party_simplified').agg({
        'candidatevotes': 'sum',
        'totalvotes': 'sum'
    }).reset_index()

    presidential_party_performance['success_rate_presidential'] = (presidential_party_performance['candidatevotes'] / presidential_party_performance['totalvotes']) * 100

    # Merge the two party performance dataframes for comparison
    merged_party_performance = pd.merge(senate_party_performance, presidential_party_performance, on='party_simplified', how='outer')

    # Calculate the difference in success rates between Senate and Presidential elections
    merged_party_performance['success_rate_difference'] = merged_party_performance['success_rate_presidential'] - merged_party_performance['success_rate_senate']

    # Adding total votes information to the merged dataframe
    merged_party_performance['total_votes_senate'] = senate_party_performance['totalvotes']
    merged_party_performance['total_votes_presidential'] = presidential_party_performance['totalvotes']

    return merged_party_performance

# Function to calculate total votes by state
def calculate_total_votes_by_state(df):
    total_votes_by_state = df.groupby('state_po')['totalvotes'].sum().reset_index()
    total_votes_by_state = total_votes_by_state.rename(columns={'totalvotes': 'total_votes'})
    return total_votes_by_state
def party_performance_by_state(data, election_type):
    state_performance = data.groupby(['state', 'party_simplified']).sum()['candidatevotes'].reset_index()
    fig = px.bar(state_performance, x='state', y='candidatevotes', color='party_simplified',
                 title=f'Party Performance by State ({election_type})')
    fig.update_layout(xaxis_title='State', yaxis_title='Total Votes')
    st.plotly_chart(fig)
if user_menu == "Party Performance":
    # Sidebar for selecting analysis type
    analysis_type = st.sidebar.selectbox('Select Analysis Type',
                                         ['Party Performance Comparison', 'Total Votes by State'])
    if analysis_type == 'Party Performance Comparison':
        # Checkbox for user to choose whether to execute the code
        execute_code = st.checkbox('Execute Party Performance Comparison')

        if execute_code:
            # Calculate party performance
            merged_party_performance = calculate_party_performance(senate_df, presidential_df)

            # Display merged party performance dataframe
            st.write(merged_party_performance)

            # Plot success rates for Senate and Presidential elections
            plt.figure(figsize=(12, 6))

            # Set bar width to avoid overlapping
            bar_width = 0.35

            plt.bar(np.arange(len(merged_party_performance)) - bar_width/2, merged_party_performance['success_rate_senate'], bar_width, color='blue', alpha=0.7, label='Senate')
            plt.bar(np.arange(len(merged_party_performance)) + bar_width/2, merged_party_performance['success_rate_presidential'], bar_width, color='red', alpha=0.7, label='Presidential')

            plt.xlabel('Political Party')
            plt.ylabel('Success Rate (%)')
            plt.title('Party Performance Comparison')
            plt.legend()
            plt.xticks(np.arange(len(merged_party_performance)), merged_party_performance['party_simplified'], rotation=45)
            plt.grid(axis='y')
            st.pyplot(plt)
        else:
            st.write('Party Performance Comparison is not executed.')

    elif analysis_type == 'Total Votes by State':
        # Calculate total votes by state for Senate and Presidential elections
        st.title("Total Votes By States")
        total_votes_senate = calculate_total_votes_by_state(senate_df)
        total_votes_presidential = calculate_total_votes_by_state(presidential_df)

        # Merge total votes dataframes
        merged_total_votes = pd.merge(total_votes_senate, total_votes_presidential, on='state_po',
                                      suffixes=('_senate', '_presidential'))

        # Replace NaN values with 0 to avoid a third color
        merged_total_votes.fillna(0, inplace=True)

        # Display total votes by state table
        st.write("Total Votes by State for Senate and Presidential Elections:")
        st.write(merged_total_votes)

        # Plot total votes by state
        plt.figure(figsize=(12, 6))

        # Set bar width
        bar_width = 0.35

        # Plot Senate total votes
        plt.bar(np.arange(len(merged_total_votes)) - bar_width / 2, merged_total_votes['total_votes_senate'], bar_width,
                color='blue', alpha=0.7, label='Senate')

        # Plot Presidential total votes
        plt.bar(np.arange(len(merged_total_votes)) + bar_width / 2, merged_total_votes['total_votes_presidential'],
                bar_width, color='red', alpha=0.7, label='Presidential')
        # Plot total votes by state (continued)
        plt.xlabel('State')
        plt.ylabel('Total Votes')
        plt.title('Total Votes by State')
        plt.legend()
        plt.xticks(np.arange(len(merged_total_votes)), merged_total_votes['state_po'], rotation=45)
        plt.grid(axis='y')

        st.pyplot(plt)
        party_performance_by_state(senate_df, "Senate")
        party_performance_by_state(presidential_df, "Presidential")

# Function to clean candidate names
def clean_candidate_names(df):
    df['candidate'] = df['candidate'].str.split(',').str[0]  # Take the first name in case of multiple names concatenated with commas
    return df

# Clean the candidate names in both datasets
senate_df = clean_candidate_names(senate_df)
presidential_df = clean_candidate_names(presidential_df)

# Function to compare Senate vs. Presidential performance
def plot_candidate_performance(df, selected_candidate, election_type):
    candidate_data = df[df['candidate'] == selected_candidate]

    if candidate_data.empty:
        st.write(f"No data available for {selected_candidate} in {election_type} elections.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.lineplot(x='year', y='candidatevotes', data=candidate_data, marker='o', ax=ax)
    ax.set_title(f'{election_type} Performance of {selected_candidate} Over the Years')
    ax.set_xlabel('Year')
    ax.set_ylabel('Candidate Votes')

    plt.tight_layout()
    st.pyplot(fig)

def top_candidates_analysis(data, election_type, top_n=5):
    top_candidates = data.groupby('candidate').sum()['candidatevotes'].nlargest(top_n).reset_index()
    fig = px.bar(top_candidates, x='candidate', y='candidatevotes', title=f'Top {top_n} Candidates Performance ({election_type})')
    fig.update_layout(xaxis_title='Candidate', yaxis_title='Total Votes')
    st.plotly_chart(fig)
if user_menu == "Candidate Performance":
    st.sidebar.header("Select a candidate for performance analysis")

    # Ensure unique candidate names are not comma-separated
    senate_candidates = senate_df['candidate'].drop_duplicates().tolist()
    presidential_candidates = presidential_df['candidate'].drop_duplicates().tolist()

    selected_election_type = st.sidebar.radio("Select Election Type", ("Senate", "Presidential"))

    if selected_election_type == "Senate":
        selected_candidate = st.sidebar.selectbox("Select a Senate candidate", senate_candidates)
        if selected_candidate:
            st.subheader(f"Senate Performance Analysis for {selected_candidate} Over the Years")
            plot_candidate_performance(senate_df, selected_candidate, "Senate")
            top_candidates_analysis(senate_df, "Senate")
    elif selected_election_type == "Presidential":
        selected_candidate = st.sidebar.selectbox("Select a Presidential candidate", presidential_candidates)
        if selected_candidate:
            st.subheader(f"Presidential Performance Analysis for {selected_candidate} Over the Years")
            plot_candidate_performance(presidential_df, selected_candidate, "Presidential")
            top_candidates_analysis(presidential_df, "Presidential")

def voting_mode_analysis(data, election_type, static_mode=None):
    if static_mode:
        data['mode'] = static_mode
    mode_data = data.groupby(['year', 'mode']).sum()['candidatevotes'].reset_index()
    fig = px.line(mode_data, x='year', y='candidatevotes', color='mode', title=f'Voting Mode Analysis Over Time ({election_type})')
    fig.update_layout(xaxis_title='Year', yaxis_title='Total Votes')
    st.plotly_chart(fig)
def special_vs_regular_elections():
    # Filter the dataset for special elections
    special_elections = senate_df[senate_df['special'] == True]
    # Filter the dataset for regular elections
    regular_elections = senate_df[senate_df['special'] == False]
    # Aggregate the data for special elections
    special_results = special_elections.groupby('year')['candidatevotes'].sum().reset_index()
    # Aggregate the data for regular elections
    regular_results = regular_elections.groupby('year')['candidatevotes'].sum().reset_index()
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='year', y='candidatevotes', data=special_results, marker='o', label='Special Elections')
    sns.lineplot(x='year', y='candidatevotes', data=regular_results, marker='o', label='Regular Elections')

    ax.set_title('Special Elections vs. Regular Elections')
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Votes')
    ax.legend()
    st.pyplot(fig)

if user_menu == "Special vs. Regular Elections":
    st.subheader("Special vs. Regular Elections Analysis")
    special_vs_regular_elections()
    party_performance_over_time(senate_df, "Senate")
    # Voting Mode Analysis for Senate
    voting_mode_analysis(senate_df, "Senate")

    # Party Performance Over Time for Presidential
    party_performance_over_time(presidential_df, "Presidential")
    # Voting Mode Analysis for Presidential with static mode "Total"
    voting_mode_analysis(presidential_df, "Presidential", static_mode="Total")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import plotly.graph_objects as go
import streamlit as st

# Ensure TextBlob corpora are downloaded
import nltk

nltk.download('punkt')

@st.cache_data
def load_data():
    Trump_reviews = pd.read_csv('C:/Users/MA/Downloads/trump_data.csv', index_col=0)
    Biden_reviews = pd.read_csv('C:/Users/MA/Downloads/biden_data.csv', index_col=0)
    return Trump_reviews, Biden_reviews

Trump_reviews, Biden_reviews = load_data()

# Function to calculate polarity
def polarity(review):
    return TextBlob(review).sentiment.polarity

# Calculate polarity for each review
Trump_reviews['polarity'] = Trump_reviews['text'].apply(polarity)
Biden_reviews['polarity'] = Biden_reviews['text'].apply(polarity)

# Classify expressions based on polarity
Trump_reviews['Expression'] = np.where(Trump_reviews['polarity'] > 0, 'Positive', 'Negative')
Trump_reviews.loc[Trump_reviews.polarity == 0, 'Expression'] = 'Neutral'
Biden_reviews['Expression'] = np.where(Biden_reviews['polarity'] > 0, 'Positive', 'Negative')
Biden_reviews.loc[Biden_reviews.polarity == 0, 'Expression'] = 'Neutral'
# Function to plot expression graph
def exp_graph(reviews, title):
    group = reviews.groupby('Expression').count()
    Pol_count = list(group['polarity'])
    Exp = list(group.index)
    group_list = list(zip(Pol_count, Exp))
    df = pd.DataFrame(group_list, columns=['Pol_count', 'Exp'])
    df['color'] = 'rgb(14,185,54)'
    df.loc[df.Exp == 'Neutral', 'color'] = 'rgb(18,29,31)'
    df.loc[df.Exp == 'Negative', 'color'] = 'rgb(206,31,31)'

    fig = go.Figure(go.Bar(x=df['Pol_count'], y=df['Exp'], orientation='h', marker={'color': df['color']}))
    fig.update_layout(title_text=title)
    st.plotly_chart(fig)

# Function to generate word cloud
def wordcloud(data, title):
    text = ' '.join(data.text)
    wc = WordCloud(max_font_size=100, max_words=500, scale=10, relative_scaling=0.6, background_color='white').generate(
        text)

    fig = plt.figure(figsize=(15, 10))
    plt.title(title, {'fontsize': 30, 'family': 'serif'})
    plt.axis('off')
    plt.imshow(wc, interpolation='bilinear')
    st.pyplot(fig)
def sentiment_pie_chart(reviews, title):
    sentiment_counts = reviews['Expression'].value_counts()
    labels = sentiment_counts.index
    values = sentiment_counts.values
    colors = ['lightgreen', 'lightcoral', 'lightskyblue']

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_traces(marker=dict(colors=colors, line=dict(color='#000000', width=1)))
    fig.update_layout(title=title)
    st.plotly_chart(fig)

if user_menu == "Sentiment Analysis":
    # Streamlit app
    st.title('Sentiment Analysis of Trump and Biden Tweets')
    # Sidebar
    st.sidebar.title("Choose Analysis")
    analysis = st.sidebar.selectbox("Select Analysis Type", ["Expression Graph", "Word Cloud","Pie Chart"])
    if analysis == "Pie Chart":
        st.subheader("Pie Chart of Sentiment Distribution")
        sentiment_pie_chart(Trump_reviews, "Sentiment Distribution of Trump's Tweets")
        sentiment_pie_chart(Biden_reviews, "Sentiment Distribution of Biden's Tweets")
    if analysis == "Expression Graph":
        st.subheader("Expression Graph")
        exp_graph(Trump_reviews, "Trump's Review Analysis")
        exp_graph(Biden_reviews, "Biden's Review Analysis")
    if analysis == "Word Cloud":
        st.subheader("Word Cloud")
        wordcloud(Trump_reviews, "Word Cloud for Trump's Tweet Replies")
        wordcloud(Biden_reviews, "Word Cloud for Biden's Tweet Replies")
###################################################################################################################################
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Preprocess the presidential data
X_presidential = presidential_df.drop(['candidate', 'party_detailed', 'writein', 'version', 'party_simplified'], axis=1)  # Features
y_presidential = presidential_df['party_simplified']  # Target variable

# Define preprocessing steps for presidential data
numeric_features_presidential = ['state_fips', 'state_cen', 'state_ic', 'candidatevotes', 'totalvotes']
categorical_features_presidential = ['state', 'office']

numeric_transformer_presidential = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

categorical_transformer_presidential = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Use OneHotEncoder for categorical features
])

preprocessor_presidential = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_presidential, numeric_features_presidential),
        ('cat', categorical_transformer_presidential, categorical_features_presidential)
    ])

# Append classifier to preprocessing pipeline for presidential data
rf_classifier_presidential = Pipeline(steps=[('preprocessor', preprocessor_presidential),
                                             ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Train the presidential model
rf_classifier_presidential.fit(X_presidential, y_presidential)

# Preprocess the senate data
X_senate = senate_df.drop(['candidate', 'party_detailed', 'writein', 'version', 'party_simplified'], axis=1)  # Features
y_senate = senate_df['party_simplified']  # Target variable

# Define preprocessing steps for senate data
numeric_features_senate = ['state_fips', 'state_cen', 'state_ic', 'candidatevotes', 'totalvotes']
categorical_features_senate = ['state', 'office']

numeric_transformer_senate = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

categorical_transformer_senate = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Use OneHotEncoder for categorical features
])

preprocessor_senate = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_senate, numeric_features_senate),
        ('cat', categorical_transformer_senate, categorical_features_senate)
    ])

# Append classifier to preprocessing pipeline for senate data
rf_classifier_senate = Pipeline(steps=[('preprocessor', preprocessor_senate),
                                       ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Train the senate model
rf_classifier_senate.fit(X_senate, y_senate)

if user_menu == 'Prediction':
    # Sidebar options
    option = st.sidebar.selectbox('Select Prediction', ['Presidential', 'Senate'])
    if option=='Presidential':
        st.header('Predict the Winner of the Next Presidential Election')
    # User selects the state for presidential prediction
        selected_state_presidential = st.selectbox('Select a State for Presidential Election Prediction', presidential_df['state'].unique())

        if st.button('Predict Presidential Election Winner'):
            # Collect features for the selected state for presidential prediction
            state_data_presidential = presidential_df[presidential_df['state'] == selected_state_presidential].head(1).drop(['candidate', 'party_detailed', 'writein', 'version', 'party_simplified'], axis=1)

        # Make presidential prediction
            presidential_prediction = rf_classifier_presidential.predict(state_data_presidential)

            # Display presidential prediction
            st.write(f'Predicted Winner of the Next Presidential Election in {selected_state_presidential}: {presidential_prediction[0]}')

    elif option == 'Senate':
        st.header('Predict the Winner of the Next Senate Election')
        # User selects the state for senate prediction
        selected_state_senate = st.selectbox('Select a State for Senate Election Prediction', senate_df['state'].unique())

        if st.button('Predict Senate Election Winner'):
        # Collect features for the selected state for senate prediction
            state_data_senate = senate_df[senate_df['state'] == selected_state_senate].head(1).drop(['candidate', 'party_detailed', 'writein', 'version', 'party_simplified'], axis=1)

        # Make senate prediction
            senate_prediction = rf_classifier_senate.predict(state_data_senate)

        # Display senate prediction
            st.write(f'Predicted Winner of the Next Senate Election in {selected_state_senate}: {senate_prediction[0]}')
#-----------------------------------------------------------------------------------------------------------------#
if user_menu=='Clustering':
    option = st.sidebar.selectbox('Select Clustering', ['Senate', 'Presidential'])

    if option == 'Senate':
        st.header('Senate Data Clustering')
        numeric_columns = senate_df.select_dtypes(include=['number']).columns.tolist()
        selected_features = st.multiselect('Select Features for Clustering', numeric_columns)

    # Preprocess data
        X_senate = senate_df[selected_features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_senate)

    # Perform KMeans clustering
        k = st.slider('Select Number of Clusters', min_value=2, max_value=10)
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X_scaled)
        senate_df['Cluster'] = kmeans.labels_

    # Display cluster centers
        st.subheader('Cluster Centers:')
        st.write(kmeans.cluster_centers_)

    # Plot clusters
        fig, ax = plt.subplots()
        for cluster in range(k):
            cluster_data = senate_df[senate_df['Cluster'] == cluster]
            ax.scatter(cluster_data[selected_features[0]], cluster_data[selected_features[1]], label=f'Cluster {cluster}')
        ax.set_xlabel(selected_features[0])
        ax.set_ylabel(selected_features[1])
        ax.set_title('Senate Data Clustering')
        ax.legend()
        st.pyplot(fig)

    else:
        st.header('Presidential Data Clustering')
        numeric_columns_presidential = presidential_df.select_dtypes(include=['number']).columns.tolist()
        selected_features_presidential = st.multiselect('Select Features for Clustering', numeric_columns_presidential)

    # Preprocess data
        X_presidential = presidential_df[selected_features_presidential].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_presidential)

    # Perform KMeans clustering
        k = st.slider('Select Number of Clusters', min_value=2, max_value=10)
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X_scaled)
        presidential_df['Cluster'] = kmeans.labels_

    # Display cluster centers
        st.subheader('Cluster Centers:')
        st.write(kmeans.cluster_centers_)

    # Plot clusters
        fig, ax = plt.subplots()
        for cluster in range(k):
            cluster_data = presidential_df[presidential_df['Cluster'] == cluster]
            ax.scatter(cluster_data[selected_features_presidential[0]], cluster_data[selected_features_presidential[1]], label=f'Cluster {cluster}')
        ax.set_xlabel(selected_features_presidential[0])
        ax.set_ylabel(selected_features_presidential[1])
        ax.set_title('Presidential Data Clustering')
        ax.legend()
        st.pyplot(fig)

st.markdown("---")
st.markdown("© 2024 All Rights Reserved by Aiman and Kanwal")