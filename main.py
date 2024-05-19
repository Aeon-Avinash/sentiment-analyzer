from transformers import pipeline
import torch
import gradio as gr


def sentiment_analysis(text):
  analyzer = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
  sentiment = analyzer(text)
  return [sentiment[0]['label'], sentiment[0]['score']]

# Test 1:
# print(analyzer(["This is awesome. Reliable product.", "Very expensive product. Company should use better pricing."]))
# print(sentiment_analysis(["This is awesome. Reliable product.", "Very expensive product. Company should use better pricing."]))
# [{'label': 'POSITIVE', 'score': 0.9998791217803955}, {'label': 'NEGATIVE', 'score': 0.9994811415672302}]

# Test with Gradio:

# gr.close_all()

# version 0.1
# demo = gr.Interface(
#     fn=sentiment_analysis, 
#     inputs=[gr.Textbox(label="Text Input for Sentiment Analysis", lines=4)], 
#     outputs=[gr.Textbox(label="Analyzed Sentiment", lines=4), gr.Textbox(label="Sentiment Strength", lines=1)], 
#     title="GenAI Sentiment Analyzer", 
#     description="This App does seniment analysis of text input")
# demo.launch()

# Uploading an excel file and getting output as required: 
import pandas as pd
import matplotlib.pyplot as plt

def create_charts(df):
    # Validate DataFrame
    if not all(col in df.columns for col in ['Review', 'Sentiment', 'Sentiment Score']):
        raise ValueError("The DataFrame must contain 'Review', 'Sentiment', and 'Sentiment Score' columns.")
    # Create Pie Chart for Sentiment Distribution
    sentiment_counts = df['Sentiment'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
    ax1.set_title('Distribution of Positive and Negative Reviews')

    # Create Scatter Plot for Sentiment Scores
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for sentiment, color in zip(['positive', 'negative'], ['green', 'red']):
        subset = df[df['Sentiment'].str.lower() == sentiment]
        ax2.scatter(subset.index, subset['Sentiment Score'], label=sentiment.capitalize(), color=color, alpha=0.6)

    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.set_xlabel('Review Index')
    ax2.set_ylabel('Sentiment Score')
    ax2.set_title('Scatter Plot of Reviews by Sentiment Score')
    ax2.legend()

    return fig1, fig2



def analyze_reviews(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)
    
   # Attempt to identify the review column if it is not labeled correctly
    if 'Review' not in df.columns:
        for col in df.columns:
            if df[col].dtype == 'object':  # Assuming reviews are text
                df.rename(columns={col: 'Review'}, inplace=True)
                break
    
    # Ensure the dataframe now has a 'Review' column
    if 'Review' not in df.columns:
        raise ValueError("The input file must contain a column with review text.")
    
    # Remove any column that contains serial numbers
    df = df[[col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) or col == 'Review']]
    

    # Apply the get_sentiment function to each review
    results = df['Review'].apply(sentiment_analysis)

    # Split the results into separate columns for sentiment and sentiment score
    [df['Sentiment'], df['Sentiment Score']] = zip(*results)

    # Adjust the sentiment score to be negative if the sentiment is negative
    df.loc[df['Sentiment'] == 'NEGATIVE', 'Sentiment Score'] *= -1

    pie_chart, scatter_plot = create_charts(df)

    return [df, pie_chart, scatter_plot]


# Example usage
file_path = '/teamspace/studios/this_studio/sentiment-analyzer/Sample_Sentiments (1).xlsx'
# result_df = analyze_reviews(file_path)
# print(result_df)




gr.close_all()

# version 0.2
demo = gr.Interface(
    fn=analyze_reviews, 
    inputs=[gr.File(label="Upload your excel file containing user reviews")], 
    outputs=[
        gr.DataFrame(label="Aanalysis of the uploaded excel file"), 
        gr.Plot(label="Sentiment Analysis - Positive & Negative"),
        gr.Plot(label="Sentiment Analysis - Sentiment Score Distribution")
        ], 
    title="GenAI Sentiment Analyzer", 
    description="This App does seniment analysis of User Reviews")
demo.launch()

 
