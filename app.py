
import streamlit as st 
from datetime import date
import matplotlib.pyplot as plt
import random,heapq,os
import google.generativeai as genai
from gtts import gTTS
from plotly import graph_objs as go
from datetime import date, timedelta,datetime
from dateutil.relativedelta import relativedelta

def business_days_count(month):
    no_of_days = 0
    Today = date.today()
    future_days = Today + relativedelta(months=month)
    while Today < future_days:
        if Today.weekday() not in [5,6]:
            no_of_days += 1
        Today += timedelta(days=1)
    return no_of_days

def display_result(train_data,final_forecast_df):
    result_generation = ['CONSIDER THESE AS SOME OF THE BEST STOCKS.', 'THESE STOCKS ARE WORTH CONSIDERING.',
    'TAKE A LOOK AT THESE TOP-PERFORMING STOCKS.', 'YOU MIGHT WANT TO CONSIDER THESE STOCKS.',
    'HERE ARE SOME NOTEWORTHY STOCKS FOR YOUR CONSIDERATION.', 'THESE STOCKS STAND OUT AS SOLID OPTIONS TO CONSIDER.']
    st.subheader(random.choice(result_generation))
    
    returns = {}
    print("display")
    # Finding the percentage growth for each stock.
    for i,j in zip(yahoo_finance_symbols,company_name):
        starting_value = train_data[i][-1] 
        predicted_value = final_forecast_df[i][-1]
        percentage = (predicted_value - starting_value)/starting_value
        returns[j] = round(percentage*100,2)
    
    # Finding top five returns
    top_returns_stock = heapq.nlargest(n, returns.items(), key=lambda x: x[1])

    # Display the top 5 stock returns
    text ,companies = "" ,"" 
    for idx, (company, num) in enumerate(top_returns_stock, start=1):
        st.write(f"{idx}. {company} will make a {num}% returns in the next {selected_time_range}")
        text += f"{idx}. {company} will make a {num}% returns in the next {selected_time_range}"
        companies += company
    return top_returns_stock, text, companies
    
# Graphical representation for the top 5 stock returns



def graphical_representation(top_returns_stock, final_forecast_df):
    for index, item in enumerate(company_name):
        # Create a new figure for each company
        
        fig = go.Figure()
        if item in dict(top_returns_stock).keys():
           
            # Plot actual historical stock prices
            fig.add_trace(go.Scatter(x=train_data.index[-no_of_days:], 
                                     y=train_data[yahoo_finance_symbols[index]][-no_of_days:],
                                     mode='lines',
                                     name=item,
                                     line=dict(color='blue')))
            
            # Plot AI-predicted stock prices
            fig.add_trace(go.Scatter(x=final_forecast_df.index[:no_of_days], 
                                     y=final_forecast_df[yahoo_finance_symbols[index]][:no_of_days],
                                     mode='lines',
                                     name="AI-predicted value",
                                     line=dict(color='orange')))
            
            fig.update_layout(title_text=f"{item} Stock Prices and Predictions",
                              xaxis_title='Date',
                              yaxis_title='Stock Price',
                              legend=dict(x=0, y=1, traceorder='reversed'),height = 600,width=700)
            
            # Display the plot in Streamlit
            st.plotly_chart(fig)


           
def summary_for_user(companies):
    
    os.environ["GOOGLE_API_KEY"]="AIzaSyDXrglS5viSsCi3inWpWRWstZy2c5oQwQc"
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-pro')
    # Prompting the large language model.
    text = "Generate a structured table encompassing 'Company Name,' 'Risk Factors,' 'Investor Overview,' and 'Latest News' for "+companies+" Populate the table with essential details, categorizing risk factors into low, medium, or high. Craft succinct overviews for each company, emphasizing crucial information for potential investors. Additionally, integrate the latest news relevant to each company, ensuring investors stay abreast of recent developments influencing their investment decisions."
    news = "Generate a concise seperate paragraph summarizing the latest news about"+companies+ "with a specific focus on information tailored to assist investors. Capture key developments, market trends, and financial updates, emphasizing insights that are pertinent for investors making informed decisions in today's dynamic market environment"
    response = model.generate_content(text, stream=True)
    response_news = model.generate_content(news, stream=True)
    msg = ''.join(chunk.text for chunk in response)
    latest_news = ''.join(chunk.text for chunk in response_news)

    title = ["Investor Snapshots: Unpacking Companies, Stakes, and Risk Insights",
    "Company Essentials: Investors, Shares, and Brief Risk Overview",
    "Quick Guides: Companies, Investor Highlights, and Risk Insights",
    "At a Glance: Companies, Investor Breakdowns, and Risk Profiles",
    "Investor Briefs: Companies, Stakes, and Concise Risk Advice"]
    st.subheader(random.choice(title))
    st.write(msg)
    return latest_news

# audio summary for user 
def audio(latest_news):
    title = ["Latest updates for investors","Recent developments impacting investors.",
    "Investor-focused news highlights","Breaking news for the investment community.",
    "News relevant to investors in recent times"]
    st.subheader(random.choice(title))
    # Generate audio
    latest_news = latest_news.replace('\n', ' ')
    latest_news = latest_news.replace('*', '')
    tts = gTTS(text=latest_news, lang="en", slow=False)
    audio_filename = "output.mp3"
    tts.save(audio_filename)

    # Play audio
    st.audio(audio_filename, format='audio/mp3', start_time=0)
    # Remove the audio file after playing
    os.remove(audio_filename)

if __name__ == "__main__":
    st.title("AI-Powered Investment Advisor")
    st.subheader("Building Long-Term Wealth through Smart Investment Suggestions")
    
    col1, col2= st.columns(2)
    col3 , col4 = st.columns(2)
    col5 = st.columns(1)[0]
    amount_to_invest = col1.number_input("Provide the amount you wish to invest",min_value=1000,step=1)

    selected_time_range = col2.selectbox("Choose the investment duration that suits you", ["None", "1 months", "2 months", "3 months", "4 months","5 months","6 months"], index=0)
  
    stock_type = col3.selectbox("Choose the stock type",["None","Indian Stocks","US Stocks"],index=0)
    Today = date.today()
    st.sidebar.text(Today.strftime("%d-%m-%Y"))

    if stock_type == "Indian Stocks":

        from indianmodel import main
        company_name = ['Reliance Industries Limited','ICICI Bank Limited','HDFC Bank Limited','HCL Technologies Limited','Maruti Suzuki India Limited','Titan Company Limited','Havells India Limited','Britannia Industries Limited','UltraTech Cement Limited','Trent Ltd.']
        yahoo_finance_symbols = ['RELIANCE.NS','ICICIBANK.NS','HDFCBANK.NS','HCLTECH.NS','MARUTI.NS','TITAN.NS','HAVELLS.NS','BRITANNIA.NS','ULTRACEMCO.NS','TRENT.NS']

        st.sidebar.subheader('AI will provide suggestions for the companies listed')
        for names in company_name:
            st.sidebar.write(f'- {names}')
    elif stock_type == "US Stocks":

        from usmodel import main
        company_name =['Microsoft','Amazon','JPMorgan Chase','Home Depot','Adobe','Thermo Fisher Scientific','Abbott Laboratories','Intuit','Danaher','Texas Instruments']
        yahoo_finance_symbols=['MSFT', 'AMZN', 'JPM', 'HD', 'ADBE', 'TMO', 'ABT', 'INTU', 'DHR', 'TXN']
        st.sidebar.subheader('AI will provide suggestions for the companies listed')
        for names in company_name:
            st.sidebar.write(f'- {names}')
    else:
        st.sidebar.subheader("Risk Disclosure Statement")
        st.sidebar.write("Before making any investment decisions, carefully read the disclaimers provided by the Securities and Exchange Board of India (SEBI). Investments are subject to market risks, and it is essential to understand these risks thoroughly. Take the time to review the disclaimer and make informed decisions based on your financial goals and risk tolerance")
        st.sidebar.write("[SEBI website](https://www.sebi.gov.in/)")
        st.sidebar.write("[Toll Free Helpline Service For Investors: 1800 266 7575](tel:18002667575)")

    n = col4.selectbox("Specify the number of Stocks",[0,3,4,5,6,7,8,9,10])

    if col5.button("APPLY"):
        if selected_time_range == "None":
            st.warning("Modify the investment duration from 'None' to a timeframe that fits your preferences.")
        elif amount_to_invest == 1000:
            st.warning("The investment must be greater than 1000")
        elif stock_type == "None":
            st.warning("Please make sure to select a stock type before proceeding.")
        elif n == 0:
            st.warning("Specify the number of Stocks other than 0 ")
        else:
            duration = {"1 months":1,"2 months":2,"3 months":3,"4 months":4,"5 months":5,"6 months":6}
            month = duration[selected_time_range]
            no_of_days = business_days_count(month)
            final_forecast_df ,train_data = main()
            if not final_forecast_df.empty:
                top_returns_stock, text, companies = display_result(train_data,final_forecast_df)
                graphical_representation(top_returns_stock, final_forecast_df)
                latest_news = summary_for_user(companies)
                audio(latest_news)


                
    



            
     
