stocks ={
        1: "GOLD (XAUUSD)",
        2: "BTC (BTCUSD)",
        3: "APPLE (AAPL)",
        4: "Amazon (AMZN)",
        5: "NVIDIA (NVDA)",
        }
ticker = stock[1].split('(')[-1].replace(')', '').strip()


# Twitter

# input : 
Ex: def func(ticker, date):()
Ticker: AAPL
Date: 2025-04-10


# Output: Dict
return {
            'twitter_score': 0.5,
            'tweet_count': 100,
            'most_neg': "This is very bad",
            'most_pos': "This is very good",
            'weighted_sentiment': 0.5,
        }






# News

# input : 
Ticker: AAPL
Date: 2025-04-10

# Output: Dict
return {
            'Predicted News Sentiment Score': 0.5,
            'Predicted Impact Score': 0.5,
            'News Count': 100,
            'Bullish Ratio': 0.6,
            'Bearish Ratio': 0.4,
            'Summary of the News': summary (String),
            'Top Topics': [Financial Markets, Technology, Earnings],
            'Most Relevant Article': most_relevant_article
        }

most_relevant_article = {
                            'title': '10 Years - Apple',
                            'summary': '10 Years - Apple',
                            'url': 'https://www.benzinga.com/insights/news/25/04/44955624/heres-how-much-you-would-have-made-owning-apple-stock-in-the-last-10-years',
                            'source': 'Zacks Commentary',
                            'time_published': 20250425T160636,
                            'relevance_score': 0.4,
                            'sentiment_score': 0.5
                        }



