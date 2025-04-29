import requests
from datetime import datetime, timedelta
# ticker for forex gold is GC
# api key: 80LCP07NHD21JDYE
# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
#url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=NVDA&apikey=80LCP07NHD21JDYE&time_from=20220210T0130&time_to=20220310T0130&limit=1000'
def get_news_sentiment_analysis(stock_ticker, date_str, api_key='80LCP07NHD21JDYE'):
    """
    Get comprehensive news sentiment analysis for a specific stock and date.
    
    Args:
        stock_ticker (str): Stock ticker symbol (e.g., 'AAPL', 'NVDA', 'GC' for gold)
        date_str (str): Date in format 'YYYY-MM-DD HH:MM:SS'
        api_key (str): Alpha Vantage API key
        
    Returns:
        dict: Dictionary containing:
            - Predicted News Sentiment Score (average sentiment)
            - Predicted Impact Score (weighted by relevance)
            - News Count (number of relevant articles)
            - Bullish Ratio (percentage of bullish articles)
            - Bearish Ratio (percentage of bearish articles)
            - Summary of the News (concise summary)
            - Top Topics (most discussed topics)
            - Most Relevant Article (most relevant article details)
    """
    # ticker mapping for Alpha Vantage , BTCUSD is CRYPTO:BTC and XAUUSD is GC
    ticker_mapping = {
        'XAUUSD': 'FOREX:USD',
        'BTCUSD': 'CRYPTO:BTC',
        'AAPL': 'AAPL',
        'AMZN': 'AMZN',
        'NVDA': 'NVDA', 
    }
    stock_ticker = ticker_mapping.get(stock_ticker)  # Map to Alpha Vantage format if needed
    print(f"Using stock ticker: {stock_ticker}")
    
    try:
        # Convert input date to Alpha Vantage format (YYYYMMDDTHHMM)
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            
        except ValueError:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
        
        from_date = (dt - timedelta(days=1)).strftime('%Y%m%dT%H%M')
        to_date = dt.strftime('%Y%m%dT%H%M')
        print(f"From date: {from_date}, To date: {to_date}")
        
        # Build API URL
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock_ticker}&apikey={api_key}&time_from={from_date}&time_to={to_date}&limit=1000'
        #url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey=demo'
        # Make API request
        r = requests.get(url)
        data = r.json()
        
        
        if 'feed' not in data or not data['feed']:
            return {
                'Predicted News Sentiment Score': 0,
                'Predicted Impact Score': 0,
                'News Count': 0,
                'Bullish Ratio': 0,
                'Summary of the News': 'No news articles found for this date',
                'Top Topics': [],
                'Most Relevant Article': None
            }
        
        # Initialize metrics
        sentiment_scores = []
        impact_scores = []
        relevance_scores = []
        bullish_count = 0
        bearish_count = 0
        topics = {}
        most_relevant_article = None
        max_relevance = 0
        summaries = []
        
        for item in data['feed']:
            # Find sentiment for our target stock
            for ticker_sentiment in item.get('ticker_sentiment', []):
                if ticker_sentiment['ticker'] == stock_ticker:
                    sentiment = float(ticker_sentiment['ticker_sentiment_score'])
                    relevance = float(ticker_sentiment['relevance_score'])
                    
                    sentiment_scores.append(sentiment)
                    relevance_scores.append(relevance)
                    impact_scores.append(sentiment * relevance)  # Weighted impact
                    
                    # Count bullish and bearish articles (sentiment > 0.15)
                    if sentiment > 0.15:
                        bullish_count += 1
                    elif sentiment < -0.15:
                        bearish_count += 1
                    
                    # Track most relevant article
                    if relevance > max_relevance:
                        most_relevant_article = {
                            'title': item.get('title', ''),
                            'summary': item.get('summary', ''),
                            'url': item.get('url', ''),
                            'source': item.get('source', ''),
                            'time_published': item.get('time_published', ''),
                            'relevance_score': relevance,
                            'sentiment_score': sentiment
                        }
                        max_relevance = relevance
                    
                    summaries.append(item.get('summary', ''))
                    break
            
            # Analyze topics
            for topic in item.get('topics', []):
                topic_name = topic.get('topic', '')
                topic_relevance = float(topic.get('relevance_score', 0))
                if topic_name:
                    topics[topic_name] = topics.get(topic_name, 0) + topic_relevance
        
        if not sentiment_scores:
            return {
                'Predicted News Sentiment Score': 0,
                'Predicted Impact Score': 0,
                'News Count': 0,
                'Bullish Ratio': 0,
                'Summary of the News': 'No relevant news found for this stock',
                'Top Topics': [],
                'Most Relevant Article': None
            }
        
        # Calculate metrics
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) # Average sentiment
        avg_impact = sum(impact_scores) / sum(relevance_scores) if sum(relevance_scores) > 0 else 0 # Average impact score by relevance
        bullish_ratio = bullish_count / len(sentiment_scores) * 100 if sentiment_scores else 0 # Bullish ratio
        bearish_ratio = bearish_count / len(sentiment_scores) * 100 if sentiment_scores else 0 # Bearish ratio
        
        # Get top 3 topics
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]
        top_topics = [topic[0] for topic in sorted_topics]
        
        # Create a summary by joining the first few summaries
        summary = " ".join(summaries[:3])[:200] + "..." if summaries else "No summary available"
        
        return {
            'Predicted News Sentiment Score': round(avg_sentiment, 2),
            'Predicted Impact Score': round(avg_impact, 2),
            'News Count': len(sentiment_scores),
            'Bullish Ratio': round(bullish_ratio, 1),
            'Bearish Ratio': round(bearish_ratio, 1),
            'Summary of the News': summary,
            'Top Topics': top_topics,
            'Most Relevant Article': most_relevant_article
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {
            'error': str(e),
            'Predicted News Sentiment Score': 0,
            'Predicted Impact Score': 0,
            'News Count': 0,
            'Bullish Ratio': 0,
            'Summary of the News': 'Error processing news data',
            'Top Topics': [],
            'Most Relevant Article': None
        }


# main function
if __name__ == '__main__':
    # Example usage:
    api_key = '80LCP07NHD21JDYE'
    stock_ticker = 'XAUUSD'  # For gold you would use 'GC'
    date_str = '2025-04-10'  # Matches one of the articles in your sample

    result = get_news_sentiment_analysis(stock_ticker, date_str, api_key)
    print(result)
    # format the output
    print(f"Predicted News Sentiment Score: {result['Predicted News Sentiment Score']}")
    print(f"Predicted Impact Score: {result['Predicted Impact Score']}")
    print(f"News Count: {result['News Count']}")
    print(f"Bullish Ratio: {result['Bullish Ratio']}%")
    print(f"Bearish Ratio: {result['Bearish Ratio']}%")
    print(f"Summary of the News: {result['Summary of the News']}")
    print(f"Top Topics: {', '.join(result['Top Topics'])}")

    print(f"Most Relevant Article Title: {result['Most Relevant Article']['title']}")
    print(f"Most Relevant Article summary: {result['Most Relevant Article']['summary']}")
    print(f"Most Relevant Article URL: {result['Most Relevant Article']['url']}")
    #print(result)

#