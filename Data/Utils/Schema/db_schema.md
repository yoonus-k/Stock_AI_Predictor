# Database Schema Documentation

Generated on: 2025-05-10 02:43:05

## Tables Overview

| Table Name | Row Count | Description |
|------------|-----------|-------------|
| users | 1 | |
| notifications | 30 | |
| live_articles | 24308 | |
| tweets | 16201 | |
| clusters | 289 | |
| patterns | 44517 | |
| stock_data | 578462 | |
| tweets_backup | 14426 | |
| predictions | 22 | |


## users

### Columns

| Column Name | Data Type | Not Null | Default Value | Primary Key |
|-------------|-----------|----------|---------------|-------------|
| UserID | INTEGER |  |  | ✓ |
| Username | VARCHAR(50) | ✓ |  |  |
| Password | VARCHAR(255) | ✓ |  |  |
| Email | VARCHAR(100) | ✓ |  |  |
| Preferences | TEXT |  |  |  |

### Sample Data

| Email | Password | Username | UserID | Preferences |
|---|---|---|---|---|
| it@arabiangates.com | <binary data: 60 bytes> | admin | 1 | admin |

### SQL Creation Statement

```sql
CREATE TABLE users (
        UserID INTEGER PRIMARY KEY,
        Username VARCHAR(50) NOT NULL,
        Password VARCHAR(255) NOT NULL,
        Email VARCHAR(100) NOT NULL,
        Preferences TEXT
    )
```

---

## notifications

### Columns

| Column Name | Data Type | Not Null | Default Value | Primary Key |
|-------------|-----------|----------|---------------|-------------|
| NotificationID | INTEGER |  |  | ✓ |
| UserID | INTEGER |  |  |  |
| PredictionID | INTEGER |  |  |  |
| SentTime | DATETIME |  | CURRENT_TIMESTAMP |  |
| NotificationType | VARCHAR(50) |  |  |  |
| Status | VARCHAR(20) |  |  |  |

### Sample Data

| Status | NotificationID | PredictionID | SentTime | UserID | NotificationType |
|---|---|---|---|---|---|
| Pending | 1 | 1 | 2025-04-10 00:00:00 | 1 | Email |
| Pending | 2 | 2 | 2025-04-10 00:00:00 | 1 | Email |
| Pending | 3 | 3 | 2025-04-10 00:00:00 | 1 | Email |
| Pending | 4 | 4 | 2025-04-10 00:00:00 | 1 | Email |
| Pending | 5 | 5 | 2025-04-10 00:00:00 | 1 | Email |

### SQL Creation Statement

```sql
CREATE TABLE notifications (
        NotificationID INTEGER PRIMARY KEY AUTOINCREMENT,
        UserID INTEGER,
        PredictionID INTEGER,
        SentTime DATETIME DEFAULT CURRENT_TIMESTAMP,
        NotificationType VARCHAR(50),
        Status VARCHAR(20)
    )
```

---

## live_articles

### Columns

| Column Name | Data Type | Not Null | Default Value | Primary Key |
|-------------|-----------|----------|---------------|-------------|
| ID | INTEGER |  |  | ✓ |
| Date | TEXT |  |  |  |
| Authors | TEXT |  |  |  |
| Source_Domain | TEXT |  |  |  |
| Source_Name | TEXT |  |  |  |
| Title | TEXT |  |  |  |
| Summary | TEXT |  |  |  |
| Url | TEXT |  |  |  |
| Topics | TEXT |  |  |  |
| Ticker_Sentiment | TEXT |  |  |  |
| Overall_Sentiment_Label | TEXT |  |  |  |
| Overall_Sentiment_Score | REAL |  |  |  |
| Event_Type | TEXT |  |  |  |
| Sentiment_Label | TEXT |  |  |  |
| Sentiment_Score | REAL |  |  |  |
| Fetch_Timestamp | TEXT |  |  |  |

### Sample Data

| Title | Summary | Authors | Topics | Source_Domain | Url | Ticker_Sentiment | Date | Event_Type | Sentiment_Score | Fetch_Timestamp | Overall_Sentiment_Score | Sentiment_Label | Overall_Sentiment_Label | ID | Source_Name |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bulls And Bears: Apple, Nike, US Steel - And The Markets Ride Out Ongoing Volatility Bulls And Bears: Apple, Nike, US Steel - And The Markets Ride Out Ongoing Volatility - Aehr Test System  ( NASDAQ:AEHR ) , Apple  ( NASDAQ:AAPL )  | Benzinga examined the prospects for many investors' favorite stocks over the last week - here's a look at some of our top stories. President Donald Trump's hold on global tariffs but imposition of a 145% levy on Chinese imports led to significant market volatility and eroding confidence in the ... | Benzinga Senior Editor | [{"topic": "Retail & Wholesale", "relevance_score": "0.25"}, {"topic": "Financial Markets", "relevance_score": "0.5855"}, {"topic": "Manufacturing", "relevance_score": "0.25"}, {"topic": "Technology", "relevance_score": "0.25"}, {"topic": "Finance", "relevance_score": "0.25"}] | www.benzinga.com | https://www.benzinga.com/25/04/44777882/benzinga-bulls-and-bears-apple-nike-us-steel-and-the-markets-ride-out-ongoing-volatility | [{"ticker": "AAPL", "relevance_score": "0.165788", "ticker_sentiment_score": "0.201345", "ticker_sentiment_label": "Somewhat-Bullish"}, {"ticker": "AVGO", "relevance_score": "0.165788", "ticker_sentiment_score": "0.291561", "ticker_sentiment_label": "Somewhat-Bullish"}, {"ticker": "TSLA", "relevance_score": "0.110973", "ticker_sentiment_score": "0.150212", "ticker_sentiment_label": "Somewhat-Bullish"}, {"ticker": "BAC", "relevance_score": "0.110973", "ticker_sentiment_score": "0.162778", "ticker_sentiment_label": "Somewhat-Bullish"}, {"ticker": "NISTF", "relevance_score": "0.055621", "ticker_sentiment_score": "-0.022069", "ticker_sentiment_label": "Neutral"}, {"ticker": "MU", "relevance_score": "0.110973", "ticker_sentiment_score": "-0.160342", "ticker_sentiment_label": "Somewhat-Bearish"}, {"ticker": "X", "relevance_score": "0.165788", "ticker_sentiment_score": "-0.062832", "ticker_sentiment_label": "Neutral"}, {"ticker": "NKE", "relevance_score": "0.165788", "ticker_sentiment_score": "-0.200942", "ticker_sentiment_label": "Somewhat-Bearish"}, {"ticker": "AMZN", "relevance_score": "0.110973", "ticker_sentiment_score": "0.150212", "ticker_sentiment_label": "Somewhat-Bullish"}, {"ticker": "MRVL", "relevance_score": "0.110973", "ticker_sentiment_score": "-0.013732", "ticker_sentiment_label": "Neutral"}, {"ticker": "FOREX:USD", "relevance_score": "0.055621", "ticker_sentiment_score": "0.094635", "ticker_sentiment_label": "Neutral"}] | 2025-04-12T12:01:20 |  | 0.0 | 2025-04-12T16:49:24.412646 | 0.170033 |  | Somewhat-Bullish | 1 | Benzinga |
| Should You Forget Apple and Buy These 2 Tech Stocks Instead? | VeriSign and Palo Alto Networks face fewer headwinds than the iPhone maker. | Leo Sun | [{"topic": "Earnings", "relevance_score": "1.0"}, {"topic": "Technology", "relevance_score": "1.0"}, {"topic": "Financial Markets", "relevance_score": "0.962106"}] | www.fool.com | https://www.fool.com/investing/2025/04/12/should-you-forget-apple-and-buy-these-2-tech-stock/ | [{"ticker": "AAPL", "relevance_score": "0.291182", "ticker_sentiment_score": "-0.01314", "ticker_sentiment_label": "Neutral"}, {"ticker": "GDDY", "relevance_score": "0.049629", "ticker_sentiment_score": "0.118165", "ticker_sentiment_label": "Neutral"}, {"ticker": "PANW", "relevance_score": "0.19661", "ticker_sentiment_score": "0.08133", "ticker_sentiment_label": "Neutral"}, {"ticker": "VRSN", "relevance_score": "0.244354", "ticker_sentiment_score": "0.24283", "ticker_sentiment_label": "Somewhat-Bullish"}] | 2025-04-12T11:00:00 |  | 0.0 | 2025-04-12T16:49:24.416646 | 0.219034 |  | Somewhat-Bullish | 2 | Motley Fool |
| Warren Buffett Owns 2 Artificial Intelligence  ( AI )  Stocks That Wall Street Says Could Soar Up to 50% | Berkshire Hathaway can be a great source of inspiration inspiration for individual investors. Warren Buffett, one of the most successful investors in American history, manages the vast majority of the company's $259 billion portfolio. | Trevor Jennewine | [{"topic": "Retail & Wholesale", "relevance_score": "0.333333"}, {"topic": "Financial Markets", "relevance_score": "0.214378"}, {"topic": "Earnings", "relevance_score": "0.999999"}, {"topic": "Technology", "relevance_score": "0.333333"}, {"topic": "Finance", "relevance_score": "0.333333"}] | www.fool.com | https://www.fool.com/investing/2025/04/12/warren-buffett-own-2-ai-stocks-wall-street-soar-50/ | [{"ticker": "SSNLF", "relevance_score": "0.042876", "ticker_sentiment_score": "0.080066", "ticker_sentiment_label": "Neutral"}, {"ticker": "AAPL", "relevance_score": "0.481177", "ticker_sentiment_score": "0.307987", "ticker_sentiment_label": "Somewhat-Bullish"}, {"ticker": "AMZN", "relevance_score": "0.481177", "ticker_sentiment_score": "0.508429", "ticker_sentiment_label": "Bullish"}, {"ticker": "BRK-A", "relevance_score": "0.042876", "ticker_sentiment_score": "0.227447", "ticker_sentiment_label": "Somewhat-Bullish"}] | 2025-04-12T08:01:00 |  | 0.0 | 2025-04-12T16:49:24.418646 | 0.277078 |  | Somewhat-Bullish | 3 | Motley Fool |
| Trump And Xi Jinping's Tariff Threats May Be 'Just For The LOLz,' Says Analyst: 'It's Not Like This Is A Great Financial Crisis' - Carrier Global  ( NYSE:CARR ) , Apple  ( NASDAQ:AAPL )  | Despite growing concerns about renewed trade tensions between Washington and Beijing, one analyst says the drama may be more performative than policy-driven. | Ananya Gairola | [{"topic": "Financial Markets", "relevance_score": "0.214378"}, {"topic": "Manufacturing", "relevance_score": "0.25"}, {"topic": "Energy & Transportation", "relevance_score": "0.25"}, {"topic": "Technology", "relevance_score": "0.25"}, {"topic": "Finance", "relevance_score": "0.25"}] | www.benzinga.com | https://www.benzinga.com/news/global/25/04/44777185/trump-and-xi-jinpings-tariff-threats-may-be-just-for-the-lolz-says-analyst-its-not-like-this-is-a-gre | [{"ticker": "CARR", "relevance_score": "0.121343", "ticker_sentiment_score": "-0.02334", "ticker_sentiment_label": "Neutral"}, {"ticker": "AAPL", "relevance_score": "0.181137", "ticker_sentiment_score": "0.075194", "ticker_sentiment_label": "Neutral"}, {"ticker": "TSLA", "relevance_score": "0.181137", "ticker_sentiment_score": "0.037306", "ticker_sentiment_label": "Neutral"}, {"ticker": "KO", "relevance_score": "0.121343", "ticker_sentiment_score": "-0.064176", "ticker_sentiment_label": "Neutral"}, {"ticker": "F", "relevance_score": "0.121343", "ticker_sentiment_score": "-0.259597", "ticker_sentiment_label": "Somewhat-Bearish"}, {"ticker": "HON", "relevance_score": "0.121343", "ticker_sentiment_score": "-0.250411", "ticker_sentiment_label": "Somewhat-Bearish"}, {"ticker": "CAT", "relevance_score": "0.121343", "ticker_sentiment_score": "-0.250411", "ticker_sentiment_label": "Somewhat-Bearish"}, {"ticker": "RTX", "relevance_score": "0.121343", "ticker_sentiment_score": "0.234329", "ticker_sentiment_label": "Somewhat-Bullish"}, {"ticker": "USEG", "relevance_score": "0.060848", "ticker_sentiment_score": "-0.055109", "ticker_sentiment_label": "Neutral"}, {"ticker": "GS", "relevance_score": "0.060848", "ticker_sentiment_score": "-0.145342", "ticker_sentiment_label": "Neutral"}, {"ticker": "CMI", "relevance_score": "0.121343", "ticker_sentiment_score": "0.234329", "ticker_sentiment_label": "Somewhat-Bullish"}] | 2025-04-12T04:21:42 |  | 0.0 | 2025-04-12T16:49:24.420647 | -0.115876 |  | Neutral | 4 | Benzinga |
| Inflation Might Be Easing But Tech Outlook Still Choppy, Analysts Warn - Apple  ( NASDAQ:AAPL )  | March CPI drops to 2.4%, below the forecasted 2.6% and down from 2.8% in February. Monthly inflation falls 0.1%, marking the weakest pace since May 2020. China's new tariffs just reignited the same market patterns that led to triple- and quadruple-digit wins for Matt Maley. | Vandana Singh | [{"topic": "Economy - Monetary", "relevance_score": "0.9545"}, {"topic": "Technology", "relevance_score": "1.0"}, {"topic": "Financial Markets", "relevance_score": "0.684621"}] | www.benzinga.com | https://www.benzinga.com/25/04/44772840/analysts-warn-of-choppy-tech-outlook-despite-inflation-relief | [{"ticker": "AAPL", "relevance_score": "0.229872", "ticker_sentiment_score": "0.0", "ticker_sentiment_label": "Neutral"}] | 2025-04-11T19:19:42 |  | 0.0 | 2025-04-12T16:49:24.422646 | -0.051002 |  | Neutral | 5 | Benzinga |

### SQL Creation Statement

```sql
CREATE TABLE "live_articles"
(
    ID                      INTEGER
        primary key autoincrement,
    Date                    TEXT,
    Authors                 TEXT,
    Source_Domain           TEXT,
    Source_Name             TEXT,
    Title                   TEXT,
    Summary                 TEXT,
    Url                     TEXT,
    Topics                  TEXT,
    Ticker_Sentiment        TEXT,
    Overall_Sentiment_Label TEXT,
    Overall_Sentiment_Score REAL,
    Event_Type              TEXT,
    Sentiment_Label         TEXT,
    Sentiment_Score         REAL,
    Fetch_Timestamp         TEXT
)
```

---

## tweets

### Columns

| Column Name | Data Type | Not Null | Default Value | Primary Key |
|-------------|-----------|----------|---------------|-------------|
| ID | INTEGER |  |  | ✓ |
| ticker_id | INTEGER |  |  |  |
| tweet_id | TEXT |  |  |  |
| tweet_text | TEXT |  |  |  |
| created_at | TEXT |  |  |  |
| retweet_count | INTEGER |  |  |  |
| reply_count | INTEGER |  |  |  |
| like_count | INTEGER |  |  |  |
| quote_count | INTEGER |  |  |  |
| bookmark_count | INTEGER |  |  |  |
| lang | TEXT |  |  |  |
| is_reply | BOOLEAN |  |  |  |
| is_quote | BOOLEAN |  |  |  |
| is_retweet | BOOLEAN |  |  |  |
| url | TEXT |  |  |  |
| search_term | TEXT |  |  |  |
| author_username | TEXT |  |  |  |
| author_name | TEXT |  |  |  |
| author_verified | BOOLEAN |  |  |  |
| author_blue_verified | BOOLEAN |  |  |  |
| author_followers | INTEGER |  |  |  |
| author_following | INTEGER |  |  |  |
| sentiment_label | TEXT |  |  |  |
| sentiment_score | REAL |  |  |  |
| sentiment_magnitude | REAL |  |  |  |
| weighted_sentiment | REAL |  |  |  |
| collected_at | TEXT |  |  |  |

### Indices

| Name | Unique | Columns |
|------|--------|--------|
| sqlite_autoindex_tweets_1 | ✓ | tweet_id |

### Sample Data

| sentiment_magnitude | search_term | reply_count | like_count | sentiment_label | author_name | author_following | collected_at | author_followers | ticker_id | created_at | ID | lang | retweet_count | author_username | tweet_id | author_blue_verified | url | weighted_sentiment | bookmark_count | is_reply | sentiment_score | quote_count | is_retweet | author_verified | tweet_text | is_quote |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8180049657821655 | $AAPL OR Apple -from:Apple | 0 | 0 | positive | DᴇRᴇᴢ | 2373 | 2025-04-15T15:22:19.550833 | 2128 | 3 | 2025-04-15T12:22:04+00:00 | 1 | en | 0 | iamDeRez1 | 1912119229421146412 | 0 | https://x.com/iamDeRez1/status/1912119229421146412 | 0.8394788772526892 | 0 | 1 | 0.8180049657821655 | 0 | 0 | 0 | @mightames_ Apple tree 🥲 | 0 |
| 0.852975606918335 | $AAPL OR Apple -from:Apple | 0 | 0 | positive | Khalil | 966 | 2025-04-15T15:22:19.550833 | 379 | 3 | 2025-04-15T12:22:03+00:00 | 3 | en | 0 | mybrotherkhalil | 1912119225113620504 | 0 | https://x.com/mybrotherkhalil/status/1912119225113620504 | 0.903861177751222 | 0 | 0 | 0.852975606918335 | 0 | 0 | 0 | Good morning I’m going to the Apple Store and asking them to put the missing chunk back | 0 |
| 0.5015538930892944 | $AAPL OR Apple -from:Apple | 0 | 0 | positive | Dev Roy | 841 | 2025-04-15T15:22:19.550833 | 45 | 3 | 2025-04-15T12:22:02+00:00 | 4 | en | 0 | debroy111 | 1912119220512366727 | 1 | https://x.com/debroy111/status/1912119220512366727 | 0.3395995729729626 | 0 | 0 | 0.5015538930892944 | 0 | 0 | 0 | Every step you take, every hour you sleep, every meal you log—your health data is being tracked by Apple Health, Fitbit, MyFitnessPal, and countless other apps.  But what’s the point if you’re not using it to improve your well-being?  Right now, your most valuable insights are scattered, unstructured, and wasted. You could be making data-driven decisions to reduce stress, improve sleep, and boost focus—but instead, you're left guessing.   What if you could actually use this data to optimize your daily performance?  At https://t.co/8VEdDtHIB4, we turn fragmented health data into a personalized Daily Wellness Score.   Even a 10% improvement in your score can lead to 25% higher productivity and lower stress-related costs.  ✅ Understand how small changes impact your energy & focus ✅ Get AI-powered recommendations to optimize well-being ✅ Turn passive data collection into actionable health insights  Your data is already working for you—it’s time to make smarter choices with it. Try IntraIntel Today, Link in the comments.   #HealthTech #AI #Wellness #Productivity #SmartDecisions | 0 |
| 0.9820035099983215 | $AAPL OR Apple -from:Apple | 1 | 0 | positive | BAC | 208 | 2025-04-15T15:22:19.550833 | 371 | 3 | 2025-04-15T12:21:56+00:00 | 5 | en | 0 | BAComer | 1912119195199758771 | 0 | https://x.com/BAComer/status/1912119195199758771 | 1.4292871370864566 | 0 | 0 | 0.9820035099983215 | 0 | 0 | 0 | Long weekend, last day of a four day tour of the Big Apple. My son’s first trip has been a good one!  #NewYorkCity #NYC https://t.co/0mCej8fvnT | 0 |
| 0.8292830586433411 | $AAPL OR Apple -from:Apple | 0 | 0 | neutral | markoni | 6 | 2025-04-15T15:22:19.550833 | 271 | 3 | 2025-04-15T12:21:56+00:00 | 6 | en | 0 | markoni2460 | 1912119194201563343 | 0 | https://x.com/markoni2460/status/1912119194201563343 | 0.0 | 0 | 0 | 0.0 | 0 | 0 | 0 | Trading stats from today's session:  )*:{-       https://t.co/zZLFbvxsHx  $AMZN $AAPL $BA $BABA $FB $TSLA $MSFT $ROKU https://t.co/Slah3CGBAj | 0 |

### SQL Creation Statement

```sql
CREATE TABLE "tweets"
(
    ID                   INTEGER
        primary key autoincrement,
    ticker_id            INTEGER,
    tweet_id             TEXT
        constraint tweets_pk
            unique,
    tweet_text           TEXT,
    created_at           TEXT,
    retweet_count        INTEGER,
    reply_count          INTEGER,
    like_count           INTEGER,
    quote_count          INTEGER,
    bookmark_count       INTEGER,
    lang                 TEXT,
    is_reply             BOOLEAN,
    is_quote             BOOLEAN,
    is_retweet           BOOLEAN,
    url                  TEXT,
    search_term          TEXT,
    author_username      TEXT,
    author_name          TEXT,
    author_verified      BOOLEAN,
    author_blue_verified BOOLEAN,
    author_followers     INTEGER,
    author_following     INTEGER,
    sentiment_label      TEXT,
    sentiment_score      REAL,
    sentiment_magnitude  REAL,
    weighted_sentiment   REAL,
    collected_at         TEXT
)
```

---

## clusters

### Columns

| Column Name | Data Type | Not Null | Default Value | Primary Key |
|-------------|-----------|----------|---------------|-------------|
| ClusterID | INTEGER |  |  | ✓ |
| StockID | INTEGER |  |  | ✓ |
| AVGPricePoints | TEXT |  |  |  |
| MarketCondition | VARCHAR(20) |  |  |  |
| Outcome | REAL |  |  |  |
| Label | VARCHAR(50) |  |  |  |
| ProbabilityScore | REAL |  |  |  |
| Pattern_Count | INTEGER |  |  |  |
| MaxGain | REAL |  |  |  |
| MaxDrawdown | REAL |  |  |  |

### Indices

| Name | Unique | Columns |
|------|--------|--------|
| sqlite_autoindex_clusters_1 | ✓ | ClusterID, StockID |

### Sample Data

| ClusterID | AVGPricePoints | MaxGain | MarketCondition | Pattern_Count | MaxDrawdown | StockID | Label | ProbabilityScore | Outcome |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.1204282999909643,0.08004142076208343,0.7813403587138953,0.9935825223445567,0.7117818008266777 | 0.0037438885658072803 | Bullish | 466 | -0.003219935389844481 | 1 | Buy | 0.5536423841059602 | 0.000572540037232745 |
| 1 | 0.9476779032864174,0.27824711248913536,0.05272065623734411,0.2329626862243028,0.7604704862866452 | -0.0033996977653736513 | Bearish | 119 | 0.003599018529897454 | 1 | Sell | 0.48676470588235293 | -0.0005121929694048881 |
| 2 | 0.07441323738639874,0.8449458732272557,0.8653368072751595,0.13170884636019664,0.8340556295736535 | -0.003521490063292828 | Bullish | 169 | 0.0029652131252742692 | 1 | Sell | 0.49125168236877526 | -0.00039551612750351605 |
| 3 | 0.6625307231674256,0.2745069290506686,0.9971139136657616,0.07055819338689445,0.19168162377711667 | 0.003562193819416933 | Bearish | 363 | -0.0028321416572186214 | 1 | Buy | 0.5622032288698955 | 0.0007538950921674978 |
| 4 | 0.5885016108526109,0.9054504281886504,0.01778713393398098,0.9096007515866507,0.27663101397487055 | 0.003567572748882749 | Bearish | 247 | -0.0033200701499902056 | 1 | Buy | 0.5362663495838288 | 0.0005341400574827298 |

### SQL Creation Statement

```sql
CREATE TABLE "clusters"
(
    ClusterID        INTEGER,
    StockID          INTEGER,
    AVGPricePoints   TEXT,
    MarketCondition  VARCHAR(20),
    Outcome          REAL,
    Label            VARCHAR(50),
    ProbabilityScore REAL,
    Pattern_Count    INTEGER,
    MaxGain          REAL,
    MaxDrawdown      REAL,
    primary key (ClusterID, StockID)
)
```

---

## patterns

### Columns

| Column Name | Data Type | Not Null | Default Value | Primary Key |
|-------------|-----------|----------|---------------|-------------|
| PatternID | INTEGER |  |  | ✓ |
| StockID | INTEGER |  |  | ✓ |
| ClusterID | INTEGER |  |  |  |
| PricePoints | TEXT |  |  |  |
| TimeSpan | VARCHAR(50) |  |  |  |
| MarketCondition | VARCHAR(20) |  |  |  |
| Outcome | REAL |  |  |  |
| Label | VARCHAR(50) |  |  |  |
| MaxGain | REAL |  |  |  |
| MaxDrawdown | REAL |  |  |  |

### Indices

| Name | Unique | Columns |
|------|--------|--------|
| sqlite_autoindex_patterns_1 | ✓ | PatternID, StockID |

### Sample Data

| ClusterID | MaxGain | TimeSpan | MarketCondition | StockID | Label | PricePoints | MaxDrawdown | PatternID | Outcome |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.004844818066583212 | 24 | Bullish | 1 | Buy | 0.215189873417728,1.0,0.6766398158803213,0.0,0.7468354430380089 | 0.0 | 0 | 0.0010342870029783789 |
| 25 | 0.0016200922444867608 | 24 | Bullish | 1 | Buy | 0.6453433678268965,0.5531514581373358,0.0,0.8400752587017735,1.0 | -0.002178210146893489 | 1 | 0.0002945622262703842 |
| 12 | -0.0037921587444085707 | 24 | Bullish | 1 | Sell | 0.5110062893081846,0.683176100628927,0.46226415094339757,0.0,1.0 | 0.0 | 2 | -0.0007661708483600917 |
| 25 | 0.0011320549899588555 | 24 | Bullish | 1 | Buy | 0.5968841285296946,0.5725413826679642,0.0,0.8695228821811014,1.0 | -0.0018996813187665605 | 3 | 0.0028611526801014617 |
| 25 | 0.005239910261685 | 24 | Bullish | 1 | Buy | 0.602987421383645,0.46226415094339757,0.0,1.0,0.6894654088050345 | -0.0007374688516445908 | 4 | 0.004657698010386666 |

### SQL Creation Statement

```sql
CREATE TABLE "patterns"
(
    PatternID       INTEGER,
    StockID         INTEGER,
    ClusterID       INTEGER,
    PricePoints     TEXT,
    TimeSpan        VARCHAR(50),
    MarketCondition VARCHAR(20),
    Outcome         REAL,
    Label           VARCHAR(50),
    MaxGain         REAL,
    MaxDrawdown     REAL,
    primary key (PatternID, StockID)
)
```

---

## stock_data

### Columns

| Column Name | Data Type | Not Null | Default Value | Primary Key |
|-------------|-----------|----------|---------------|-------------|
| StockEntryID | INTEGER |  |  | ✓ |
| StockID | INTEGER |  |  | ✓ |
| StockSymbol | TEXT |  |  |  |
| Timestamp | TEXT |  |  |  |
| TimeFrame | INTEGER |  |  | ✓ |
| OpenPrice | REAL |  |  |  |
| ClosePrice | REAL |  |  |  |
| HighPrice | REAL |  |  |  |
| LowPrice | REAL |  |  |  |

### Indices

| Name | Unique | Columns |
|------|--------|--------|
| sqlite_autoindex_stock_data_1 | ✓ | StockEntryID, StockID, TimeFrame |

### Sample Data

| StockSymbol | TimeFrame | LowPrice | HighPrice | StockEntryID | StockID | Timestamp | OpenPrice | ClosePrice |
|---|---|---|---|---|---|---|---|---|
| XAUUSD | 15 | 1281.43 | 1282.57 | 0 | 1 | 2019-01-02 01:00:00 | 1281.47 | 1281.46 |
| XAUUSD | 15 | 1280.87 | 1281.5 | 1 | 1 | 2019-01-02 01:15:00 | 1281.47 | 1281.05 |
| XAUUSD | 15 | 1280.78 | 1281.54 | 2 | 1 | 2019-01-02 01:30:00 | 1281.01 | 1281.35 |
| XAUUSD | 15 | 1280.93 | 1281.59 | 3 | 1 | 2019-01-02 01:45:00 | 1281.35 | 1281.29 |
| XAUUSD | 15 | 1281.25 | 1281.79 | 4 | 1 | 2019-01-02 02:00:00 | 1281.29 | 1281.71 |

### SQL Creation Statement

```sql
CREATE TABLE "stock_data"
(
    StockEntryID INTEGER,
    StockID      INTEGER,
    StockSymbol  TEXT,
    Timestamp    TEXT,
    TimeFrame    INTEGER,
    OpenPrice    REAL,
    ClosePrice   REAL,
    HighPrice    REAL,
    LowPrice     REAL,
    primary key (StockEntryID, StockID, TimeFrame)
)
```

---

## tweets_backup

### Columns

| Column Name | Data Type | Not Null | Default Value | Primary Key |
|-------------|-----------|----------|---------------|-------------|
| ID | INT |  |  |  |
| ticker_id | INT |  |  |  |
| tweet_id | TEXT |  |  |  |
| tweet_text | TEXT |  |  |  |
| created_at | TEXT |  |  |  |
| retweet_count | INT |  |  |  |
| reply_count | INT |  |  |  |
| like_count | INT |  |  |  |
| quote_count | INT |  |  |  |
| bookmark_count | INT |  |  |  |
| lang | TEXT |  |  |  |
| is_reply | NUM |  |  |  |
| is_quote | NUM |  |  |  |
| is_retweet | NUM |  |  |  |
| url | TEXT |  |  |  |
| search_term | TEXT |  |  |  |
| author_username | TEXT |  |  |  |
| author_name | TEXT |  |  |  |
| author_verified | NUM |  |  |  |
| author_blue_verified | NUM |  |  |  |
| author_followers | INT |  |  |  |
| author_following | INT |  |  |  |
| sentiment_label | TEXT |  |  |  |
| sentiment_score | REAL |  |  |  |
| sentiment_magnitude | REAL |  |  |  |
| weighted_sentiment | REAL |  |  |  |
| collected_at | TEXT |  |  |  |

### Sample Data

| sentiment_magnitude | search_term | reply_count | like_count | sentiment_label | author_name | author_following | collected_at | author_followers | ticker_id | created_at | ID | lang | retweet_count | author_username | tweet_id | author_blue_verified | url | weighted_sentiment | bookmark_count | is_reply | sentiment_score | quote_count | is_retweet | author_verified | tweet_text | is_quote |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8180049657821655 | $AAPL OR Apple -from:Apple | 0 | 0 | positive | DᴇRᴇᴢ | 2373 | 2025-04-15T15:22:19.550833 | 2128 | 3 | 2025-04-15T12:22:04+00:00 | 1 | en | 0 | iamDeRez1 | 1912119229421146412 | 0 | https://x.com/iamDeRez1/status/1912119229421146412 | 0.8394788772526892 | 0 | 1 | 0.8180049657821655 | 0 | 0 | 0 | @mightames_ Apple tree 🥲 | 0 |
| 0.852975606918335 | $AAPL OR Apple -from:Apple | 0 | 0 | positive | Khalil | 966 | 2025-04-15T15:22:19.550833 | 379 | 3 | 2025-04-15T12:22:03+00:00 | 3 | en | 0 | mybrotherkhalil | 1912119225113620504 | 0 | https://x.com/mybrotherkhalil/status/1912119225113620504 | 0.903861177751222 | 0 | 0 | 0.852975606918335 | 0 | 0 | 0 | Good morning I’m going to the Apple Store and asking them to put the missing chunk back | 0 |
| 0.5015538930892944 | $AAPL OR Apple -from:Apple | 0 | 0 | positive | Dev Roy | 841 | 2025-04-15T15:22:19.550833 | 45 | 3 | 2025-04-15T12:22:02+00:00 | 4 | en | 0 | debroy111 | 1912119220512366727 | 1 | https://x.com/debroy111/status/1912119220512366727 | 0.3395995729729626 | 0 | 0 | 0.5015538930892944 | 0 | 0 | 0 | Every step you take, every hour you sleep, every meal you log—your health data is being tracked by Apple Health, Fitbit, MyFitnessPal, and countless other apps.  But what’s the point if you’re not using it to improve your well-being?  Right now, your most valuable insights are scattered, unstructured, and wasted. You could be making data-driven decisions to reduce stress, improve sleep, and boost focus—but instead, you're left guessing.   What if you could actually use this data to optimize your daily performance?  At https://t.co/8VEdDtHIB4, we turn fragmented health data into a personalized Daily Wellness Score.   Even a 10% improvement in your score can lead to 25% higher productivity and lower stress-related costs.  ✅ Understand how small changes impact your energy & focus ✅ Get AI-powered recommendations to optimize well-being ✅ Turn passive data collection into actionable health insights  Your data is already working for you—it’s time to make smarter choices with it. Try IntraIntel Today, Link in the comments.   #HealthTech #AI #Wellness #Productivity #SmartDecisions | 0 |
| 0.9820035099983215 | $AAPL OR Apple -from:Apple | 1 | 0 | positive | BAC | 208 | 2025-04-15T15:22:19.550833 | 371 | 3 | 2025-04-15T12:21:56+00:00 | 5 | en | 0 | BAComer | 1912119195199758771 | 0 | https://x.com/BAComer/status/1912119195199758771 | 1.4292871370864566 | 0 | 0 | 0.9820035099983215 | 0 | 0 | 0 | Long weekend, last day of a four day tour of the Big Apple. My son’s first trip has been a good one!  #NewYorkCity #NYC https://t.co/0mCej8fvnT | 0 |
| 0.8292830586433411 | $AAPL OR Apple -from:Apple | 0 | 0 | neutral | markoni | 6 | 2025-04-15T15:22:19.550833 | 271 | 3 | 2025-04-15T12:21:56+00:00 | 6 | en | 0 | markoni2460 | 1912119194201563343 | 0 | https://x.com/markoni2460/status/1912119194201563343 | 0.0 | 0 | 0 | 0.0 | 0 | 0 | 0 | Trading stats from today's session:  )*:{-       https://t.co/zZLFbvxsHx  $AMZN $AAPL $BA $BABA $FB $TSLA $MSFT $ROKU https://t.co/Slah3CGBAj | 0 |

### SQL Creation Statement

```sql
CREATE TABLE tweets_backup
(
    ID                   INT,
    ticker_id            INT,
    tweet_id             TEXT,
    tweet_text           TEXT,
    created_at           TEXT,
    retweet_count        INT,
    reply_count          INT,
    like_count           INT,
    quote_count          INT,
    bookmark_count       INT,
    lang                 TEXT,
    is_reply             NUM,
    is_quote             NUM,
    is_retweet           NUM,
    url                  TEXT,
    search_term          TEXT,
    author_username      TEXT,
    author_name          TEXT,
    author_verified      NUM,
    author_blue_verified NUM,
    author_followers     INT,
    author_following     INT,
    sentiment_label      TEXT,
    sentiment_score      REAL,
    sentiment_magnitude  REAL,
    weighted_sentiment   REAL,
    collected_at         TEXT
)
```

---

## predictions

### Columns

| Column Name | Data Type | Not Null | Default Value | Primary Key |
|-------------|-----------|----------|---------------|-------------|
| PredictionID | INTEGER |  |  | ✓ |
| StockID | INTEGER |  |  |  |
| PatternID | INTEGER |  |  |  |
| NewsID | INTEGER |  |  |  |
| TweetID | INTEGER |  |  |  |
| PredictionDate | DATETIME DATETIME |  | CURRENT_TIMESTAMP |  |
| PredictedOutcome | TEXT |  |  |  |
| ConfidenceLevel | FLOAT |  |  |  |

### Sample Data

| TweetID | PredictionDate | PredictionID | StockID | NewsID | ConfidenceLevel | PatternID | PredictedOutcome |
|---|---|---|---|---|---|---|---|
| 0 | 2025-04-10 00:00:00 | 32 | 1 | 0 | 0.5368532654792197 | 19 | {"date": "2025-04-10 00:00:00", "stock_id": 1, "stock_name": "GOLD (XAUUSD)", "current_price": 3080.0, "pattern_prediction": 3081.774923960842, "final_prediction": 3082.3888788077315, "confidence": 0.5368532654792197, "action": "WEAK BUY", "position_size": 9626.542566817852, "pattern_metrics": {"pattern_id": 19, "type": "Buy", "probability": 0.4868532654792197, "max_gain": 0.003830958336284974, "max_drawdown": -0.003058014228975986, "reward_risk_ratio": 1.252760140873451}, "sentiment_metrics": {"Predicted News Sentiment Score": 0.05, "Predicted Impact Score": 0.05, "News Count": 1, "Bullish Ratio": 0.0, "Bearish Ratio": 0.0, "Summary of the News": "U.S. stock futures climbed early Thursday, buoyed by strong earnings reports from two of the \"Magnificent 7\" companies after Wednesday's market close. While markets were rattled by the Commerce Depart...", "Top Topics": ["Financial Markets", "Technology", "Earnings"], "Most Relevant Article": {"title": "US Stock Futures Rise As Meta, Microsoft Earnings Fuel Optimism: GDP Slump Linked To Tariff Anticipation, Not Recession, Says Analyst - Apple  ( NASDAQ:AAPL ) , Airbnb  ( NASDAQ:ABNB ) ", "summary": "U.S. stock futures climbed early Thursday, buoyed by strong earnings reports from two of the \"Magnificent 7\" companies after Wednesday's market close. While markets were rattled by the Commerce Department's report of a 0.3% GDP contraction, major indexes rebounded and ended the previous session ...", "url": "https://www.benzinga.com/markets/equities/25/05/45128568/us-stock-futures-rise-as-meta-microsoft-earnings-fuel-optimism-gdp-slump-linked-to-tariff-antici", "source": "Benzinga", "time_published": "20250501T101007", "relevance_score": 0.042647, "sentiment_score": 0.054498}}, "twitter_sentiment": {"tweets_sentiment_score": -0.055430758212293894, "tweets_count": 28, "most_positive_tweet": "$CHAU Gold pegged stabil coin (CHAU)\n\nTodays Prices \nXAU/USDT $3042 ( Gold ) \ud83d\udcc8\nCHAU/USDT $3042 ( Gold ) \ud83d\udcc8\n\nCHAU is a de-facto over collateralised stabil coin pegged to the price of gold. Decentralised pegging mechanism a hybrid between the solutions developed by Maker (MKR) and Synthetix (SYN). It borrows the best aspect from both systems where it has:\n\u25cf A lot lower collateral requirement than Synthetix\n\u25cf Better reward structure for opening of positions compared to Maker DAO\n\u25cf Non-inflationary stability mechanisms for the recapitalization of the system\n\u25cf 1:1 profit with the actual tracked asset\n\n\ud83d\udd25 https://t.co/hdmC7febBL\n\n#ChrysusDAO #CHAU #GOVtoken #MKR #Gold #Stabilcoin\n\n\ud83e\udd47\ud83d\udcca \ud83e\udd47\ud83d\udcca \ud83e\udd47\ud83d\udcca \ud83e\udd47\ud83d\udcca \ud83e\udd47\ud83d\udcca", "most_negative_tweet": "The greatest trick the devil ever pulled was convincing Wall Street the US could reshore its defense industrial base without crushing the real value of the UST market.\n\n\"And just like that, the purchasing power of your LT USTs in gold and BTC terms...is gone.\" https://t.co/GKmeZgBthw", "tweets_weighted_sentiment_score": -0.03835470163257222}} |
| 0 | 2025-04-10 00:00:00 | 33 | 3 | 0 | 0.6479381443298969 | 56 | {"date": "2025-04-10 00:00:00", "stock_id": 3, "stock_name": "APPL (AAPL)", "current_price": 199.43, "pattern_prediction": 200.77194381968945, "final_prediction": 200.8953937502843, "confidence": 0.6479381443298969, "action": "WEAK BUY", "position_size": 13422.625887720935, "pattern_metrics": {"pattern_id": 56, "type": "Buy", "probability": 0.5979381443298969, "max_gain": 0.013203984318550679, "max_drawdown": -0.008105474144064088, "reward_risk_ratio": 1.629020595694627}, "sentiment_metrics": {"Predicted News Sentiment Score": 0.11, "Predicted Impact Score": 0.16, "News Count": 50, "Bullish Ratio": 36.0, "Bearish Ratio": 2.0, "Summary of the News": "Smart Beta ETF report for ... U.S. stock futures climbed early Thursday, buoyed by strong earnings reports from two of the \"Magnificent 7\" companies after Wednesday's market close. While markets were ...", "Top Topics": ["Financial Markets", "Technology", "Earnings"], "Most Relevant Article": {"title": "Apple Ordered To Ease App Store Rules In US - Epic Games CEO Tim Sweeney Proposes Peace Deal, Saying 'We'll Return Fortnite' And Drop All Litigation Only If... - Tencent Holdings  ( OTC:TCEHY ) ", "summary": "Following a U.S. court ruling that found Apple Inc. AAPL violated the spirit of an injunction against anti-steering practices, Epic Games, backed by Tencent Holdings TCEHY, has offered to bring Fortnite back to the App Store if the tech giant fulfills this one condition.", "url": "https://www.benzinga.com/news/legal/25/04/45123864/apple-ordered-to-ease-app-store-rules-in-us-epic-games-ceo-tim-sweeney-proposes-peace-deal-saying-well", "source": "Benzinga", "time_published": "20250501T013457", "relevance_score": 0.811403, "sentiment_score": 0.133089}}, "twitter_sentiment": {"tweets_sentiment_score": 0.10743673245112101, "tweets_count": 15, "most_positive_tweet": "This is insane.\n\n$NVDA +15%\n$TSLA +17%\n$AAPL +11%\n$META +11.6%\n$PLTR +17.3%\n$AMZN +9.8%\n$MSFT +8.4%\n\nAnother day for the history books. https://t.co/lONWXlVaCQ", "most_negative_tweet": "I know we were in squeeze mode but $AAPL up 15% when Trump just increased tariff on China to 125% seems wrong lol https://t.co/i47aFSJy6B", "tweets_weighted_sentiment_score": 0.030175801188507698}} |
| 0 | 2025-04-30 00:00:00 | 34 | 3 | 0 | 0.46322751322751327 | 8 | {"date": "2025-04-30 00:00:00", "stock_id": 3, "stock_name": "APPL (AAPL)", "current_price": 182.39999389648438, "pattern_prediction": 181.9756876908115, "final_prediction": 182.0951514404402, "confidence": 0.46322751322751327, "action": "WEAK SELL", "position_size": -8675.784069046786, "pattern_metrics": {"pattern_id": 8, "type": "Sell", "probability": 0.5132275132275133, "max_gain": -0.017764152597915413, "max_drawdown": 0.014980200419682475, "reward_risk_ratio": -1.1858421182786785}, "sentiment_metrics": {"Predicted News Sentiment Score": 0.13, "Predicted Impact Score": 0.17, "News Count": 50, "Bullish Ratio": 40.0, "Bearish Ratio": 2.0, "Summary of the News": "A more conservative goal of $100 monthly dividend income would require 1,200 shares of Apple. An investor would need to own $1,275,000 worth of Apple to generate a monthly dividend income of $500. Tod...", "Top Topics": ["Financial Markets", "Technology", "Earnings"], "Most Relevant Article": {"title": "Apple Ordered To Ease App Store Rules In US - Epic Games CEO Tim Sweeney Proposes Peace Deal, Saying 'We'll Return Fortnite' And Drop All Litigation Only If... - Tencent Holdings  ( OTC:TCEHY ) ", "summary": "Following a U.S. court ruling that found Apple Inc. AAPL violated the spirit of an injunction against anti-steering practices, Epic Games, backed by Tencent Holdings TCEHY, has offered to bring Fortnite back to the App Store if the tech giant fulfills this one condition.", "url": "https://www.benzinga.com/news/legal/25/04/45123864/apple-ordered-to-ease-app-store-rules-in-us-epic-games-ceo-tim-sweeney-proposes-peace-deal-saying-well", "source": "Benzinga", "time_published": "20250501T013457", "relevance_score": 0.811403, "sentiment_score": 0.133089}}, "twitter_sentiment": {"tweets_sentiment_score": 0.5215881280601025, "tweets_count": 32, "most_positive_tweet": "$AAPL #AAPL Well these look quite similar... https://t.co/eIpCbCXLy6", "most_negative_tweet": "If you purchased $AAPL stock on its IPO day in 1980, and put $1000 in, today you will have $2.2 million. \n\nBut what you may not know is that if you purchased $1000 of $aapl stock in 1980, in 2000, you would have had $840.", "tweets_weighted_sentiment_score": 0.24279340940939736}} |
| 0 | 2025-04-30 00:00:00 | 35 | 3 | 0 | 0.593010752688172 | 40 | {"date": "2025-04-30 00:00:00", "stock_id": 3, "stock_name": "APPL (AAPL)", "current_price": 206.80499267578125, "pattern_prediction": 207.19743799177968, "final_prediction": 207.33288591224948, "confidence": 0.593010752688172, "action": "WEAK BUY", "position_size": 9077.480037246449, "pattern_metrics": {"pattern_id": 40, "type": "Buy", "probability": 0.543010752688172, "max_gain": 0.01341931697536911, "max_drawdown": -0.011774807627105735, "reward_risk_ratio": 1.1396633728841308}, "sentiment_metrics": {"Predicted News Sentiment Score": 0.13, "Predicted Impact Score": 0.17, "News Count": 50, "Bullish Ratio": 40.0, "Bearish Ratio": 2.0, "Summary of the News": "A more conservative goal of $100 monthly dividend income would require 1,200 shares of Apple. An investor would need to own $1,275,000 worth of Apple to generate a monthly dividend income of $500. Tod...", "Top Topics": ["Financial Markets", "Technology", "Earnings"], "Most Relevant Article": {"title": "Apple Ordered To Ease App Store Rules In US - Epic Games CEO Tim Sweeney Proposes Peace Deal, Saying 'We'll Return Fortnite' And Drop All Litigation Only If... - Tencent Holdings  ( OTC:TCEHY ) ", "summary": "Following a U.S. court ruling that found Apple Inc. AAPL violated the spirit of an injunction against anti-steering practices, Epic Games, backed by Tencent Holdings TCEHY, has offered to bring Fortnite back to the App Store if the tech giant fulfills this one condition.", "url": "https://www.benzinga.com/news/legal/25/04/45123864/apple-ordered-to-ease-app-store-rules-in-us-epic-games-ceo-tim-sweeney-proposes-peace-deal-saying-well", "source": "Benzinga", "time_published": "20250501T013457", "relevance_score": 0.811403, "sentiment_score": 0.133089}}, "twitter_sentiment": {"tweets_sentiment_score": 0.5215881280601025, "tweets_count": 32, "most_positive_tweet": "$AAPL #AAPL Well these look quite similar... https://t.co/eIpCbCXLy6", "most_negative_tweet": "If you purchased $AAPL stock on its IPO day in 1980, and put $1000 in, today you will have $2.2 million. \n\nBut what you may not know is that if you purchased $1000 of $aapl stock in 1980, in 2000, you would have had $840.", "tweets_weighted_sentiment_score": 0.24279340940939736}} |
| 0 | 2025-05-01 00:00:00 | 36 | 3 | 0 | 0.5340041279669763 | 19 | {"date": "2025-05-01 00:00:00", "stock_id": 3, "stock_name": "APPL (AAPL)", "current_price": 206.32000732421875, "pattern_prediction": 207.80379427571776, "final_prediction": 207.93892455270867, "confidence": 0.5340041279669763, "action": "WEAK BUY", "position_size": 11010.900381701491, "pattern_metrics": {"pattern_id": 19, "type": "Buy", "probability": 0.4840041279669763, "max_gain": 0.017122932554513647, "max_drawdown": -0.011927566461856012, "reward_risk_ratio": 1.4355763691841295}, "sentiment_metrics": {"Predicted News Sentiment Score": 0.13, "Predicted Impact Score": 0.17, "News Count": 50, "Bullish Ratio": 42.0, "Bearish Ratio": 2.0, "Summary of the News": "Apple stock shows short-term bullish signals, but longer-term indicators suggest resistance remains. Tariffs and AI delays may curb hardware momentum; services segment expected to grow double digits. ...", "Top Topics": ["Financial Markets", "Technology", "Earnings"], "Most Relevant Article": {"title": "Apple Ordered To Ease App Store Rules In US - Epic Games CEO Tim Sweeney Proposes Peace Deal, Saying 'We'll Return Fortnite' And Drop All Litigation Only If... - Tencent Holdings  ( OTC:TCEHY ) ", "summary": "Following a U.S. court ruling that found Apple Inc. AAPL violated the spirit of an injunction against anti-steering practices, Epic Games, backed by Tencent Holdings TCEHY, has offered to bring Fortnite back to the App Store if the tech giant fulfills this one condition.", "url": "https://www.benzinga.com/news/legal/25/04/45123864/apple-ordered-to-ease-app-store-rules-in-us-epic-games-ceo-tim-sweeney-proposes-peace-deal-saying-well", "source": "Benzinga", "time_published": "20250501T013457", "relevance_score": 0.811403, "sentiment_score": 0.133089}}, "twitter_sentiment": {"tweets_sentiment_score": 0.07639995962381363, "tweets_count": 16, "most_positive_tweet": "TRADE PLAN for Wednesday \ud83d\udd25\ud83d\udcc8\n\n$SPX setting up for a run to 5670 by Friday if we can get a positive reaction to $META $AMZN $AAPL earnings. \nSPX 5600C can work above 5565\n\n$TSLA finally closed above 290. TSLA to 300 possible tomorrow. If it reclaims 300..325 can come by next week. \nTSLA 300C is best above 290\n\n$META 600 in play if theres a positive reaction to earnings.\nMETA May 2 600C is best as an earnings lotto.", "most_negative_tweet": "Breaking: Apple, $AAPL, violated an antitrust ruling over its App Store, a judge ruled. The case was referred to federal prosecutors for a criminal contempt investigation.", "tweets_weighted_sentiment_score": 0.023229663543968326}} |

### SQL Creation Statement

```sql
CREATE TABLE "predictions" (
        PredictionID INTEGER PRIMARY KEY AUTOINCREMENT,
        StockID INTEGER,
        PatternID INTEGER,
        NewsID INTEGER,
        TweetID INTEGER,
        PredictionDate DATETIME DATETIME DEFAULT CURRENT_TIMESTAMP,
        PredictedOutcome TEXT CHECK(json_valid(PredictedOutcome)),
        ConfidenceLevel FLOAT
    )
```

---

