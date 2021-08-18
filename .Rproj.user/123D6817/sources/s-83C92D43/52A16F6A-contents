# Bernie Sentiment Analysis ------------------------------------------------------
rm(list = ls())
# step 0: download tweets from Twitter 
# if you have not registered a Twitter developer account yet, skip to Step 1 to use the pre-downloaded tweets
library(twitteR)
library(ROAuth)
## Twitter authentication
# you need to create a Twitter developer account, then generate API keys
# if you have not done so, skip to step 1
# and read the pre-downloaded tweets saved in the csv file

# consumer_key <-'your consumer key'
# consumer_secret <- 'your consumer secret'
# access_token <- 'your access token'
# access_secret <- 'your access secret'

consumer_key <- 'UnUKsCQPgvvMTl49lm4hBmx1Z'
consumer_secret <- 'CMTXmY2efhdQrfnhxCPUfpCehTjb8EsRNDnKQc8KcIqWAgsxZT'
access_token <- '1352906495755755523-G8Wn6VFHIuBaA3kuTXTBA2q7d0and6'
access_secret <- 'CoV8VgvGUrTsz4O8czrrp1LCImLOyMo6X9rqBNOXlA5rT'

setup_twitter_oauth(consumer_key, consumer_secret, access_token,
                    access_secret)

# twitter_user is the twitter ID you want to process
# 3200 is the maximum number of tweets Twitter allows to retrieve (for free)
# we set to first download 500 tweets
twitter_user <- 'BernieSanders'
twitter_max <- 1000
# userTimeline download the timeline (tweets) from? twitter_user
# you can determine whether include retweets and replied tweets by parameters 
# userTimeline(user, n=20, maxID=NULL, sinceID=NULL, includeRts=FALSE, excludeReplies=FALSE, ...)
# see the help document for the details of those parameters
tweets <- userTimeline(twitter_user, n = twitter_max, includeRts=FALSE) #excluse retweets
(n.tweet <- length(tweets)) 
# convert tweets to a data frame
tweets.df <- twListToDF(tweets)
# save the downloaded tweets into a csv for future use
file_name = "Bernie_tweets.csv"
write.csv(tweets.df, file = file_name)

###### Step 1: read tweets from csv file 
tweets.df <- read.csv('Bernie_tweets.csv', stringsAsFactors = FALSE)
# # check the tweet #150
display_n <- 150
tweets.df[display_n, c("id", "created", "screenName", "replyToSN",
                       "favoriteCount", "retweetCount", "longitude", "latitude", "text")]

display_n <- 50
tweets.df[display_n, c("text")]

###### step 2 Text cleaning 
library(tm)
# build a corpus, and specify the source to be character vectors 

myCorpus_raw <- Corpus(VectorSource(tweets.df$text))
myCorpus <- myCorpus_raw 

lapply(myCorpus_raw[1:3], as.character)
lapply(myCorpus[1:3], as.character)

#Remove non-graphical characters 
# define an operation 'toSpace' to replace non-graphical characters with space
# gsub(pattern, replacement, x) is pattern replacement function 
# content_transformer() is a warp function 
# [:graph:] is an Regular Expression for visible characters 
# [^[:graph:]] is an Regular Expression for all non-visible characters 
# see more details about regular expression at:
# http://www.regular-expressions.info/posixbrackets.html
toSpace <- content_transformer(function(x, pattern) {return (gsub(pattern," ",x))})
myCorpus<- tm_map(myCorpus,toSpace,"[^[:graph:]]")

# convert to lower case
myCorpus <- tm_map(myCorpus, content_transformer(tolower))
# remove URLs
# [:space:] is an Regular Expression for whitespace 
# * here means to match '[^[:space:]]' (non-space) zero time or multiple times
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
myCorpus <- tm_map(myCorpus, content_transformer(removeURL))
# remove anything other than English letters or space 
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x) 
myCorpus <- tm_map(myCorpus, content_transformer(removeNumPunct))
# remove stopwords
myCorpus <- tm_map(myCorpus, removeWords, stopwords()) 
# remove extra whitespace
myCorpus <- tm_map(myCorpus, stripWhitespace)
# Text stemming (reduces words to their root form)
library("SnowballC")
myCorpus <- tm_map(myCorpus, stemDocument)

lapply(myCorpus_raw[1:3], as.character)
lapply(myCorpus[1:3], as.character)

###### step 3: Build Term Document Matrix
# create a term-document sparse matrix
# each row: a word (a feature)
# each column: a document (a tweet / a sample)
tdm <- TermDocumentMatrix(myCorpus,control = list(wordLengths = c(1, Inf)))
tdm
nrow(tdm) # number of words
ncol(tdm) # number of tweets


# inspect frequent words
# lowfreq means the lower frequency bound
freq_thre = 5
(freq.terms <- findFreqTerms(tdm, lowfreq = freq_thre))

# calculate the word frenquency 
term.freq <- rowSums(as.matrix(tdm)) 
# only keep the frequencies of words(terms) appeared more then freq_thre times
term.freq <- subset(term.freq, term.freq >= freq_thre) 
# select the words(terms) appeared more then freq_thre times, according to selected term.freq
df <- data.frame(term = names(term.freq), freq = term.freq)
library(ggplot2)
ggplot(df, aes(x=term, y=freq)) + geom_bar(stat="identity") +
  xlab("Terms") + ylab("Count") + coord_flip() +
  theme(axis.text=element_text(size=7))+theme_minimal()

####### next we will use the data prepared for different tasks

###### Associations
# which words are associated with a specific word? 
word <- 'covid'
cor_limit <- 0.2
findAssocs(tdm, word ,cor_limit )

# ExtinctionR
word <- 'climate'

word <- 'uk'
findAssocs(tdm, word ,cor_limit )


###### Task: Topic Modelling
dtm <- as.DocumentTermMatrix(tdm)
library(topicmodels)
topic_num = 6
term_num = 2
rowTotals <- apply(dtm , 1, sum) #Find the sum of words in each Document (tweet)
dtm  <- dtm[rowTotals> 0, ] #remove all docs without words
lda <- LDA(dtm, k = topic_num) # find k topics
term <- terms(lda, term_num) # first term_num terms of every topic
(term <- apply(term, MARGIN = 2, paste, collapse = ", "))


###### Task: Sentiment Analysis
# 1) positive, negative or neutral
# install package sentiment140
# install.packages('devtools')
library(devtools)
# install a package directly from GitHub
# see https://github.com/okugami79/sentiment140
install_github("okugami79/sentiment140")
# sentiment analysis
library(sentiment)
# use the raw text for sentiment analysis 
sentiments <- sentiment(tweets.df$text)
table(sentiments$polarity)

# sentiment visualisation
library(data.table)
sentiments$score <- 0
sentiments$score[sentiments$polarity == "positive"] <- 1
sentiments$score[sentiments$polarity == "negative"] <- -1
sentiments$date <- as.IDate(tweets.df$created)
result <- aggregate(score ~ date, data = sentiments, sum)
par(mar=c(2,2,2,2)) #set the margins of the plot
plot(result, type = "l")


# 2) using the emotion lexicon
install.packages('syuzhet')
library(syuzhet)
# use the cleaned text for emotion analysis 
# since get_nrc_sentiment cannot deal with non-graphic data 
tweet_clean <- data.frame(text = sapply(myCorpus, as.character), stringsAsFactors = FALSE)
# each row: a document (a tweet); each column: an emotion
emotion_matrix <- get_nrc_sentiment(tweet_clean$text)
tweet_clean$text[2]
emotion_matrix[2,]

# Matrix Transpose
# so each row will be an emotion and each column will be a tweet
td <- data.frame(t(emotion_matrix)) 
#The function rowSums computes column sums across rows for each level of a grouping variable.
td_new <- data.frame(rowSums(td))
td_new

#Transformation and cleaning
names(td_new)[1] <- "count"
td_new
td_new <- cbind("sentiment" = rownames(td_new), td_new)
td_new
rownames(td_new) <- NULL
td_new

# emotion Visualisation
library("ggplot2")
qplot(sentiment, data=td_new, weight=count, geom="bar",fill=sentiment)+ggtitle("Tweets emotion and sentiment")

