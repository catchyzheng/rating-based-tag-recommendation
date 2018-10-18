# rating-based-tag-recommendation



In tag recommendation, the most important thing is to deal with relationships among users, items, tags and ratings. So far there are plenty of research and conclusions about the first three factors, but few people consider the influence of ratings on them. When tagging a specific item, a user would more or less be influenced by personal emotion, and emotion can to some extent be reflected by rating. 

This project raises a method named rating-based multi-layer tag recommendation(RMTR). By constructing a basic model and two sub models, this method can use the rating difference to analyze the hidden emotional influence, and therefore recommend different emotional tags. 

There are three models in this project.

Main model: {User: {Item: [Tags, [Ratings]] } } 

Item submodel: {Item: {Rating: [User]}} 

user submodel {User: {Rating: [Items]}}

All tags can generally be divided into three parts, sentiment, genre, proper. 

In this project I used a corpus named SentiWordNet. Each emotion word have a positive emotion value and negative emotion value in it. By converting it into absolute value, I can compare it with the score value given by users. By calculating how matching these two values, the word can be recommended as a potential tag for user. 

When there are enough samples of similar users and items, we will use rating-weighted collaborative filtering. When lacking such samples, we will use the cold-start strategy based on global sentimental emotion distribution. By adjusting the proportion of three parts, the recommendation list could be both subjective and objective. 

As for a specific user tagging a specific item, the recommending tags under different ratings could be harmony in diversity. 

When testing on two datasets of MovieLens which separately has 10 millions and 20 millions ratings and tags, RMTR can perform better than topN, 5.3% increase on Precision and 6.7% on Recall.