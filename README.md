# Self-supervised Learning for recommend
## 1.Model application scenarios and goals
- News recommendations
- Topic recommendations
- Proposal recommendations
- Recommended by similar members
## 2.Data processing
Match the IDs of topics and proposals with the user IDs in the person column and separate them.
## 3.Algorithm selection
### 3.1 Based on matrix decomposition (mainly used for recommendation of proposals, but can also be used for other recommendations that will be expanded later)
#### 3.1.1 Obtain data that can be trained by cleaning user historical behavior data 
#### 3.1.2Conduct matrix decomposition training on the cleaned data
#### 3.1.3 Obtain news/topics/proposals that users are interested in。By modifying the regular expression split, the user's interaction data of other items can be obtained, and through the matrix decomposition algorithm, other items that the user may be interested in in this category can be obtained and saved to the database.
### 3.2 Based on similar users (mainly used for topic recommendation to obtain similar users through two methods)
#### 3.2.1 Obtain similar users based on user topic interaction history.The solution uses statistics and summary of the access of a certain user's topic by other users to obtain the user's attention by those users at the same time, and then performs recommended operations.
#### 3.2.2 Obtain similar users based on the user’s participation in the same conference, etc.
### 3.3 Word vector training (recommendation methods for main users and new users)
#### 3.3.1 Word segmentation of Chinese sentences
#### 3.3.2 Obtain the word vector of the word segmentation. The trained model used here is trained"baike_26g_news_13g_novel_229g.bin" (including encyclopedia 26g, news 13g, novel 229g). The model comes from the Internet to obtain the word vector of the sentence.
#### 3.3.3 Label the word vector of the sentence and the word vector of the given label, so as to obtain the hidden label content contained in each sentence.
#### 3.3.4 For new users or users with less interaction, recommendations such as proposals can be made based on the tags selected by the user when registering.
![image name](https://github.com/AI-imp/Recommend/blob/main/SELFRec/model.png?raw=true)


