"""

1.	Business Problem
1.1.	What is the business objective?
1.1.	Are there any constraints?

2.	Work on each feature of the dataset to create a data dictionary as displayed in the image below:


3.	Data Pre-processing
2.1 Data Cleaning and Data Mining.
4.	Exploratory Data Analysis (EDA):
4.1.	Summary.
4.2.	Univariate analysis.
4.3.	Bivariate analysis.

5.	Model Building
5.1	Build the Recommender Engine model on the given data sets.
6.	Write about the benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?


"""










"""
Problem Statement: -

Build a recommender system with the given data using UBCF.

This dataset is related to the video gaming industry and a survey was conducted to build a 
recommendation engine so that the store can improve the sales of its gaming DVDs. A snapshot of the dataset is given below. 
Build a Recommendation Engine and suggest top-selling DVDs to the store customers.

"""







"""

Business Objective:
Increase gaming DVD sales by providing personalized recommendations to customers.

Business Constraints:

Consider inventory limits and avoid recommending out-of-stock items.
Ensure computational efficiency for real-time recommendations.
Maintain diversity in recommendations to prevent repetitive suggestions.
Stay within budget constraints.

Success Criteria:
Business Success: Achieve a 15% increase in sales and a 20% improvement in customer satisfaction.
ML Success: Reach 75% accuracy in recommendations and ensure diversity.
Economic Success: Achieve a positive ROI within 12 months, staying within budget.

"""




# SOLVED THE PROBLEM USING UBCF(USER BASED COLLABORATIVE FILTERING).
# BECAUSE OF UBCF I CALCULATE THE USER - USER SIMILARITY AND THEN ON THE BASIS OF THAT I RECOMMENDED A MOVIE.

# SOMETIME IT WILL HAPPEN SOME USER WILL NOT HAVE A SIMILAR TYPE OF USER THEN IN THAT CASE MY MODEL DOES NOT RECOMMEND ANYTHING. 




# Data Dictionary

"""

ColumnName		Description
userId	        Unique identifier for each user. This is used to track which user has rated which game.
game	        The name or identifier of the game that has been rated by users. This column represents the items in the recommendation system.
rating	        The rating given by the user to the game. Typically ranges from 0 to 5, where higher values indicate greater preference or satisfaction.

"""


# importing the neccessary library.
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random
from sqlalchemy import create_engine, text
from urllib.parse import quote
from joblib import dump, load



# Loading the dataset
file_path = 'C:/Users/user/Desktop/DATA SCIENCE SUBMISSION/Data Science [ 6 ] - Recommendation Engine/Solution -  Approach 1 USER_BASED_COLLABORATIVE_FILTERING/Recommendation - User_Based_Collaborative_Solution/game.csv'  # Adjust the file path
data = pd.read_csv(file_path)  # Reading the dataset from a CSV file



# Database connection (if needed, this was in your original code)
user = "root"
pw = "12345678"
db = "univ_db"


# Establishing a connection to a MySQL database
engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))
# Storing data in the database
data.to_sql('game', con=engine, if_exists='replace', chunksize=1000, index=False)
# Retrieving data from the database
sql = 'SELECT * FROM game'
data = pd.read_sql_query(text(sql), con=engine)



# Creating the user-item matrix
# Pivot the data to create a user-item rating matrix from the dataset
user_item_matrix = data.pivot_table(index='userId', columns='game', values='rating')



# Fill NaN values with 0, unrated games have no rating from the user
user_item_matrix_filled = user_item_matrix.fillna(0)






# Checking for duplicate value.
user_item_matrix_filled.duplicated().sum()

# Dropping the duplicate value
user_item_matrix_filled.drop_duplicates(inplace=True)

# Again Checking for duplicates value
user_item_matrix_filled.duplicated().sum()







# Computing cosine similarity between users
# Calculating similarity between users using cosine similarity
user_similarity = cosine_similarity(user_item_matrix_filled,user_item_matrix_filled)
# Creating a DataFrame to easily work with the similarity matrix
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix_filled.index, columns=user_item_matrix_filled.index)





# Save using Joblib
dump(user_similarity, 'user_similarity.joblib')




# Output the first 100 users' similarity scores for inspection
user_similarity_df.head(100)




# Get user input for selecting a user ID and check if the input is valid
while True:
    try:
        user_input = input("Please enter a user ID for whom to generate recommendations: ")
        user_input = int(user_input)  # Attempt to convert the input to an integer
        if user_input in user_item_matrix_filled.index:  # Check if the user ID exists in the dataset
            user_id = user_input
            print(f"Selected User ID: {user_id}")
            break
        else:
            print("Invalid user ID. Please enter a valid user ID.")
    except ValueError:
        print("Invalid input. Please enter a numeric user ID.")




# Finding similar users to the selected user
# Sort users by similarity score, exclude the user themselves, and select the top 10 most similar users
similar_users = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id).head(10)
# Further filter to take only non-zero similarity scores
similar_users = similar_users[similar_users != 0].sort_values(ascending = False).head()
similar_users.head()
similar_users.info()





# Getting the games the selected user hasn't rated yet
user_ratings = user_item_matrix_filled.loc[user_id]  # Ratings for the selected user
unrated_mask = user_ratings == 0  # Identifying unrated games for the selected user
unrated_games = user_ratings[unrated_mask].index  # Getting the list of unrated game titles




# Recommending games based on ratings from similar users.
recommendations_dict = {}
# Iterate through similar users and their ratings for the unseen games by the selected user
for similar_user in similar_users.index:
    similar_user_ratings = user_item_matrix_filled.loc[similar_user, unrated_games]
    # Checking if the similar user's rating for each game is greater than zero
    for game, rating in similar_user_ratings.items():
        if rating > 0:  # Only consider games the similar user has rated
            if game in recommendations_dict:
                recommendations_dict[game].append(rating)
            else:
                recommendations_dict[game] = [rating]

print(recommendations_dict)





# Calculating the average rating for each game.
if recommendations_dict:
    # For each game, calculate the average of ratings from similar users
    recommendations = {game: sum(ratings) / len(ratings) for game, ratings in recommendations_dict.items()}

    # Convert the dict to a Pandas Series and sort by the highest average rating
    recommendations_series = pd.Series(recommendations).sort_values(ascending=False).head(10)

    # Output the top recommendations for the selected user
    print(f"\nTop Recommendations for User {user_id}:\n")
    print(recommendations_series)
else:
    # In case no new recommendations are available
    print(f"\nNo new recommendations available for User {user_id}.")












# Benefit of solution:
    
"""
Personalized Experience: By recommending games that align with a user's preferences, the system enhances user satisfaction and engagement, 
potentially leading to increased time spent on the platform and higher sales.


Customer Retention: Personalized recommendations encourage users to return to the platform, improving customer loyalty and retention rates.


Increased Revenue: With more relevant game suggestions, users are more likely to make additional purchases, 
directly boosting the business's revenue.


Efficient Marketing: The system helps target marketing efforts more effectively by understanding user preferences, 
enabling tailored promotions and offers.


Competitive Advantage: Offering a recommendation system can differentiate the business from competitors, 
attracting new users and maintaining a leading position in the market.



Data-Driven Insights: The system can provide valuable insights into user behavior and preferences, 
helping the business make informed decisions about inventory, promotions, and content development.
    
"""  
    