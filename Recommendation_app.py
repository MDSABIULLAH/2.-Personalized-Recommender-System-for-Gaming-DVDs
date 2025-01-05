from flask import Flask, request, render_template, redirect, url_for, flash
from sqlalchemy import create_engine, text, Table, Column, Integer, String, Float, MetaData
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote
from joblib import load
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For flash messages

# Load the user similarity matrix
user_similarity = load('user_similarity.joblib')

def get_db_engine(db_user, db_password, db_name):
    return create_engine(f"mysql+pymysql://{db_user}:%s@localhost/{db_name}" % quote(f'{db_password}'))

def init_db(engine):
    metadata = MetaData()
    game_table = Table('game', metadata,
        Column('userId', Integer, primary_key=True),
        Column('game', String(255), primary_key=True),
        Column('rating', Float)
    )
    metadata.create_all(engine)

def compute_user_similarity(user_item_matrix):
    return cosine_similarity(user_item_matrix)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        db_user = request.form['db_user']
        db_password = request.form['db_password']
        db_name = request.form['db_name']
        
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and file.filename.endswith('.csv'):
            try:
                engine = get_db_engine(db_user, db_password, db_name)
                init_db(engine)  # Ensure the table exists
                
                df = pd.read_csv(file)
                df.to_sql('game', con=engine, if_exists='replace', index=False)
                flash('File successfully uploaded and database initialized')
                return redirect(url_for('index'))
            except Exception as e:
                flash(f'An error occurred: {str(e)}')
                return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    db_user = request.form['db_user']
    db_password = request.form['db_password']
    db_name = request.form['db_name']
    user_id = int(request.form['user_id'])
    
    try:
        engine = get_db_engine(db_user, db_password, db_name)
        init_db(engine)  # Ensure the table exists
        
        # Create a session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Check if the table is empty
        result = session.execute(text("SELECT COUNT(*) FROM game")).scalar()
        if result == 0:
            flash('The game table is empty. Please upload data first.')
            return redirect(url_for('upload_file'))
        
        # Retrieve data from the database
        data = pd.read_sql_query(text('SELECT * FROM game'), con=engine)
        
        # Create user-item matrix
        user_item_matrix = data.pivot_table(index='userId', columns='game', values='rating')
        user_item_matrix_filled = user_item_matrix.fillna(0)
        
        # Check if pre-computed similarity matrix matches current data
        if user_similarity.shape[0] != user_item_matrix_filled.shape[0]:
            # Recompute similarity if mismatch
            user_similarity_new = compute_user_similarity(user_item_matrix_filled)
            user_similarity_df = pd.DataFrame(user_similarity_new, index=user_item_matrix_filled.index, columns=user_item_matrix_filled.index)
        else:
            user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix_filled.index, columns=user_item_matrix_filled.index)
        
        # Check if user exists in the dataset
        if user_id not in user_similarity_df.index:
            flash(f'User ID {user_id} not found in the dataset.')
            return redirect(url_for('index'))
        
        # Find similar users
        similar_users = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id).head(10)
        similar_users = similar_users[similar_users != 0]
        
        # Get unrated games
        user_ratings = user_item_matrix_filled.loc[user_id]
        unrated_games = user_ratings[user_ratings == 0].index
        
        # Generate recommendations
        recommendations_dict = {}
        for similar_user in similar_users.index:
            similar_user_ratings = user_item_matrix_filled.loc[similar_user, unrated_games]
            for game, rating in similar_user_ratings.items():
                if rating > 0:
                    if game in recommendations_dict:
                        recommendations_dict[game].append(rating)
                    else:
                        recommendations_dict[game] = [rating]
        
        recommendations = {game: sum(ratings) / len(ratings) for game, ratings in recommendations_dict.items()}
        recommendations_series = pd.Series(recommendations).sort_values(ascending=False).head(10)
        
        # Store recommendations in the database
        recommendations_df = pd.DataFrame(recommendations_series).reset_index()
        recommendations_df.columns = ['game', 'rating']
        recommendations_df['user_id'] = user_id
        recommendations_df.to_sql('recommendations', con=engine, if_exists='append', index=False)
        
        session.close()
        
        message = "Data stored successfully in the database."
        return render_template('result.html', message=message, recommendations=recommendations_series.items(), user_id=user_id)
    
    except Exception as e:
        flash(f'An error occurred: {str(e)}')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)