import pandas as pd
import seaborn as sns
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

ratings = pd.read_csv('/kaggle/input/movielens-32m/ml-32m/ratings.csv')
movies = pd.read_csv('/kaggle/input/movielens-32m/ml-32m/movies.csv')
movies['genres'] = movies['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
user_counts = ratings['userId'].value_counts()
eligible_users = user_counts[user_counts >= 20].index[:500]
ratings = ratings[ratings['userId'].isin(eligible_users)]
grid_size = 10
all_genres = list(set(g for genre_list in movies['genres'] for g in genre_list))
genre_grid = np.random.choice(all_genres, size=(grid_size, grid_size))

alpha = 0.1
gamma = 0.9
epsilon = 0.1
total_epochs = 200
actions = [0, 1, 2, 3]  # up, right, down, left

def get_reward(x, y, genre_counts):
    genre = genre_grid[x, y]
    return genre_counts.get(genre, 0)
    
def take_action(x, y, action):
    if action == 0 and x > 0: x -= 1
    elif action == 1 and y < grid_size - 1: y += 1
    elif action == 2 and x < grid_size - 1: x += 1
    elif action == 3 and y > 0: y -= 1
    return x, y

recall_progress = []
recalls_after_training = []

first_10_users = eligible_users[:10]

for epoch_limit in range(10, total_epochs + 1, 10):
    recalls = []

    for user_id in first_10_users:
        user_data = ratings[ratings['userId'] == user_id].sort_values('timestamp')
        train_data, test_data = train_test_split(user_data, test_size=0.3, random_state=42)

        train_movie_ids = set(train_data['movieId'])
        test_movie_ids = set(test_data['movieId'])

        train_movies = movies[movies['movieId'].isin(train_movie_ids)]
        test_movies = movies[movies['movieId'].isin(test_movie_ids)]

        genre_counts = Counter(g for g_list in train_movies['genres'] for g in g_list)

        q_table = defaultdict(lambda: np.zeros(4))

        for _ in range(epoch_limit):
            x, y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
            for _ in range(20):  # steps per episode
                state = (x, y)
                if random.random() < epsilon:
                    action = random.choice(actions)
                else:
                    action = np.argmax(q_table[state])
                new_x, new_y = take_action(x, y, action)
                reward = get_reward(new_x, new_y, genre_counts)
                next_state = (new_x, new_y)
                q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
                x, y = new_x, new_y

        best_states = sorted(q_table.items(), key=lambda x: np.max(x[1]), reverse=True)[:10]
        top_genres = {genre_grid[s[0][0], s[0][1]] for s in best_states}

        recommended = test_movies[test_movies['genres'].apply(lambda gs: any(g in top_genres for g in gs))]

        recall = len(recommended) / len(test_movies) if len(test_movies) > 0 else 0
        recalls.append(recall)

    avg_recall = np.mean(recalls)
    recall_progress.append(avg_recall)

    if epoch_limit == total_epochs:
        recalls_after_training = recalls

random_user = random.choice(first_10_users)  # Pick a random user from the first 10 users
random_user_data = ratings[ratings['userId'] == random_user].sort_values('timestamp')
train_data, test_data = train_test_split(random_user_data, test_size=0.3, random_state=42)

test_movie_ids = set(test_data['movieId'])
test_movies = movies[movies['movieId'].isin(test_movie_ids)]

# Genre counts for the random user
user_movies = movies[movies['movieId'].isin(set(train_data['movieId']))]
user_genre_counts = Counter(g for g_list in user_movies['genres'] for g in g_list)


q_matrix_random = defaultdict(lambda: np.zeros(4))

for _ in range(total_epochs):
    x, y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
    for _ in range(20):
        state = (x, y)
        action = random.choice(actions) if random.random() < epsilon else np.argmax(q_matrix_random[state])
        new_x, new_y = take_action(x, y, action)
        reward = get_reward(new_x, new_y, user_genre_counts)
        next_state = (new_x, new_y)
        q_matrix_random[state][action] += alpha * (reward + gamma * np.max(q_matrix_random[next_state]) - q_matrix_random[state][action])
        x, y = new_x, new_y


best_states_random = sorted(q_matrix_random.items(), key=lambda x: np.max(x[1]), reverse=True)[:10]
top_genres_random = {genre_grid[s[0][0], s[0][1]] for s in best_states_random}

recommended_random = test_movies[test_movies['genres'].apply(lambda gs: any(g in top_genres_random for g in gs))]
recommended_random = recommended_random.head(4)

print(f"\nRecommended Movies for Random User {random_user} at Last Epoch:")
for idx, row in recommended_random.iterrows():
    movie_id = row['movieId']
    movie_name = row['title']
    print(f"Movie ID: {movie_id}, Movie Name: {movie_name}")

selected_user = eligible_users[0]  
print(f"Visualizing value matrix for User {selected_user}")

user_data = ratings[ratings['userId'] == selected_user].sort_values('timestamp')
train_data, test_data = train_test_split(user_data, test_size=0.3, random_state=42)
user_movies = movies[movies['movieId'].isin(set(train_data['movieId']))]
user_genre_counts = Counter(g for g_list in user_movies['genres'] for g in g_list)

q_matrix = defaultdict(lambda: np.zeros(4))
for _ in range(total_epochs):
    x, y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
    for _ in range(20):
        state = (x, y)
        action = random.choice(actions) if random.random() < epsilon else np.argmax(q_matrix[state])
        new_x, new_y = take_action(x, y, action)
        reward = get_reward(new_x, new_y, user_genre_counts)
        next_state = (new_x, new_y)
        q_matrix[state][action] += alpha * (reward + gamma * np.max(q_matrix[next_state]) - q_matrix[state][action])
        x, y = new_x, new_y


value_matrix = np.zeros((grid_size, grid_size))
for i in range(grid_size):
    for j in range(grid_size):
        state = (i, j)
        value_matrix[i][j] = np.max(q_matrix[state]) if state in q_matrix else 0



plt.figure(figsize=(10, 8))
sns.heatmap(value_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title(f"Value Matrix for User {selected_user}")
plt.xlabel("Column")
plt.ylabel("Row")
plt.show()

print(f"\nTop valuable states for User {selected_user}:")
best_states = sorted(q_matrix.items(), key=lambda x: np.max(x[1]), reverse=True)[:5]
for state, values in best_states:
    genre = genre_grid[state[0], state[1]]
    print(f"Position {state}: Value {np.max(values):.2f}, Genre: {genre}")
    

plt.figure(figsize=(10, 5))
plt.plot(range(10, total_epochs + 1, 10), recall_progress, marker='o')
plt.title('Average Recall@10 Every 10 Epochs')
plt.xlabel('Epochs')
plt.ylabel('Average Recall@10')
plt.xticks(range(10, total_epochs + 1, 10))
plt.grid(True)
plt.show()

print("\nFinal Recall@10 for First 10 Users after 100 Epochs:\n")
for i, recall in enumerate(recalls_after_training, 1):
    print(f"User {i}: {recall:.2f}")
