from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.smart_recommendation_model import SmartClothingRecommendationSystem
import pandas as pd
import json

app = Flask(__name__)
CORS(app)

# Global variables
smart_rec_system = None
processed_df = None
user_preferences = {}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Smart API is running'})

@app.route('/initialize', methods=['POST'])
def initialize_model():
    global smart_rec_system, processed_df
    
    try:
        # Get CSV path from request or use default
        data = request.get_json() if request.is_json else {}
        csv_path = data.get('csv_path', 'data/raw/clothing_dataset.csv')
        
        print(f"Loading dataset from: {csv_path}")
        
        # Check if file exists
        if not os.path.exists(csv_path):
            return jsonify({
                'status': 'error', 
                'message': f'CSV file not found at {csv_path}'
            })
        
        # Load and process data
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} items from CSV")
        
        smart_rec_system = SmartClothingRecommendationSystem()
        processed_df = smart_rec_system.preprocess_data(df)
        print(f"Processed {len(processed_df)} items")
        
        # Initialize smart model with embeddings
        print("Initializing Sentence-BERT model...")
        result = smart_rec_system.initialize_model(processed_df)
        
        return jsonify({
            'status': 'success',
            'message': 'Smart model initialized successfully!',
            'total_items': len(processed_df),
            'categories': processed_df['category'].value_counts().to_dict(),
            'embedding_dimension': result.get('embedding_dimension', 'unknown')
        })
    
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/items', methods=['GET'])
def get_all_items():
    global processed_df
    
    if processed_df is None:
        return jsonify({'status': 'error', 'message': 'Model not initialized'})
    
    try:
        # Convert processed_df to list of dictionaries for frontend
        items = []
        for idx, row in processed_df.iterrows():
            item = {
                'item_id': int(row['item_id']),
                'name': row['name'],
                'category': row['category'],
                'price': row['price'],
                'price_numeric': float(row['price_numeric']),
                'price_range': row['price_range'],
                'color_from_name': row['color_from_name'],
                'num_colors': int(row['num_colors']),
                'style_features': row['style_features'],
                'image_url': row['image_url']
            }
            items.append(item)
        
        return jsonify({
            'status': 'success',
            'items': items,
            'total_count': len(items)
        })
    
    except Exception as e:
        print(f"Error getting items: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/recommend', methods=['POST'])
def get_smart_recommendations():
    global smart_rec_system, user_preferences
    
    if smart_rec_system is None or not smart_rec_system.is_initialized:
        return jsonify({'status': 'error', 'message': 'Smart model not initialized'})
    
    try:
        data = request.get_json()
        user_id = data.get('user_id', 0)
        top_k = data.get('top_k', 20)
        
        print(f"Getting smart recommendations for user {user_id}")
        
        # Get user preferences
        user_prefs = user_preferences.get(user_id, {
            'liked_items': [],
            'disliked_items': [],
            'interactions': []
        })
        
        # Get smart recommendations using sentence embeddings
        recommendations = smart_rec_system.get_smart_recommendations(user_prefs, top_k)
        
        print(f"Generated {len(recommendations)} smart recommendations")
        
        return jsonify({
            'status': 'success',
            'recommendations': recommendations,
            'user_id': user_id,
            'method': 'sentence-bert',
            'total_likes': len(user_prefs['liked_items']),
            'total_dislikes': len(user_prefs['disliked_items'])
        })
    
    except Exception as e:
        print(f"Error getting smart recommendations: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/similar-items', methods=['POST'])
def get_similar_items():
    global smart_rec_system, user_preferences
    
    if smart_rec_system is None or not smart_rec_system.is_initialized:
        return jsonify({'status': 'error', 'message': 'Smart model not initialized'})
    
    try:
        data = request.get_json()
        item_id = data.get('item_id')
        user_id = data.get('user_id', 0)
        top_k = data.get('top_k', 10)
        
        if item_id is None:
            return jsonify({'status': 'error', 'message': 'item_id required'})
        
        # Get user's seen items to exclude
        user_prefs = user_preferences.get(user_id, {})
        seen_items = user_prefs.get('liked_items', []) + user_prefs.get('disliked_items', [])
        
        # Get similar items
        similar_items = smart_rec_system.find_similar_items(item_id, seen_items, top_k)
        
        return jsonify({
            'status': 'success',
            'similar_items': similar_items,
            'target_item_id': item_id
        })
    
    except Exception as e:
        print(f"Error getting similar items: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/feedback', methods=['POST'])
def record_feedback():
    global user_preferences, smart_rec_system
    
    try:
        data = request.get_json()
        user_id = data.get('user_id', 0)
        item_id = data.get('item_id')
        liked = data.get('liked', False)
        
        # Initialize user preferences if not exists
        if user_id not in user_preferences:
            user_preferences[user_id] = {
                'interactions': [],
                'liked_items': [],
                'disliked_items': []
            }
        
        # Record interaction
        interaction = {
            'item_id': item_id,
            'liked': liked,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        user_preferences[user_id]['interactions'].append(interaction)
        
        if liked:
            user_preferences[user_id]['liked_items'].append(item_id)
        else:
            user_preferences[user_id]['disliked_items'].append(item_id)
        
        print(f"Recorded feedback: User {user_id}, Item {item_id}, Liked: {liked}")
        
        # Get item insights if smart model is ready
        insights = {}
        if smart_rec_system and smart_rec_system.is_initialized and item_id is not None:
            try:
                insights = smart_rec_system.get_item_insights(item_id)
            except Exception as e:
                print(f"Error getting insights: {e}")
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback recorded successfully',
            'total_interactions': len(user_preferences[user_id]['interactions']),
            'insights': insights
        })
    
    except Exception as e:
        print(f"Error recording feedback: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/user-stats', methods=['GET'])
def get_user_stats():
    global user_preferences
    
    try:
        user_id = request.args.get('user_id', 0, type=int)
        
        if user_id not in user_preferences:
            return jsonify({
                'status': 'success',
                'stats': {
                    'total_interactions': 0,
                    'liked_count': 0,
                    'disliked_count': 0,
                    'like_ratio': 0
                }
            })
        
        user_data = user_preferences[user_id]
        total_interactions = len(user_data['interactions'])
        liked_count = len(user_data['liked_items'])
        disliked_count = len(user_data['disliked_items'])
        like_ratio = liked_count / total_interactions if total_interactions > 0 else 0
        
        return jsonify({
            'status': 'success',
            'stats': {
                'total_interactions': total_interactions,
                'liked_count': liked_count,
                'disliked_count': disliked_count,
                'like_ratio': round(like_ratio, 2)
            }
        })
    
    except Exception as e:
        print(f"Error getting user stats: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/user-preferences', methods=['GET'])
def get_user_preferences():
    global user_preferences, processed_df
    
    try:
        user_id = request.args.get('user_id', 0, type=int)
        
        if user_id not in user_preferences or processed_df is None:
            return jsonify({
                'status': 'success',
                'preferences': {
                    'top_categories': [],
                    'top_colors': [],
                    'top_price_ranges': []
                }
            })
        
        user_data = user_preferences[user_id]
        liked_items = user_data['liked_items']
        disliked_items = user_data['disliked_items']
        
        if len(liked_items) < 2:
            return jsonify({
                'status': 'success',
                'preferences': {
                    'top_categories': [],
                    'top_colors': [],
                    'top_price_ranges': []
                }
            })
        
        # Get liked and disliked item details
        liked_data = processed_df[processed_df['item_id'].isin(liked_items)]
        disliked_data = processed_df[processed_df['item_id'].isin(disliked_items)]
        
        if len(liked_data) == 0:
            return jsonify({
                'status': 'success',
                'preferences': {
                    'top_categories': [],
                    'top_colors': [],
                    'top_price_ranges': []
                }
            })
        
        # Count preferences
        liked_categories = liked_data['category'].value_counts()
        liked_colors = liked_data['color_from_name'].value_counts() 
        liked_prices = liked_data['price_range'].value_counts()
        
        # Get top preferences
        top_categories = liked_categories.head(3).index.tolist()
        top_colors = liked_colors.head(3).index.tolist()
        top_price_ranges = liked_prices.head(2).index.tolist()
        
        return jsonify({
            'status': 'success',
            'preferences': {
                'top_categories': top_categories,
                'top_colors': top_colors,
                'top_price_ranges': top_price_ranges,
                'total_likes': len(liked_items),
                'total_dislikes': len(disliked_items)
            }
        })
        
    except Exception as e:
        print(f"Error getting user preferences: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    print("Starting Smart Clothing Recommendation API...")
    print("Using Sentence-BERT for semantic understanding")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /initialize - Initialize smart model")
    print("  GET  /items - Get all items")
    print("  POST /recommend - Get smart recommendations")
    print("  POST /similar-items - Get similar items")
    print("  POST /feedback - Record user feedback")
    print("  GET  /user-stats - Get user statistics")
    print("  GET  /user-preferences - Get user preferences")
    
    app.run(debug=True, port=5000, host='0.0.0.0')