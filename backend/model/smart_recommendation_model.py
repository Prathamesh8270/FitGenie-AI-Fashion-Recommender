import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

class SmartClothingRecommendationSystem:
    def __init__(self):
        self.sentence_model = None
        self.item_embeddings = None
        self.processed_df = None
        self.is_initialized = False
        
    def preprocess_data(self, df):
        """Preprocess the clothing dataset"""
        processed_df = df.copy()
        
        # Drop irrelevant columns
        columns_to_drop = ['current_color', 'sale_price', 'badge', 'sizes_available', 
                          'rating', 'review_count', 'product_url', 'scraped_at']
        processed_df = processed_df.drop(columns=[col for col in columns_to_drop if col in processed_df.columns])
        
        # Extract features from name
        processed_df['category'] = processed_df['name'].apply(self._extract_category)
        processed_df['style_features'] = processed_df['name'].apply(self._extract_style_features)
        processed_df['color_from_name'] = processed_df['name'].apply(self._extract_color_from_name)
        
        # Process price
        processed_df['price_numeric'] = processed_df['price'].apply(self._extract_price)
        processed_df['price_range'] = processed_df['price_numeric'].apply(self._categorize_price)
        
        # Process colors available
        processed_df['num_colors'] = processed_df['colors_available'].apply(self._count_colors)
        
        # Create item embeddings features
        processed_df['item_id'] = range(len(processed_df))
        
        # Create rich text description for embeddings
        processed_df['embedding_text'] = processed_df.apply(self._create_embedding_text, axis=1)
        
        return processed_df
    
    def _create_embedding_text(self, row):
        """Create rich text description for each item"""
        parts = []
        
        # Add name
        parts.append(row['name'])
        
        # Add category
        parts.append(f"category {row['category']}")
        
        # Add color
        parts.append(f"color {row['color_from_name']}")
        
        # Add price range
        parts.append(f"price {row['price_range']}")
        
        # Add style features
        if row['style_features'] and row['style_features'] != 'basic':
            styles = row['style_features'].replace(',', ' ')
            parts.append(f"style {styles}")
        
        return ' '.join(parts)
    
    def initialize_model(self, processed_df):
        """Initialize the sentence transformer model and create embeddings"""
        print("Loading Sentence-BERT model...")
        
        # Load pretrained sentence transformer (lightweight and fast)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.processed_df = processed_df
        
        print("Creating item embeddings...")
        
        # Create embeddings for all items
        embedding_texts = processed_df['embedding_text'].tolist()
        self.item_embeddings = self.sentence_model.encode(embedding_texts, show_progress_bar=True)
        
        print(f"Created embeddings for {len(self.item_embeddings)} items")
        self.is_initialized = True
        
        return {
            'status': 'success',
            'message': 'Smart model initialized successfully',
            'total_items': len(processed_df),
            'embedding_dimension': self.item_embeddings.shape[1]
        }
    
    def get_smart_recommendations(self, user_preferences, top_k=10):
        """Get recommendations based on user preferences and semantic similarity"""
        if not self.is_initialized:
            return []
        
        liked_items = user_preferences.get('liked_items', [])
        disliked_items = user_preferences.get('disliked_items', [])
        
        if len(liked_items) == 0:
            # No preferences yet - return diverse sample
            indices = np.random.choice(len(self.processed_df), size=min(top_k, len(self.processed_df)), replace=False)
            return self._format_recommendations(indices, [0.5] * len(indices))
        
        # Calculate similarity scores for all items
        item_scores = np.zeros(len(self.processed_df))
        
        # Positive signals from liked items
        for liked_id in liked_items:
            if liked_id < len(self.item_embeddings):
                liked_embedding = self.item_embeddings[liked_id].reshape(1, -1)
                similarities = cosine_similarity(liked_embedding, self.item_embeddings)[0]
                item_scores += similarities * 1.5  # Boost liked similarities
        
        # Negative signals from disliked items
        for disliked_id in disliked_items:
            if disliked_id < len(self.item_embeddings):
                disliked_embedding = self.item_embeddings[disliked_id].reshape(1, -1)
                similarities = cosine_similarity(disliked_embedding, self.item_embeddings)[0]
                item_scores -= similarities * 1.0  # Penalize disliked similarities
        
        # Don't recommend items already seen
        seen_items = set(liked_items + disliked_items)
        for item_id in seen_items:
            if item_id < len(item_scores):
                item_scores[item_id] = -999  # Exclude seen items
        
        # Get top recommendations
        top_indices = np.argsort(item_scores)[-top_k:][::-1]
        top_scores = item_scores[top_indices]
        
        # Normalize scores to 0-1 range
        if len(top_scores) > 0:
            min_score = top_scores.min()
            max_score = top_scores.max()
            if max_score > min_score:
                top_scores = (top_scores - min_score) / (max_score - min_score)
            else:
                top_scores = np.ones_like(top_scores) * 0.5
        
        return self._format_recommendations(top_indices, top_scores)
    
    def find_similar_items(self, item_id, exclude_seen=None, top_k=20):
        """Find items similar to a specific item"""
        if not self.is_initialized or item_id >= len(self.item_embeddings):
            return []
        
        exclude_seen = exclude_seen or []
        
        # Get similarity to target item
        target_embedding = self.item_embeddings[item_id].reshape(1, -1)
        similarities = cosine_similarity(target_embedding, self.item_embeddings)[0]
        
        # Exclude the item itself and already seen items
        similarities[item_id] = -1
        for seen_id in exclude_seen:
            if seen_id < len(similarities):
                similarities[seen_id] = -1
        
        # Get most similar items
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]
        
        return self._format_recommendations(top_indices, top_scores)
    
    def _format_recommendations(self, indices, scores):
        """Format recommendations into standard format"""
        recommendations = []
        
        for idx, score in zip(indices, scores):
            if idx < len(self.processed_df):
                item = self.processed_df.iloc[idx]
                recommendations.append({
                    'item_id': int(item['item_id']),
                    'name': item['name'],
                    'score': float(score),
                    'category': item['category'],
                    'price': item['price'],
                    'image_url': item['image_url'],
                    'color_from_name': item['color_from_name'],
                    'price_range': item['price_range'],
                    'style_features': item['style_features']
                })
        
        return recommendations
    
    def get_item_insights(self, item_id):
        """Get insights about what makes an item similar to others"""
        if not self.is_initialized or item_id >= len(self.item_embeddings):
            return {}
        
        similar_items = self.find_similar_items(item_id, top_k=5)
        target_item = self.processed_df.iloc[item_id]
        
        # Analyze what makes items similar
        categories = [item['category'] for item in similar_items]
        colors = [item['color_from_name'] for item in similar_items]
        styles = []
        for item in similar_items:
            if item['style_features'] and item['style_features'] != 'basic':
                styles.extend(item['style_features'].split(','))
        
        from collections import Counter
        
        return {
            'target_item': target_item['name'],
            'similar_categories': dict(Counter(categories).most_common(3)),
            'similar_colors': dict(Counter(colors).most_common(3)),
            'similar_styles': dict(Counter([s.strip() for s in styles]).most_common(3)),
            'similarity_strength': float(np.mean([item['score'] for item in similar_items]))
        }
    
    # Helper methods from original model
    def _extract_category(self, name):
        name_lower = name.lower()
        if 'bra' in name_lower:
            return 'bra'
        elif 'legging' in name_lower:
            return 'legging'
        elif 'top' in name_lower or 'shirt' in name_lower:
            return 'top'
        elif 'dress' in name_lower:
            return 'dress'
        elif 'jacket' in name_lower:
            return 'jacket'
        elif 'pant' in name_lower:
            return 'pants'
        else:
            return 'other'
    
    def _extract_style_features(self, name):
        name_lower = name.lower()
        features = []
        
        style_keywords = ['high-waist', 'cropped', 'oversized', 'fitted', 'loose', 
                         'athletic', 'casual', 'formal', 'vintage', 'modern',
                         'airlift', 'seamless', 'ribbed', 'mesh', 'intrigue']
        
        for keyword in style_keywords:
            if keyword in name_lower:
                features.append(keyword)
        
        return ','.join(features) if features else 'basic'
    
    def _extract_color_from_name(self, name):
        colors = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 
                 'purple', 'orange', 'brown', 'grey', 'gray', 'navy', 'espresso',
                 'almond', 'rose', 'anthracite', 'olive', 'neutral']
        
        name_lower = name.lower()
        for color in colors:
            if color in name_lower:
                return color
        return 'neutral'
    
    def _extract_price(self, price_str):
        if pd.isna(price_str) or price_str == '':
            return 0
        
        price_clean = re.sub(r'[^\d.]', '', str(price_str))
        try:
            return float(price_clean) if price_clean else 0
        except:
            return 0
    
    def _categorize_price(self, price):
        if price < 50:
            return 'budget'
        elif price < 100:
            return 'mid'
        else:
            return 'premium'
    
    def _count_colors(self, colors_str):
        if pd.isna(colors_str) or colors_str == '':
            return 0
        return len(str(colors_str).split(','))