import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class ClothingDataProcessor:
    def __init__(self):
        self.category_encoder = LabelEncoder()
        self.color_encoder = LabelEncoder()
        self.price_scaler = StandardScaler()
        
    def preprocess_data(self, df):
        """Preprocess the clothing dataset"""
        # Create a copy to avoid modifying original
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
        processed_df['main_colors'] = processed_df['colors_available'].apply(self._extract_main_colors)
        
        # Create item embeddings features
        processed_df['item_id'] = range(len(processed_df))
        
        return processed_df
    
    def _extract_category(self, name):
        """Extract clothing category from name"""
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
        """Extract style features from name"""
        name_lower = name.lower()
        features = []
        
        style_keywords = ['high-waist', 'cropped', 'oversized', 'fitted', 'loose', 
                         'athletic', 'casual', 'formal', 'vintage', 'modern',
                         'airlift', 'seamless', 'ribbed', 'mesh']
        
        for keyword in style_keywords:
            if keyword in name_lower:
                features.append(keyword)
        
        return ','.join(features) if features else 'basic'
    
    def _extract_color_from_name(self, name):
        """Extract color from product name"""
        colors = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 
                 'purple', 'orange', 'brown', 'grey', 'gray', 'navy', 'espresso',
                 'almond', 'rose', 'anthracite', 'olive']
        
        name_lower = name.lower()
        for color in colors:
            if color in name_lower:
                return color
        return 'neutral'
    
    def _extract_price(self, price_str):
        """Extract numeric price from price string"""
        if pd.isna(price_str) or price_str == '':
            return 0
        
        # Remove $ and convert to float
        price_clean = re.sub(r'[^\d.]', '', str(price_str))
        try:
            return float(price_clean) if price_clean else 0
        except:
            return 0
    
    def _categorize_price(self, price):
        """Categorize price into ranges"""
        if price < 50:
            return 'budget'
        elif price < 100:
            return 'mid'
        else:
            return 'premium'
    
    def _count_colors(self, colors_str):
        """Count available colors"""
        if pd.isna(colors_str) or colors_str == '':
            return 0
        return len(colors_str.split(','))
    
    def _extract_main_colors(self, colors_str):
        """Extract main colors available"""
        if pd.isna(colors_str) or colors_str == '':
            return 'none'
        
        colors = colors_str.lower().split(',')[:3]  # Take first 3 colors
        return ','.join([c.strip() for c in colors])


class RecommendationDataset(Dataset):
    def __init__(self, user_item_interactions, item_features):
        self.interactions = user_item_interactions
        self.item_features = item_features
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        interaction = self.interactions.iloc[idx]
        
        return {
            'user_id': torch.tensor(interaction['user_id'], dtype=torch.long),
            'item_id': torch.tensor(interaction['item_id'], dtype=torch.long),
            'category': torch.tensor(interaction['category_encoded'], dtype=torch.long),
            'price_range': torch.tensor(interaction['price_range_encoded'], dtype=torch.long),
            'color': torch.tensor(interaction['color_encoded'], dtype=torch.long),
            'num_colors': torch.tensor(interaction['num_colors'], dtype=torch.float),
            'price_numeric': torch.tensor(interaction['price_numeric'], dtype=torch.float),
            'rating': torch.tensor(interaction['rating'], dtype=torch.float)
        }


class DeepRecommendationModel(nn.Module):
    def __init__(self, num_users, num_items, num_categories, num_colors, num_price_ranges,
                 embedding_dim=64, hidden_dims=[128, 64, 32]):
        super(DeepRecommendationModel, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.category_embedding = nn.Embedding(num_categories, embedding_dim//4)
        self.color_embedding = nn.Embedding(num_colors, embedding_dim//4)
        self.price_range_embedding = nn.Embedding(num_price_ranges, embedding_dim//4)
        
        # Calculate input dimension for MLP
        mlp_input_dim = (embedding_dim * 2 +  # user + item
                        embedding_dim//4 * 3 +  # category + color + price_range
                        2)  # num_colors + price_numeric
        
        # MLP layers
        layers = []
        prev_dim = mlp_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.category_embedding.weight)
        nn.init.xavier_uniform_(self.color_embedding.weight)
        nn.init.xavier_uniform_(self.price_range_embedding.weight)
    
    def forward(self, user_id, item_id, category, color, price_range, num_colors, price_numeric):
        # Get embeddings
        user_emb = self.user_embedding(user_id)
        item_emb = self.item_embedding(item_id)
        category_emb = self.category_embedding(category)
        color_emb = self.color_embedding(color)
        price_range_emb = self.price_range_embedding(price_range)
        
        # Concatenate all features
        features = torch.cat([
            user_emb, item_emb, category_emb, color_emb, price_range_emb,
            num_colors.unsqueeze(1), price_numeric.unsqueeze(1)
        ], dim=1)
        
        # Pass through MLP
        output = self.mlp(features)
        return output.squeeze()


class ClothingRecommendationSystem:
    def __init__(self):
        self.processor = ClothingDataProcessor()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def generate_synthetic_interactions(self, processed_df, num_users=50, interactions_per_user=20):
        """Generate synthetic user interactions for training"""
        interactions = []
        
        # Create user profiles with different preferences
        user_profiles = self._create_user_profiles(num_users)
        
        for user_id, profile in enumerate(user_profiles):
            user_interactions = self._generate_user_interactions(
                user_id, processed_df, profile, interactions_per_user
            )
            interactions.extend(user_interactions)
        
        return pd.DataFrame(interactions)
    
    def _create_user_profiles(self, num_users):
        """Create diverse user profiles"""
        profiles = []
        
        categories = ['bra', 'legging', 'top', 'dress', 'jacket', 'pants', 'other']
        price_preferences = ['budget', 'mid', 'premium']
        color_preferences = ['black', 'white', 'neutral', 'colorful']
        
        for _ in range(num_users):
            profile = {
                'preferred_categories': np.random.choice(categories, size=np.random.randint(1, 4), replace=False),
                'price_preference': np.random.choice(price_preferences),
                'color_preference': np.random.choice(color_preferences),
                'style_adventurousness': np.random.uniform(0, 1)
            }
            profiles.append(profile)
        
        return profiles
    
    def _generate_user_interactions(self, user_id, df, profile, num_interactions):
        """Generate interactions for a specific user based on their profile"""
        interactions = []
        
        for _ in range(num_interactions):
            # Select item based on user preferences
            item_idx = self._select_item_for_user(df, profile)
            item = df.iloc[item_idx]
            
            # Calculate probability of liking based on preferences
            like_prob = self._calculate_like_probability(item, profile)
            rating = 1 if np.random.random() < like_prob else 0
            
            interaction = {
                'user_id': user_id,
                'item_id': item_idx,
                'category_encoded': self._encode_category(item['category']),
                'price_range_encoded': self._encode_price_range(item['price_range']),
                'color_encoded': self._encode_color(item['color_from_name']),
                'num_colors': item['num_colors'],
                'price_numeric': item['price_numeric'],
                'rating': rating
            }
            interactions.append(interaction)
        
        return interactions
    
    def _select_item_for_user(self, df, profile):
        """Select item based on user profile preferences"""
        # Weight selection by user preferences
        weights = np.ones(len(df))
        
        for i, row in df.iterrows():
            weight = 1.0
            
            # Category preference
            if row['category'] in profile['preferred_categories']:
                weight *= 2.0
            
            # Price preference
            if row['price_range'] == profile['price_preference']:
                weight *= 1.5
            
            weights[i] = weight
        
        # Normalize weights
        weights = weights / weights.sum()
        return np.random.choice(len(df), p=weights)
    
    def _calculate_like_probability(self, item, profile):
        """Calculate probability user will like this item"""
        prob = 0.3  # Base probability
        
        # Category match
        if item['category'] in profile['preferred_categories']:
            prob += 0.3
        
        # Price match
        if item['price_range'] == profile['price_preference']:
            prob += 0.2
        
        # Color preference
        if profile['color_preference'] == 'colorful' and item['num_colors'] > 3:
            prob += 0.1
        elif profile['color_preference'] in ['black', 'white', 'neutral']:
            if item['color_from_name'] in ['black', 'white', 'neutral']:
                prob += 0.1
        
        # Style adventurousness
        if 'basic' not in item['style_features']:
            prob += profile['style_adventurousness'] * 0.1
        
        return min(1.0, prob)
    
    def _encode_category(self, category):
        categories = ['bra', 'legging', 'top', 'dress', 'jacket', 'pants', 'other']
        return categories.index(category) if category in categories else len(categories)
    
    def _encode_price_range(self, price_range):
        ranges = ['budget', 'mid', 'premium']
        return ranges.index(price_range) if price_range in ranges else 0
    
    def _encode_color(self, color):
        colors = ['black', 'white', 'neutral', 'red', 'blue', 'green', 'pink', 'brown', 'grey', 'navy', 'espresso']
        return colors.index(color) if color in colors else 0
    
    def train_model(self, processed_df, epochs=50, batch_size=64, learning_rate=0.001):
        """Train the deep learning recommendation model"""
        print("Generating synthetic interactions...")
        interactions_df = self.generate_synthetic_interactions(processed_df)
        
        print(f"Generated {len(interactions_df)} interactions")
        
        # Create dataset
        dataset = RecommendationDataset(interactions_df, processed_df)
        
        # Split train/test
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        num_users = interactions_df['user_id'].nunique()
        num_items = len(processed_df)
        num_categories = 8
        num_colors = 11
        num_price_ranges = 3
        
        self.model = DeepRecommendationModel(
            num_users=num_users,
            num_items=num_items,
            num_categories=num_categories,
            num_colors=num_colors,
            num_price_ranges=num_price_ranges
        ).to(self.device)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print("Starting training...")
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Move batch to device
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    batch['user_id'], batch['item_id'], batch['category'],
                    batch['color'], batch['price_range'], batch['num_colors'],
                    batch['price_numeric']
                )
                
                loss = criterion(outputs, batch['rating'])
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}')
        
        print("Training completed!")
        self._evaluate_model(test_loader)
    
    def _evaluate_model(self, test_loader):
        """Evaluate model performance"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                
                outputs = self.model(
                    batch['user_id'], batch['item_id'], batch['category'],
                    batch['color'], batch['price_range'], batch['num_colors'],
                    batch['price_numeric']
                )
                
                predicted = (outputs > 0.5).float()
                total += batch['rating'].size(0)
                correct += (predicted == batch['rating']).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
    
    def recommend_items(self, user_id, processed_df, top_k=10):
        """Generate recommendations for a user"""
        if self.model is None:
            print("Model not trained yet!")
            return []
        
        self.model.eval()
        recommendations = []
        
        with torch.no_grad():
            for idx, item in processed_df.iterrows():
                # Prepare input
                user_tensor = torch.tensor([user_id], dtype=torch.long).to(self.device)
                item_tensor = torch.tensor([idx], dtype=torch.long).to(self.device)
                category_tensor = torch.tensor([self._encode_category(item['category'])], dtype=torch.long).to(self.device)
                color_tensor = torch.tensor([self._encode_color(item['color_from_name'])], dtype=torch.long).to(self.device)
                price_range_tensor = torch.tensor([self._encode_price_range(item['price_range'])], dtype=torch.long).to(self.device)
                num_colors_tensor = torch.tensor([item['num_colors']], dtype=torch.float).to(self.device)
                price_tensor = torch.tensor([item['price_numeric']], dtype=torch.float).to(self.device)
                
                # Get prediction
                score = self.model(
                    user_tensor, item_tensor, category_tensor, color_tensor,
                    price_range_tensor, num_colors_tensor, price_tensor
                ).item()
                
                recommendations.append({
                    'item_id': idx,
                    'name': item['name'],
                    'score': score,
                    'category': item['category'],
                    'price': item['price'],
                    'image_url': item['image_url']
                })
        
        # Sort by score and return top k
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_k]