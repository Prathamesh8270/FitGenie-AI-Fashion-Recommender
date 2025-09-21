import React, { useState, useEffect } from 'react';
import { ThumbsDown, ThumbsUp, Loader } from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://your-api-domain.com' 
  : 'http://localhost:5000';

const RecommendationApp = () => {
  // State management
  const [isModelInitialized, setIsModelInitialized] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [items, setItems] = useState([]);
  const [currentItemIndex, setCurrentItemIndex] = useState(0);
  const [userStats, setUserStats] = useState({
    total_interactions: 0,
    liked_count: 0,
    disliked_count: 0,
    like_ratio: 0
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const userId = 0; // Fixed user ID for demo

  // Initialize model and load items
// REPLACE the initializeModel function
  const initializeModel = async () => {
    setIsInitializing(true);
    setError('');
    setSuccess('');
    
    try {
      // Step 1: Initialize the smart model with Sentence-BERT
      setSuccess('Loading Sentence-BERT model... (10-15 seconds)');
      const initResponse = await axios.post(`${API_BASE_URL}/initialize`, {
        csv_path: 'data/raw/clothing_dataset.csv'
      });
      
      if (initResponse.data.status === 'success') {
        setSuccess('Smart AI model loaded! Getting your items...');
        
        // Step 2: Load all items
        const itemsResponse = await axios.get(`${API_BASE_URL}/items`);
        
        if (itemsResponse.data.status === 'success') {
          // Shuffle items immediately on load for variety
          const shuffledItems = itemsResponse.data.items.sort(() => Math.random() - 0.5);
          setItems(shuffledItems);
          setIsModelInitialized(true);
          setSuccess(`🚀 Ready! Smart AI loaded ${shuffledItems.length} items with semantic understanding!`);
          
          // Clear success message after 4 seconds
          setTimeout(() => setSuccess(''), 4000);
        }
      }
    } catch (err) {
      console.error('Initialization error:', err);
      if (err.response?.data?.message) {
        setError(`Error: ${err.response.data.message}`);
      } else if (err.code === 'ECONNREFUSED') {
        setError('Cannot connect to backend server. Make sure the Flask server is running on port 5000.');
      } else {
        setError('Failed to initialize smart model. Please try again.');
      }
    } finally {
      setIsInitializing(false);
    }
  };

  // ADD THIS NEW FUNCTION after getRecommendations()
  const getSmartRecommendations = async () => {
    try {
      // Get user's interaction history from stats
      const statsResponse = await axios.get(`${API_BASE_URL}/user-stats`, {
        params: { user_id: userId }
      });
      
      if (statsResponse.data.status === 'success' && statsResponse.data.stats.total_interactions > 5) {
        // Get AI-powered recommendations from backend
        const response = await axios.post(`${API_BASE_URL}/recommend`, {
          user_id: userId,
          top_k: Math.min(30, items.length)
        });
        
        if (response.data.status === 'success') {
          const recommendedIds = response.data.recommendations.map(r => r.item_id);
          const recommendedItems = response.data.recommendations.map(rec => {
            return items.find(item => item.item_id === rec.item_id);
          }).filter(Boolean);
          
          const remainingItems = items.filter(item => 
            !recommendedIds.includes(item.item_id)
          );
          
          // Shuffle both arrays for variety
          const shuffledRecommended = recommendedItems.sort(() => Math.random() - 0.5);
          const shuffledRemaining = remainingItems.sort(() => Math.random() - 0.5);
          
          setItems([...shuffledRecommended, ...shuffledRemaining]);
          setCurrentItemIndex(0);
          setSuccess('AI found better matches for you!');
          setTimeout(() => setSuccess(''), 2000);
          return;
        }
      }
      
      // Fallback: Just shuffle and restart
      const shuffledItems = [...items].sort(() => Math.random() - 0.5);
      setItems(shuffledItems);
      setCurrentItemIndex(0);
      
    } catch (err) {
      console.error('Error getting smart recommendations:', err);
      // Fallback: Just shuffle and restart
      const shuffledItems = [...items].sort(() => Math.random() - 0.5);
      setItems(shuffledItems);
      setCurrentItemIndex(0);
    }
  };

  // Handle swipe action
// COMPLETELY REPLACE the handleSwipe function
  const handleSwipe = async (liked) => {
    if (items.length === 0) return;
    
    const currentItem = items[currentItemIndex];
    setLoading(true);
    
    try {
      // Record feedback
      await axios.post(`${API_BASE_URL}/feedback`, {
        user_id: userId,
        item_id: currentItem.item_id,
        liked: liked
      });
      
      // Update stats
      await updateUserStats();
      
      // AGGRESSIVE LEARNING - Immediate filtering
      const remainingItems = items.slice(currentItemIndex + 1);
      const seenItems = items.slice(0, currentItemIndex + 1);
      
      if (liked) {
        // LIKE: Prioritize exact matches heavily
        const exactMatches = remainingItems.filter(item => 
          item.category === currentItem.category &&
          item.color_from_name === currentItem.color_from_name
        );
        
        const categoryMatches = remainingItems.filter(item => 
          item.category === currentItem.category &&
          item.color_from_name !== currentItem.color_from_name
        );
        
        const colorMatches = remainingItems.filter(item => 
          item.category !== currentItem.category &&
          item.color_from_name === currentItem.color_from_name
        );
        
        const priceMatches = remainingItems.filter(item => 
          item.price_range === currentItem.price_range &&
          item.category !== currentItem.category &&
          item.color_from_name !== currentItem.color_from_name
        );
        
        const others = remainingItems.filter(item => 
          item.category !== currentItem.category &&
          item.color_from_name !== currentItem.color_from_name &&
          item.price_range !== currentItem.price_range
        );
        
        // Shuffle each group
        const shuffledExact = exactMatches.sort(() => Math.random() - 0.5);
        const shuffledCategory = categoryMatches.sort(() => Math.random() - 0.5);
        const shuffledColor = colorMatches.sort(() => Math.random() - 0.5);
        const shuffledPrice = priceMatches.sort(() => Math.random() - 0.5);
        const shuffledOthers = others.sort(() => Math.random() - 0.5);
        
        // AGGRESSIVE PRIORITIZATION: 50% exact matches, 30% similar, 20% diverse
        const totalRemaining = remainingItems.length;
        const exactCount = Math.min(shuffledExact.length, Math.floor(totalRemaining * 0.5));
        const categoryCount = Math.min(shuffledCategory.length, Math.floor(totalRemaining * 0.2));
        const colorCount = Math.min(shuffledColor.length, Math.floor(totalRemaining * 0.1));
        const priceCount = Math.min(shuffledPrice.length, Math.floor(totalRemaining * 0.1));
        
        const reorderedItems = [
          ...shuffledExact.slice(0, exactCount),
          ...shuffledCategory.slice(0, categoryCount),
          ...shuffledColor.slice(0, colorCount),
          ...shuffledPrice.slice(0, priceCount),
          ...shuffledOthers,
          ...shuffledExact.slice(exactCount),
          ...shuffledCategory.slice(categoryCount),
          ...shuffledColor.slice(colorCount),
          ...shuffledPrice.slice(priceCount)
        ];
        
        setItems([...seenItems, ...reorderedItems]);
        
      } else {
        // DISLIKE: Heavily penalize similar items
        const avoidExact = remainingItems.filter(item => 
          item.category === currentItem.category &&
          item.color_from_name === currentItem.color_from_name
        );
        
        const avoidCategory = remainingItems.filter(item => 
          item.category === currentItem.category
        );
        
        const avoidColor = remainingItems.filter(item => 
          item.color_from_name === currentItem.color_from_name
        );
        
        const preferred = remainingItems.filter(item => 
          item.category !== currentItem.category &&
          item.color_from_name !== currentItem.color_from_name &&
          item.price_range !== currentItem.price_range
        );
        
        const neutral = remainingItems.filter(item => 
          item.category !== currentItem.category &&
          item.color_from_name !== currentItem.color_from_name &&
          item.price_range === currentItem.price_range
        );
        
        // Shuffle groups
        const shuffledPreferred = preferred.sort(() => Math.random() - 0.5);
        const shuffledNeutral = neutral.sort(() => Math.random() - 0.5);
        const shuffledAvoidColor = avoidColor.sort(() => Math.random() - 0.5);
        const shuffledAvoidCategory = avoidCategory.sort(() => Math.random() - 0.5);
        const shuffledAvoidExact = avoidExact.sort(() => Math.random() - 0.5);
        
        // HEAVY AVOIDANCE: Show preferred first, avoided items last
        const reorderedItems = [
          ...shuffledPreferred,
          ...shuffledNeutral,
          ...shuffledAvoidColor.slice(0, Math.floor(avoidColor.length * 0.3)),
          ...shuffledAvoidCategory.slice(0, Math.floor(avoidCategory.length * 0.2)),
          ...shuffledAvoidExact.slice(0, Math.floor(avoidExact.length * 0.1)),
          ...shuffledAvoidColor.slice(Math.floor(avoidColor.length * 0.3)),
          ...shuffledAvoidCategory.slice(Math.floor(avoidCategory.length * 0.2)),
          ...shuffledAvoidExact.slice(Math.floor(avoidExact.length * 0.1))
        ];
        
        setItems([...seenItems, ...reorderedItems]);
      }
      
      // Move to next item
      if (currentItemIndex < items.length - 1) {
        setCurrentItemIndex(prev => prev + 1);
      } else {
        // Get smart recommendations when finished
        await getAggressiveRecommendations();
      }
      
    } catch (err) {
      console.error('Error:', err);
      setError('Failed to record feedback');
    } finally {
      setLoading(false);
    }
  };

// ADD this new function after other functions
  const getAggressiveRecommendations = async () => {
    try {
      const statsResponse = await axios.get(`${API_BASE_URL}/user-stats`, {
        params: { user_id: userId }
      });
      
      if (statsResponse.data.status === 'success' && statsResponse.data.stats.total_interactions > 3) {
        const prefsResponse = await axios.get(`${API_BASE_URL}/user-preferences`, {
          params: { user_id: userId }
        });
        
        if (prefsResponse.data.status === 'success') {
          const preferences = prefsResponse.data.preferences;
          const topCategories = preferences.top_categories;
          const topColors = preferences.top_colors;
          const topPriceRanges = preferences.top_price_ranges;
          
          // AGGRESSIVE FILTERING based on learned preferences
          const perfectMatches = items.filter(item => 
            topCategories.includes(item.category) &&
            topColors.includes(item.color_from_name) &&
            topPriceRanges.includes(item.price_range)
          );
          
          const goodMatches = items.filter(item => 
            (topCategories.includes(item.category) && topColors.includes(item.color_from_name)) ||
            (topCategories.includes(item.category) && topPriceRanges.includes(item.price_range)) ||
            (topColors.includes(item.color_from_name) && topPriceRanges.includes(item.price_range))
          );
          
          const okMatches = items.filter(item => 
            topCategories.includes(item.category) ||
            topColors.includes(item.color_from_name) ||
            topPriceRanges.includes(item.price_range)
          );
          
          const others = items.filter(item => 
            !topCategories.includes(item.category) &&
            !topColors.includes(item.color_from_name) &&
            !topPriceRanges.includes(item.price_range)
          );
          
          // Remove duplicates and shuffle each group
          const uniquePerfect = [...new Set(perfectMatches.map(i => i.item_id))].map(id => 
            perfectMatches.find(i => i.item_id === id)
          ).sort(() => Math.random() - 0.5);
          
          const uniqueGood = [...new Set(goodMatches.map(i => i.item_id))].map(id => 
            goodMatches.find(i => i.item_id === id)
          ).filter(item => !uniquePerfect.find(p => p.item_id === item.item_id)).sort(() => Math.random() - 0.5);
          
          const uniqueOk = [...new Set(okMatches.map(i => i.item_id))].map(id => 
            okMatches.find(i => i.item_id === id)
          ).filter(item => 
            !uniquePerfect.find(p => p.item_id === item.item_id) &&
            !uniqueGood.find(g => g.item_id === item.item_id)
          ).sort(() => Math.random() - 0.5);
          
          const uniqueOthers = [...new Set(others.map(i => i.item_id))].map(id => 
            others.find(i => i.item_id === id)
          ).sort(() => Math.random() - 0.5);
          
          // SUPER AGGRESSIVE: 70% perfect matches, 20% good, 10% diversity
          const totalItems = items.length;
          const perfectCount = Math.min(uniquePerfect.length, Math.floor(totalItems * 0.7));
          const goodCount = Math.min(uniqueGood.length, Math.floor(totalItems * 0.2));
          const okCount = Math.min(uniqueOk.length, Math.floor(totalItems * 0.05));
          const diversityCount = Math.min(uniqueOthers.length, Math.floor(totalItems * 0.05));
          
          const finalOrder = [
            ...uniquePerfect.slice(0, perfectCount),
            ...uniqueGood.slice(0, goodCount),
            ...uniqueOk.slice(0, okCount),
            ...uniqueOthers.slice(0, diversityCount),
            ...uniquePerfect.slice(perfectCount),
            ...uniqueGood.slice(goodCount),
            ...uniqueOk.slice(okCount),
            ...uniqueOthers.slice(diversityCount)
          ];
          
          setItems(finalOrder);
          setCurrentItemIndex(0);
          setSuccess(`Found ${perfectCount} perfect matches for your style!`);
          setTimeout(() => setSuccess(''), 2000);
          return;
        }
      }
      
      // Fallback: shuffle and restart
      const shuffled = [...items].sort(() => Math.random() - 0.5);
      setItems(shuffled);
      setCurrentItemIndex(0);
      
    } catch (err) {
      console.error('Error getting recommendations:', err);
      const shuffled = [...items].sort(() => Math.random() - 0.5);
      setItems(shuffled);
      setCurrentItemIndex(0);
    }
  };
  
  // Get personalized recommendations
  const getRecommendations = async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/recommend`, {
        user_id: userId,
        top_k: Math.min(50, items.length) // Get up to 50 recommendations
      });
      
      if (response.data.status === 'success') {
        // Create new item list with recommendations first, then remaining items
        const recommendedIds = response.data.recommendations.map(r => r.item_id);
        const recommendedItems = response.data.recommendations.map(rec => {
          // Find the full item data
          return items.find(item => item.item_id === rec.item_id);
        }).filter(Boolean);
        
        const remainingItems = items.filter(item => 
          !recommendedIds.includes(item.item_id)
        );
        
        setItems([...recommendedItems, ...remainingItems]);
        setCurrentItemIndex(0);
      }
    } catch (err) {
      console.error('Error getting recommendations:', err);
      // If recommendations fail, just restart from beginning
      setCurrentItemIndex(0);
    }
  };

  // Update user statistics
  const updateUserStats = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/user-stats`, {
        params: { user_id: userId }
      });
      
      if (response.data.status === 'success') {
        setUserStats(response.data.stats);
      }
    } catch (err) {
      console.error('Error updating user stats:', err);
    }
  };

  // Check backend health on component mount
  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        await axios.get(`${API_BASE_URL}/health`);
      } catch (err) {
        setError('Backend server is not running. Please start the Flask server first.');
      }
    };
    
    checkBackendHealth();
  }, []);

  const currentItem = items[currentItemIndex];

  // Render initialization screen
  if (!isModelInitialized) {
    return (
      <div className="recommendation-container">
        <div className="recommendation-card">
          <div className="card-header">
            <h1>Clothing Recommendation System</h1>
            <p>Deep Learning Powered Fashion AI</p>
          </div>
          
          <div className="initialize-section">
            {error && (
              <div className="error-message">
                {error}
              </div>
            )}
            
            {success && (
              <div className="success-message">
                {success}
              </div>
            )}
            
            {!isInitializing ? (
              <div>
                <h3>Ready for Smart AI Fashion Recommendations?</h3>
                <p style={{ margin: '20px 0', color: '#666' }}>
                  Our system uses advanced Sentence-BERT technology to understand fashion semantics and provide intelligent recommendations from day one.
                </p>
                <button
                  onClick={initializeModel}
                  className="initialize-button"
                  disabled={isInitializing}
                >
                  Load Smart AI Model
                </button>
              </div>
            ) : (
              <div className="loading-spinner">
                <div className="spinner"></div>
                <p>Loading Sentence-BERT model...</p>
                <p style={{ fontSize: '12px', color: '#888' }}>
                  Setting up semantic understanding (10-15 seconds)
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Render main app
  return (
    <div className="recommendation-container">
      <div className="recommendation-card">
        {/* Header */}
        <div className="card-header">
          <h1>FitGenie - Fashion AI Recommender</h1>
          <p>Item {currentItemIndex + 1} of {items.length}</p>
        </div>

        {/* Item Display */}
        <div className="card-content">
          {loading && (
            <div className="loading-spinner">
              <Loader className="spinner" size={24} />
            </div>
          )}
          
          {!loading && currentItem && (
            <div>
              <img 
                src={currentItem.image_url} 
                alt={currentItem.name}
                className="item-image"
                onError={(e) => {
                  e.target.src = `https://via.placeholder.com/280x350/cccccc/666666?text=${encodeURIComponent(currentItem.category.toUpperCase())}`;
                }}
              />
              
              <div className="item-name">{currentItem.name}</div>
              <div className="item-price">{currentItem.price}</div>
              <div className="item-details">
                {currentItem.category} • {currentItem.color_from_name} • {currentItem.price_range}
              </div>
              
              {currentItem.style_features && currentItem.style_features !== 'basic' && (
                <div className="style-tags">
                  {currentItem.style_features.split(',').join(' • ')}
                </div>
              )}
            </div>
          )}
          
          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </div>

        {/* Swipe Buttons */}
        <div className="swipe-buttons">
          <button
            onClick={() => handleSwipe(false)}
            className="swipe-button dislike-button"
            disabled={loading || !currentItem}
          >
            <ThumbsDown size={24} />
          </button>
          
          <button
            onClick={() => handleSwipe(true)}
            className="swipe-button like-button"
            disabled={loading || !currentItem}
          >
            <ThumbsUp size={24} />
          </button>
        </div>
        
        <div className="button-labels">
          <span>Dislike</span>
          <span>Like</span>
        </div>

        {/* User Statistics */}
        {userStats.total_interactions > 0 && (
          <div className="stats-section">
            <h4>AI Learning Progress</h4>
            <div style={{ marginTop: '10px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '11px' }}>
              <div>Total Swipes: {userStats.total_interactions}</div>
              <div>Liked: {userStats.liked_count}</div>
              <div>Disliked: {userStats.disliked_count}</div>
              <div>Like Rate: {(userStats.like_ratio * 100).toFixed(1)}%</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default RecommendationApp;