//! Intent Classification Library
//!
//! This module provides the main `IntentClassifier` struct and its implementation.
//! The classifier uses a combination of feature-based machine learning and rule-based
//! approaches to classify user intents from natural language text.

use crate::types::*;
use ahash::RandomState;
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};
use uuid::Uuid;

/// Main intent classifier that handles training and prediction
#[derive(Clone)]
pub struct IntentClassifier {
    /// Training data storage
    training_data: Arc<RwLock<Vec<TrainingExample>>>,
    
    /// Vocabulary mapping words to indices
    vocabulary: Arc<DashMap<String, usize, RandomState>>,
    
    /// Intent patterns for matching
    intent_patterns: Arc<DashMap<IntentId, Vec<String>, RandomState>>,
    
    /// Configuration for the classifier
    config: ClassifierConfig,
}

impl IntentClassifier {
    /// Create a new intent classifier with default configuration
    pub async fn new() -> Result<Self> {
        Self::with_config(ClassifierConfig::default()).await
    }
    
    /// Create a new intent classifier with custom configuration
    pub async fn with_config(config: ClassifierConfig) -> Result<Self> {
        let classifier = Self {
            training_data: Arc::new(RwLock::new(Vec::new())),
            vocabulary: Arc::new(DashMap::with_hasher(RandomState::new())),
            intent_patterns: Arc::new(DashMap::with_hasher(RandomState::new())),
            config,
        };
        
        // Load bootstrap data
        classifier.load_bootstrap_data().await?;
        
        if classifier.config.debug_mode {
            info!("Intent classifier initialized with {} dimensions", classifier.config.feature_dimensions);
        }
        
        Ok(classifier)
    }
    
    /// Predict intent from natural language text
    pub async fn predict_intent(&self, text: &str) -> Result<IntentPrediction> {
        let start_time = std::time::Instant::now();
        
        if self.config.debug_mode {
            debug!("Classifying intent for text: '{}'", text);
        }
        
        // Check for exact matches first (for high-confidence bootstrap cases)
        if let Some(exact_match) = self.find_exact_match(text).await? {
            return Ok(exact_match);
        }
        
        // Extract features from the text
        let features = self.extract_features(text).await?;
        
        // Calculate scores for all known intents
        let intent_scores = self.calculate_intent_scores(&features).await?;
        
        // Find the best intent
        let (best_intent, best_confidence) = self.find_best_intent(&intent_scores)?;
        
        // Get alternative intents
        let alternative_intents = self.get_alternative_intents(&intent_scores, &best_intent);
        
        // Generate reasoning
        let reasoning = self.generate_reasoning(text, &best_intent, &features).await;
        
        let prediction = IntentPrediction {
            intent: best_intent,
            confidence: best_confidence,
            alternative_intents,
            reasoning,
        };
        
        if self.config.debug_mode {
            let elapsed = start_time.elapsed();
            info!("Intent prediction: {} (confidence: {:.3}, time: {:?})", 
                  prediction.intent, prediction.confidence.value(), elapsed);
        }
        
        Ok(prediction)
    }
    
    /// Classify text with additional request options
    pub async fn classify(&self, request: ClassificationRequest) -> Result<ClassificationResponse> {
        let start_time = std::time::Instant::now();
        let request_id = Uuid::new_v4();
        
        let mut prediction = self.predict_intent(&request.text).await?;
        
        // Filter response based on request options
        if !request.include_alternatives {
            prediction.alternative_intents.clear();
        }
        
        if !request.include_reasoning {
            prediction.reasoning = String::new();
        }
        
        let processing_time_ms = start_time.elapsed().as_millis() as f64;
        
        Ok(ClassificationResponse {
            prediction,
            processing_time_ms,
            request_id,
        })
    }
    
    /// Add a training example
    pub async fn add_training_example(&self, example: TrainingExample) -> Result<()> {
        // Validate the example
        if example.text.trim().is_empty() {
            return Err(IntentError::InvalidParameter {
                parameter: "text".to_string(),
                message: "Training example text cannot be empty".to_string(),
            });
        }
        
        if !(0.0..=1.0).contains(&example.confidence) {
            return Err(IntentError::InvalidParameter {
                parameter: "confidence".to_string(),
                message: format!("Confidence must be between 0.0 and 1.0, got {}", example.confidence),
            });
        }
        
        // Add to training data
        {
            let mut training_data = self.training_data.write().await;
            training_data.push(example.clone());
        }
        
        // Update patterns
        self.update_intent_patterns(&example.intent, &example.text).await?;
        
        // Update vocabulary
        self.update_vocabulary(&example.text).await;
        
        if self.config.debug_mode {
            info!("Added training example: '{}' -> {}", example.text, example.intent);
        }
        
        Ok(())
    }
    
    /// Add user feedback to improve the classifier
    pub async fn add_feedback(&self, feedback: IntentFeedback) -> Result<()> {
        if self.config.debug_mode {
            info!("Adding feedback: '{}' -> {} (predicted: {}, satisfaction: {})", 
                  feedback.text, feedback.actual_intent, feedback.predicted_intent, feedback.satisfaction_score);
        }
        
        // Convert feedback to training example
        let confidence = feedback.satisfaction_score / 5.0; // Normalize to 0-1
        let example = TrainingExample {
            text: feedback.text,
            intent: feedback.actual_intent,
            confidence,
            source: TrainingSource::UserFeedback,
        };
        
        self.add_training_example(example).await?;
        
        // Check if retraining is needed
        if self.should_retrain().await {
            self.retrain().await?;
        }
        
        Ok(())
    }
    
    /// Get classifier statistics
    pub async fn get_stats(&self) -> ClassifierStats {
        let training_data = self.training_data.read().await;
        
        ClassifierStats {
            training_examples: training_data.len(),
            vocabulary_size: self.vocabulary.len(),
            intent_count: self.intent_patterns.len(),
            feedback_examples: training_data
                .iter()
                .filter(|e| matches!(e.source, TrainingSource::UserFeedback))
                .count(),
            last_updated: Some(chrono::Utc::now()),
        }
    }
    
    /// Export training data as JSON
    pub async fn export_training_data(&self) -> Result<String> {
        let training_data = self.training_data.read().await;
        serde_json::to_string_pretty(&*training_data)
            .map_err(IntentError::SerializationError)
    }
    
    /// Import training data from JSON
    pub async fn import_training_data(&self, json_data: &str) -> Result<()> {
        let examples: Vec<TrainingExample> = serde_json::from_str(json_data)
            .map_err(IntentError::SerializationError)?;
        
        for example in examples {
            self.add_training_example(example).await?;
        }
        
        Ok(())
    }
    
    /// Clear all training data
    pub async fn clear_training_data(&self) -> Result<()> {
        {
            let mut training_data = self.training_data.write().await;
            training_data.clear();
        }
        
        self.vocabulary.clear();
        self.intent_patterns.clear();
        
        // Reload bootstrap data
        self.load_bootstrap_data().await?;
        
        if self.config.debug_mode {
            info!("Cleared all training data and reloaded bootstrap data");
        }
        
        Ok(())
    }
    
    /// Find exact match in training data
    async fn find_exact_match(&self, text: &str) -> Result<Option<IntentPrediction>> {
        let training_data = self.training_data.read().await;
        
        for example in training_data.iter() {
            if example.text == text {
                let confidence = Confidence::new(example.confidence)
                    .unwrap_or_else(|_| Confidence::default());
                
                return Ok(Some(IntentPrediction {
                    intent: example.intent.clone(),
                    confidence,
                    alternative_intents: vec![],
                    reasoning: "Exact match found in training data".to_string(),
                }));
            }
        }
        
        Ok(None)
    }
    
    /// Extract features from text
    async fn extract_features(&self, text: &str) -> Result<FeatureVector> {
        let cleaned_text = self.preprocess_text(text);
        
        // Extract text features using simple bag-of-words approach
        let text_features = self.extract_text_features(&cleaned_text).await?;
        
        // Extract context features
        let context_features = self.extract_context_features(&cleaned_text);
        
        // Create metadata
        let mut metadata = HashMap::new();
        metadata.insert("text_length".to_string(), cleaned_text.len() as f64);
        metadata.insert("word_count".to_string(), cleaned_text.split_whitespace().count() as f64);
        
        Ok(FeatureVector {
            text_features,
            context_features,
            metadata,
        })
    }
    
    /// Calculate intent scores for the given features
    async fn calculate_intent_scores(&self, features: &FeatureVector) -> Result<HashMap<IntentId, f64>> {
        let mut scores = HashMap::new();
        
        for entry in self.intent_patterns.iter() {
            let (intent, pattern_texts) = (entry.key(), entry.value());
            let mut intent_score: f64 = 0.0;
            
            // Calculate similarity to known patterns
            for pattern_text in pattern_texts {
                let pattern_features = self.extract_text_features(pattern_text).await?;
                let similarity = self.cosine_similarity(&features.text_features, &pattern_features);
                intent_score = intent_score.max(similarity);
            }
            
            // Add context boost
            intent_score += self.calculate_context_boost(intent, features);
            
            scores.insert(intent.clone(), intent_score.min(1.0));
        }
        
        // Apply rule-based fallback if no good matches
        if scores.values().all(|&score| score < self.config.min_confidence_threshold) {
            let fallback_scores = self.rule_based_classification(features).await;
            for (intent, score) in fallback_scores {
                scores.entry(intent).or_insert(score);
            }
        }
        
        Ok(scores)
    }
    
    /// Find the best intent from scores
    fn find_best_intent(&self, scores: &HashMap<IntentId, f64>) -> Result<(IntentId, Confidence)> {
        let (best_intent, best_score) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| IntentError::ClassificationFailed("No intents found".to_string()))?;
        
        let confidence = Confidence::new(*best_score)
            .unwrap_or_else(|_| Confidence::default());
        
        Ok((best_intent.clone(), confidence))
    }
    
    /// Get alternative intents from scores
    fn get_alternative_intents(&self, scores: &HashMap<IntentId, f64>, best_intent: &IntentId) -> Vec<(IntentId, Confidence)> {
        let mut alternatives: Vec<(IntentId, Confidence)> = scores
            .iter()
            .filter(|(intent, _)| *intent != best_intent)
            .filter_map(|(intent, score)| {
                Confidence::new(*score)
                    .ok()
                    .map(|confidence| (intent.clone(), confidence))
            })
            .collect();
        
        // Sort by confidence descending
        alternatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        alternatives.truncate(3); // Keep top 3
        
        alternatives
    }
    
    /// Generate human-readable reasoning
    async fn generate_reasoning(&self, _text: &str, intent: &IntentId, features: &FeatureVector) -> String {
        if let Some(intent_patterns) = self.intent_patterns.get(intent) {
            if let Some(best_pattern) = intent_patterns.first() {
                return format!(
                    "Classified as '{}' based on similarity to pattern: '{}' (using {} text features)",
                    intent, best_pattern, features.text_features.len()
                );
            }
        }
        
        format!("Classified as '{}' using rule-based analysis", intent)
    }
    
    /// Load bootstrap training data
    async fn load_bootstrap_data(&self) -> Result<()> {
        let bootstrap_examples = self.get_bootstrap_examples();
        
        for (text, intent_str) in bootstrap_examples {
            let example = TrainingExample {
                text: text.to_string(),
                intent: IntentId::from(intent_str),
                confidence: 1.0,
                source: TrainingSource::Bootstrap,
            };
            
            self.add_training_example(example).await?;
        }
        
        if self.config.debug_mode {
            info!("Loaded {} bootstrap training examples", self.get_bootstrap_examples().len());
        }
        
        Ok(())
    }
    
    /// Get bootstrap examples
    fn get_bootstrap_examples(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            // Data operations
            ("merge these JSON files together", "data_merge"),
            ("combine multiple JSON documents", "data_merge"),
            ("join several data files into one", "data_merge"),
            ("consolidate JSON objects", "data_merge"),
            ("split this large JSON file", "data_split"),
            ("break apart this data into smaller pieces", "data_split"),
            ("divide this file into multiple parts", "data_split"),
            ("convert JSON to CSV format", "data_transform"),
            ("transform this data structure", "data_transform"),
            ("change the format of this file", "data_transform"),
            ("analyze this dataset for patterns", "data_analyze"),
            ("examine the data for insights", "data_analyze"),
            ("what trends do you see in this data", "data_analyze"),
            ("give me statistics about this data", "data_analyze"),
            
            // File operations
            ("read the contents of this file", "file_read"),
            ("load this document", "file_read"),
            ("open and parse this file", "file_read"),
            ("save this data to a file", "file_write"),
            ("write this content to disk", "file_write"),
            ("create a new file with this data", "file_write"),
            ("convert PDF to markdown", "file_convert"),
            ("change this file format", "file_convert"),
            ("export as different format", "file_convert"),
            ("compare these two files", "file_compare"),
            ("what's different between these documents", "file_compare"),
            ("find differences in these files", "file_compare"),
            
            // Network operations
            ("make an API request to this URL", "network_request"),
            ("call this REST endpoint", "network_request"),
            ("send HTTP request", "network_request"),
            ("download this file from the internet", "network_download"),
            ("fetch data from this URL", "network_download"),
            ("retrieve file from web", "network_download"),
            ("check if this website is up", "network_monitor"),
            ("monitor API endpoint", "network_monitor"),
            ("test connectivity to server", "network_monitor"),
            
            // Processing operations
            ("extract text from this document", "extraction"),
            ("pull out specific information", "extraction"),
            ("get the important parts from this", "extraction"),
            ("validate this data against schema", "validation"),
            ("check if this data is correct", "validation"),
            ("verify the format of this file", "validation"),
            ("generate a report from this data", "generation"),
            ("create summary of this information", "generation"),
            ("produce documentation", "generation"),
            ("classify this content", "classification"),
            ("categorize this data", "classification"),
            ("determine the type of this file", "classification"),
            
            // Code operations
            ("analyze this code for issues", "code_analyze"),
            ("review this source code", "code_analyze"),
            ("check code quality", "code_analyze"),
            ("process this text document", "text_process"),
            ("clean up this text", "text_process"),
            ("parse natural language", "text_process"),
        ]
    }
    
    /// Preprocess text for feature extraction
    fn preprocess_text(&self, text: &str) -> String {
        text.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    /// Extract text features using bag-of-words
    async fn extract_text_features(&self, text: &str) -> Result<Vec<f64>> {
        let mut features = vec![0.0; self.config.feature_dimensions];
        
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len() as f64;
        
        if word_count == 0.0 {
            return Ok(features);
        }
        
        // Simple term frequency approach
        for word in words {
            if let Some(index) = self.vocabulary.get(word) {
                if *index < features.len() {
                    features[*index] += 1.0 / word_count;
                }
            }
        }
        
        Ok(features)
    }
    
    /// Extract context features
    fn extract_context_features(&self, text: &str) -> Vec<f64> {
        vec![
            text.len() as f64 / 100.0, // Normalized text length
            text.split_whitespace().count() as f64 / 20.0, // Normalized word count
            if text.contains('?') { 1.0 } else { 0.0 }, // Question indicator
            if text.contains("file") { 1.0 } else { 0.0 }, // File operation indicator
            if text.contains("data") { 1.0 } else { 0.0 }, // Data operation indicator
        ]
    }
    
    /// Calculate cosine similarity between feature vectors
    fn cosine_similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
    
    /// Calculate context boost for intent scores
    fn calculate_context_boost(&self, intent: &IntentId, features: &FeatureVector) -> f64 {
        let mut boost = 0.0;
        
        if intent.0.contains("file") && features.context_features.get(3).unwrap_or(&0.0) > &0.0 {
            boost += 0.1;
        }
        
        if intent.0.contains("data") && features.context_features.get(4).unwrap_or(&0.0) > &0.0 {
            boost += 0.1;
        }
        
        boost
    }
    
    /// Rule-based classification fallback
    async fn rule_based_classification(&self, _features: &FeatureVector) -> HashMap<IntentId, f64> {
        let mut scores = HashMap::new();
        scores.insert(IntentId::from("general_processing"), 0.5);
        scores
    }
    
    /// Update intent patterns with new text
    async fn update_intent_patterns(&self, intent: &IntentId, text: &str) -> Result<()> {
        self.intent_patterns
            .entry(intent.clone())
            .or_insert_with(Vec::new)
            .push(text.to_string());
        Ok(())
    }
    
    /// Update vocabulary with new words
    async fn update_vocabulary(&self, text: &str) {
        for word in text.split_whitespace() {
            let vocab_len = self.vocabulary.len();
            if vocab_len < self.config.max_vocabulary_size && !self.vocabulary.contains_key(word) {
                self.vocabulary.insert(word.to_string(), vocab_len);
            }
        }
    }
    
    /// Check if model should be retrained
    async fn should_retrain(&self) -> bool {
        let training_data = self.training_data.read().await;
        let feedback_count = training_data
            .iter()
            .filter(|example| matches!(example.source, TrainingSource::UserFeedback))
            .count();
        
        feedback_count >= self.config.retraining_threshold
    }
    
    /// Retrain the model
    async fn retrain(&self) -> Result<()> {
        if self.config.debug_mode {
            info!("Retraining intent classification model");
        }
        
        // For now, just rebuild vocabulary and patterns
        // In a more sophisticated implementation, this would retrain ML models
        
        let training_data = self.training_data.read().await;
        
        // Clear and rebuild vocabulary
        self.vocabulary.clear();
        for example in training_data.iter() {
            self.update_vocabulary(&example.text).await;
        }
        
        if self.config.debug_mode {
            info!("Model retraining completed. Vocabulary size: {}", self.vocabulary.len());
        }
        
        Ok(())
    }
}

impl Default for IntentClassifier {
    fn default() -> Self {
        // Note: This can't be async, so we use a synchronous version
        // In practice, users should use `IntentClassifier::new().await`
        Self {
            training_data: Arc::new(RwLock::new(Vec::new())),
            vocabulary: Arc::new(DashMap::with_hasher(RandomState::new())),
            intent_patterns: Arc::new(DashMap::with_hasher(RandomState::new())),
            config: ClassifierConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_intent_classification() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        let prediction = classifier
            .predict_intent("merge these JSON files together")
            .await
            .unwrap();
        
        assert_eq!(prediction.intent.0, "data_merge");
        assert!(prediction.confidence.value() > 0.5);
    }
    
    #[tokio::test]
    async fn test_feedback_learning() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        let feedback = IntentFeedback {
            text: "combine data files".to_string(),
            predicted_intent: IntentId::from("data_transform"),
            actual_intent: IntentId::from("data_merge"),
            satisfaction_score: 5.0,
            notes: None,
            timestamp: chrono::Utc::now(),
        };
        
        classifier.add_feedback(feedback).await.unwrap();
        
        let stats = classifier.get_stats().await;
        assert!(stats.feedback_examples > 0);
    }
    
    #[tokio::test]
    async fn test_training_data_export_import() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        let example = TrainingExample {
            text: "test example".to_string(),
            intent: IntentId::from("test_intent"),
            confidence: 0.9,
            source: TrainingSource::Programmatic,
        };
        
        classifier.add_training_example(example).await.unwrap();
        
        let exported = classifier.export_training_data().await.unwrap();
        
        let new_classifier = IntentClassifier::new().await.unwrap();
        new_classifier.import_training_data(&exported).await.unwrap();
        
        let stats = new_classifier.get_stats().await;
        assert!(stats.training_examples > 0);
    }
}

/// Test helper methods - only available in test builds
#[cfg(test)]
impl IntentClassifier {
    /// Test helper to access cosine_similarity method
    pub fn test_cosine_similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        self.cosine_similarity(a, b)
    }
    
    /// Test helper to access extract_context_features method  
    pub fn test_extract_context_features(&self, text: &str) -> Vec<f64> {
        self.extract_context_features(text)
    }
    
    /// Test helper to access preprocess_text method
    pub fn test_preprocess_text(&self, text: &str) -> String {
        self.preprocess_text(text)
    }
}
