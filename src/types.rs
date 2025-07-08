//! Core types for intent classification
//!
//! This module defines the fundamental types used throughout the intent classification library.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

/// Custom error types for the intent classification system
#[derive(Debug, thiserror::Error)]
pub enum IntentError {
    #[error("Classification failed: {0}")]
    ClassificationFailed(String),
    
    #[error("Invalid parameter '{parameter}': {message}")]
    InvalidParameter { parameter: String, message: String },
    
    #[error("Training failed: {0}")]
    TrainingFailed(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Invalid confidence value: {0}")]
    InvalidConfidence(String),
}

/// Result type for intent classification operations
pub type Result<T> = std::result::Result<T, IntentError>;

/// Unique identifier for intents
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IntentId(pub String);

impl fmt::Display for IntentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for IntentId {
    fn from(s: String) -> Self {
        IntentId(s)
    }
}

impl From<&str> for IntentId {
    fn from(s: &str) -> Self {
        IntentId(s.to_string())
    }
}

/// Confidence score (0.0 to 1.0)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Confidence(f64);

impl Confidence {
    /// Create a new confidence score
    pub fn new(value: f64) -> Result<Self> {
        if (0.0..=1.0).contains(&value) {
            Ok(Confidence(value))
        } else {
            Err(IntentError::InvalidParameter {
                parameter: "confidence".to_string(),
                message: format!("Confidence must be between 0.0 and 1.0, got {}", value),
            })
        }
    }
    
    /// Get the confidence value
    pub fn value(&self) -> f64 {
        self.0
    }
    
    /// Check if confidence is high (>= 0.8)
    pub fn is_high(&self) -> bool {
        self.0 >= 0.8
    }
    
    /// Check if confidence is medium (0.5 to 0.8)
    pub fn is_medium(&self) -> bool {
        self.0 >= 0.5 && self.0 < 0.8
    }
    
    /// Check if confidence is low (< 0.5)
    pub fn is_low(&self) -> bool {
        self.0 < 0.5
    }
}

impl Default for Confidence {
    fn default() -> Self {
        Confidence(1.0)
    }
}

impl From<Confidence> for f64 {
    fn from(confidence: Confidence) -> Self {
        confidence.0
    }
}

/// Intent classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentPrediction {
    /// The predicted intent
    pub intent: IntentId,
    
    /// Confidence score for the prediction
    pub confidence: Confidence,
    
    /// Alternative intents with their confidence scores
    pub alternative_intents: Vec<(IntentId, Confidence)>,
    
    /// Human-readable reasoning for the classification
    pub reasoning: String,
}

/// Training example for intent classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    /// The input text
    pub text: String,
    
    /// The correct intent for this text
    pub intent: IntentId,
    
    /// Confidence score for this example (0.0 to 1.0)
    pub confidence: f64,
    
    /// Source of this training example
    pub source: TrainingSource,
}

/// Source of training data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingSource {
    /// Bootstrap data provided with the library
    Bootstrap,
    
    /// User-provided feedback
    UserFeedback,
    
    /// Programmatically added examples
    Programmatic,
}

/// Feature vector for machine learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    /// Text-based features (e.g., TF-IDF)
    pub text_features: Vec<f64>,
    
    /// Context-based features (e.g., text length, word count)
    pub context_features: Vec<f64>,
    
    /// Additional metadata features
    pub metadata: HashMap<String, f64>,
}

/// Configuration for the intent classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierConfig {
    /// Number of dimensions for feature vectors
    pub feature_dimensions: usize,
    
    /// Maximum vocabulary size
    pub max_vocabulary_size: usize,
    
    /// Minimum confidence threshold for predictions
    pub min_confidence_threshold: f64,
    
    /// Number of feedback examples required before retraining
    pub retraining_threshold: usize,
    
    /// Whether to enable debug logging
    pub debug_mode: bool,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            feature_dimensions: 1000,
            max_vocabulary_size: 10000,
            min_confidence_threshold: 0.3,
            retraining_threshold: 10,
            debug_mode: false,
        }
    }
}

/// Statistics about the classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierStats {
    /// Number of training examples
    pub training_examples: usize,
    
    /// Size of the vocabulary
    pub vocabulary_size: usize,
    
    /// Number of known intents
    pub intent_count: usize,
    
    /// Number of user feedback examples
    pub feedback_examples: usize,
    
    /// Last update timestamp
    pub last_updated: Option<DateTime<Utc>>,
}

/// Feedback for improving the classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentFeedback {
    /// The original text that was classified
    pub text: String,
    
    /// The intent that was predicted
    pub predicted_intent: IntentId,
    
    /// The correct intent (according to user feedback)
    pub actual_intent: IntentId,
    
    /// User satisfaction score (1.0 to 5.0)
    pub satisfaction_score: f64,
    
    /// Additional notes from the user
    pub notes: Option<String>,
    
    /// Timestamp of the feedback
    pub timestamp: DateTime<Utc>,
}

/// Intent classification request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationRequest {
    /// The text to classify
    pub text: String,
    
    /// Optional context information
    pub context: Option<HashMap<String, String>>,
    
    /// Whether to include alternative intents in the response
    pub include_alternatives: bool,
    
    /// Whether to include reasoning in the response
    pub include_reasoning: bool,
}

/// Intent classification response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResponse {
    /// The classification result
    pub prediction: IntentPrediction,
    
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    
    /// Request ID for tracking
    pub request_id: Uuid,
}
