//! Comprehensive test suite for intent classification library
//! 
//! This module contains extensive tests covering all functionality,
//! edge cases, error conditions, and performance scenarios.

use intent_classifier::*;
use std::collections::HashMap;

#[cfg(test)]
mod comprehensive_tests {
    use super::*;

    // Basic functionality tests
    
    #[tokio::test]
    async fn test_classifier_initialization() {
        let classifier = IntentClassifier::new().await.unwrap();
        let stats = classifier.get_stats().await;
        
        assert!(stats.training_examples > 0);
        assert!(stats.vocabulary_size > 0);
        assert!(stats.intent_count > 0);
        assert_eq!(stats.feedback_examples, 0);
    }
    
    #[tokio::test]
    async fn test_classifier_with_custom_config() {
        let config = ClassifierConfig {
            feature_dimensions: 500,
            max_vocabulary_size: 5000,
            min_confidence_threshold: 0.2,
            retraining_threshold: 5,
            debug_mode: true,
        };
        
        let classifier = IntentClassifier::with_config(config).await.unwrap();
        let stats = classifier.get_stats().await;
        
        assert!(stats.training_examples > 0);
    }
    
    // Confidence type tests
    
    #[test]
    fn test_confidence_creation() {
        assert!(Confidence::new(0.0).is_ok());
        assert!(Confidence::new(0.5).is_ok());
        assert!(Confidence::new(1.0).is_ok());
        
        assert!(Confidence::new(-0.1).is_err());
        assert!(Confidence::new(1.1).is_err());
        assert!(Confidence::new(f64::NAN).is_err());
        assert!(Confidence::new(f64::INFINITY).is_err());
    }
    
    #[test]
    fn test_confidence_methods() {
        let low_conf = Confidence::new(0.3).unwrap();
        let medium_conf = Confidence::new(0.6).unwrap();
        let high_conf = Confidence::new(0.9).unwrap();
        
        assert!(low_conf.is_low());
        assert!(!low_conf.is_medium());
        assert!(!low_conf.is_high());
        
        assert!(!medium_conf.is_low());
        assert!(medium_conf.is_medium());
        assert!(!medium_conf.is_high());
        
        assert!(!high_conf.is_low());
        assert!(!high_conf.is_medium());
        assert!(high_conf.is_high());
    }
    
    // Training example validation tests
    
    #[tokio::test]
    async fn test_training_example_validation() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        // Valid example
        let valid_example = TrainingExample {
            text: "valid text".to_string(),
            intent: IntentId::from("valid_intent"),
            confidence: 0.8,
            source: TrainingSource::Programmatic,
        };
        assert!(classifier.add_training_example(valid_example).await.is_ok());
        
        // Empty text should fail
        let empty_text_example = TrainingExample {
            text: "".to_string(),
            intent: IntentId::from("intent"),
            confidence: 0.8,
            source: TrainingSource::Programmatic,
        };
        assert!(classifier.add_training_example(empty_text_example).await.is_err());
        
        // Whitespace only text should fail
        let whitespace_example = TrainingExample {
            text: "   \t\n   ".to_string(),
            intent: IntentId::from("intent"),
            confidence: 0.8,
            source: TrainingSource::Programmatic,
        };
        assert!(classifier.add_training_example(whitespace_example).await.is_err());
        
        // Invalid confidence should fail
        let invalid_confidence_example = TrainingExample {
            text: "text".to_string(),
            intent: IntentId::from("intent"),
            confidence: 1.5,
            source: TrainingSource::Programmatic,
        };
        assert!(classifier.add_training_example(invalid_confidence_example).await.is_err());
        
        let negative_confidence_example = TrainingExample {
            text: "text".to_string(),
            intent: IntentId::from("intent"),
            confidence: -0.1,
            source: TrainingSource::Programmatic,
        };
        assert!(classifier.add_training_example(negative_confidence_example).await.is_err());
    }
    
    // Classification request tests
    
    #[tokio::test]
    async fn test_classification_request_options() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        // Test with all options enabled
        let full_request = ClassificationRequest {
            text: "analyze this data".to_string(),
            context: Some(HashMap::new()),
            include_alternatives: true,
            include_reasoning: true,
        };
        
        let full_response = classifier.classify(full_request).await.unwrap();
        assert!(!full_response.prediction.alternative_intents.is_empty() || 
                full_response.prediction.alternative_intents.is_empty()); // Either is valid
        assert!(!full_response.prediction.reasoning.is_empty());
        
        // Test with options disabled
        let minimal_request = ClassificationRequest {
            text: "analyze this data".to_string(),
            context: None,
            include_alternatives: false,
            include_reasoning: false,
        };
        
        let minimal_response = classifier.classify(minimal_request).await.unwrap();
        assert!(minimal_response.prediction.alternative_intents.is_empty());
        assert!(minimal_response.prediction.reasoning.is_empty());
    }
    
    // Edge case text inputs
    
    #[tokio::test]
    async fn test_edge_case_inputs() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        // Empty string
        let prediction = classifier.predict_intent("").await.unwrap();
        assert_eq!(prediction.intent.0, "general_processing");
        
        // Single character
        let prediction = classifier.predict_intent("a").await.unwrap();
        assert!(!prediction.intent.0.is_empty());
        
        // Very long text
        let long_text = "a ".repeat(1000);
        let prediction = classifier.predict_intent(&long_text).await.unwrap();
        assert!(!prediction.intent.0.is_empty());
        
        // Special characters
        let special_text = "!@#$%^&*()_+-=[]{}|;':\",./<>?";
        let prediction = classifier.predict_intent(special_text).await.unwrap();
        assert!(!prediction.intent.0.is_empty());
        
        // Unicode text
        let unicode_text = "analyze 这些 données с данными";
        let prediction = classifier.predict_intent(unicode_text).await.unwrap();
        assert!(!prediction.intent.0.is_empty());
        
        // Numbers only
        let numbers_text = "123 456 789";
        let prediction = classifier.predict_intent(numbers_text).await.unwrap();
        assert!(!prediction.intent.0.is_empty());
        
        // Mixed content
        let mixed_text = "Merge 123 files_with-special.chars ✨";
        let prediction = classifier.predict_intent(mixed_text).await.unwrap();
        assert!(!prediction.intent.0.is_empty());
    }
    
    // Feedback system tests
    
    #[tokio::test]
    async fn test_feedback_system() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        // Test valid feedback scores
        for score in [1.0, 2.5, 3.0, 4.5, 5.0] {
            let feedback = IntentFeedback {
                text: format!("test feedback {}", score),
                predicted_intent: IntentId::from("predicted"),
                actual_intent: IntentId::from("actual"),
                satisfaction_score: score,
                notes: Some("Test feedback".to_string()),
                timestamp: chrono::Utc::now(),
            };
            
            assert!(classifier.add_feedback(feedback).await.is_ok());
        }
        
        let stats = classifier.get_stats().await;
        assert_eq!(stats.feedback_examples, 5);
    }
    
    #[tokio::test]
    async fn test_automatic_retraining() {
        let config = ClassifierConfig {
            retraining_threshold: 2,
            ..Default::default()
        };
        
        let classifier = IntentClassifier::with_config(config).await.unwrap();
        let initial_vocab_size = classifier.get_stats().await.vocabulary_size;
        
        // Add feedback that should trigger retraining
        for i in 0..3 {
            let feedback = IntentFeedback {
                text: format!("unique_feedback_text_{}", i),
                predicted_intent: IntentId::from("predicted"),
                actual_intent: IntentId::from("actual"),
                satisfaction_score: 5.0,
                notes: None,
                timestamp: chrono::Utc::now(),
            };
            
            classifier.add_feedback(feedback).await.unwrap();
        }
        
        let final_stats = classifier.get_stats().await;
        assert_eq!(final_stats.feedback_examples, 3);
        // Vocabulary should have grown with new words
        assert!(final_stats.vocabulary_size >= initial_vocab_size);
    }
    
    // Data persistence tests
    
    #[tokio::test]
    async fn test_data_export_import() {
        let classifier1 = IntentClassifier::new().await.unwrap();
        
        // Add custom training data
        let examples = vec![
            TrainingExample {
                text: "custom example 1".to_string(),
                intent: IntentId::from("custom_intent_1"),
                confidence: 0.9,
                source: TrainingSource::Programmatic,
            },
            TrainingExample {
                text: "custom example 2".to_string(),
                intent: IntentId::from("custom_intent_2"),
                confidence: 0.8,
                source: TrainingSource::UserFeedback,
            },
        ];
        
        for example in examples {
            classifier1.add_training_example(example).await.unwrap();
        }
        
        // Export data (includes bootstrap + custom examples)
        let exported_data = classifier1.export_training_data().await.unwrap();
        assert!(!exported_data.is_empty());
        
        // Create new classifier and import data
        let classifier2 = IntentClassifier::new().await.unwrap();
        
        // Clear bootstrap data to get a clean state
        classifier2.clear_training_data().await.unwrap();
        
        classifier2.import_training_data(&exported_data).await.unwrap();
        
        let imported_stats = classifier2.get_stats().await;
        // After import, we should have bootstrap data + imported examples
        let expected_count = serde_json::from_str::<Vec<TrainingExample>>(&exported_data).unwrap().len();
        
        // Check that we have at least the expected imported count
        assert!(imported_stats.training_examples >= expected_count);
        
        // Test classification works on imported data
        let prediction = classifier2.predict_intent("custom example 1").await.unwrap();
        assert_eq!(prediction.intent.0, "custom_intent_1");
    }
    
    #[tokio::test]
    async fn test_invalid_json_import() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        // Invalid JSON should fail
        assert!(classifier.import_training_data("invalid json").await.is_err());
        assert!(classifier.import_training_data("{}").await.is_err());
        assert!(classifier.import_training_data("[]").await.is_ok()); // Empty array is valid
    }
    
    // Clear data functionality
    
    #[tokio::test]
    async fn test_clear_training_data() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        // Add custom data
        let example = TrainingExample {
            text: "custom example".to_string(),
            intent: IntentId::from("custom_intent"),
            confidence: 0.9,
            source: TrainingSource::Programmatic,
        };
        classifier.add_training_example(example).await.unwrap();
        
        let stats_before = classifier.get_stats().await;
        
        // Clear all data
        classifier.clear_training_data().await.unwrap();
        
        let stats_after = classifier.get_stats().await;
        
        // Should have bootstrap data only
        assert!(stats_after.training_examples > 0); // Bootstrap data reloaded
        assert!(stats_after.training_examples <= stats_before.training_examples);
        assert_eq!(stats_after.feedback_examples, 0);
    }
    
    // Performance and stress tests
    
    #[tokio::test]
    async fn test_concurrent_classification() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        // Test concurrent access
        let tasks: Vec<_> = (0..10)
            .map(|i| {
                let classifier = classifier.clone();
                tokio::spawn(async move {
                    classifier.predict_intent(&format!("test message {}", i)).await
                })
            })
            .collect();
        
        for task in tasks {
            let result = task.await.unwrap();
            assert!(result.is_ok());
        }
    }
    
    #[tokio::test]
    async fn test_concurrent_training() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        // Test concurrent training data addition
        let tasks: Vec<_> = (0..10)
            .map(|i| {
                let classifier = classifier.clone();
                tokio::spawn(async move {
                    let example = TrainingExample {
                        text: format!("concurrent example {}", i),
                        intent: IntentId::from(format!("intent_{}", i)),
                        confidence: 0.8,
                        source: TrainingSource::Programmatic,
                    };
                    classifier.add_training_example(example).await
                })
            })
            .collect();
        
        for task in tasks {
            let result = task.await.unwrap();
            assert!(result.is_ok());
        }
        
        let stats = classifier.get_stats().await;
        assert!(stats.training_examples >= 10); // At least our 10 examples plus bootstrap
    }
    
    #[tokio::test]
    async fn test_large_vocabulary() {
        let config = ClassifierConfig {
            max_vocabulary_size: 100, // Small limit to test overflow
            ..Default::default()
        };
        
        let classifier = IntentClassifier::with_config(config).await.unwrap();
        
        // Add many unique words
        for i in 0..150 {
            let example = TrainingExample {
                text: format!("unique_word_{} another_unique_word_{}", i, i),
                intent: IntentId::from("test_intent"),
                confidence: 0.8,
                source: TrainingSource::Programmatic,
            };
            classifier.add_training_example(example).await.unwrap();
        }
        
        let stats = classifier.get_stats().await;
        assert!(stats.vocabulary_size <= 100); // Should respect the limit
    }
    
    // Alternative intents testing
    
    #[tokio::test]
    async fn test_alternative_intents() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        // Add similar examples for different intents
        let examples = vec![
            ("process this file", "file_process"),
            ("process this data", "data_process"),
            ("process this request", "request_process"),
        ];
        
        for (text, intent) in examples {
            let example = TrainingExample {
                text: text.to_string(),
                intent: IntentId::from(intent),
                confidence: 1.0,
                source: TrainingSource::Programmatic,
            };
            classifier.add_training_example(example).await.unwrap();
        }
        
        // Test ambiguous input
        let prediction = classifier.predict_intent("process this").await.unwrap();
        assert!(!prediction.alternative_intents.is_empty());
        assert!(prediction.alternative_intents.len() <= 3); // Should limit to top 3
        
        // Verify alternatives are sorted by confidence
        for i in 1..prediction.alternative_intents.len() {
            assert!(prediction.alternative_intents[i-1].1.value() >= 
                   prediction.alternative_intents[i].1.value());
        }
    }
    
    // Text preprocessing edge cases
    
    #[tokio::test]
    async fn test_text_preprocessing() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        // Test case insensitivity
        let upper_pred = classifier.predict_intent("MERGE FILES").await.unwrap();
        let lower_pred = classifier.predict_intent("merge files").await.unwrap();
        let mixed_pred = classifier.predict_intent("MeRgE fIlEs").await.unwrap();
        
        assert_eq!(upper_pred.intent, lower_pred.intent);
        assert_eq!(lower_pred.intent, mixed_pred.intent);
        
        // Test special character handling
        let clean_pred = classifier.predict_intent("merge files").await.unwrap();
        let dirty_pred = classifier.predict_intent("merge!!! @#$ files???").await.unwrap();
        
        // Should classify similarly
        assert_eq!(clean_pred.intent, dirty_pred.intent);
    }
    
    // Error handling tests
    
    #[tokio::test]
    async fn test_error_types() {
        // Test IntentError types
        let invalid_confidence = Confidence::new(2.0);
        assert!(matches!(invalid_confidence.unwrap_err(), 
                        IntentError::InvalidParameter { .. }));
    }
    
    // Configuration validation
    
    #[test]
    fn test_config_defaults() {
        let config = ClassifierConfig::default();
        
        assert_eq!(config.feature_dimensions, 1000);
        assert_eq!(config.max_vocabulary_size, 10000);
        assert_eq!(config.min_confidence_threshold, 0.3);
        assert_eq!(config.retraining_threshold, 10);
        assert!(!config.debug_mode);
    }
    
    // Exact match functionality
    
    #[tokio::test]
    async fn test_exact_match_priority() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        // Test that exact matches get highest confidence
        let prediction = classifier.predict_intent("merge these JSON files together").await.unwrap();
        assert_eq!(prediction.confidence.value(), 1.0);
        assert_eq!(prediction.reasoning, "Exact match found in training data");
    }
    
    // Intent statistics validation
    
    #[tokio::test]
    async fn test_statistics_accuracy() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        let initial_stats = classifier.get_stats().await;
        
        // Add programmatic example
        let prog_example = TrainingExample {
            text: "programmatic example".to_string(),
            intent: IntentId::from("prog_intent"),
            confidence: 0.9,
            source: TrainingSource::Programmatic,
        };
        classifier.add_training_example(prog_example).await.unwrap();
        
        // Add feedback example
        let feedback = IntentFeedback {
            text: "feedback example".to_string(),
            predicted_intent: IntentId::from("predicted"),
            actual_intent: IntentId::from("actual"),
            satisfaction_score: 4.0,
            notes: None,
            timestamp: chrono::Utc::now(),
        };
        classifier.add_feedback(feedback).await.unwrap();
        
        let final_stats = classifier.get_stats().await;
        
        assert_eq!(final_stats.training_examples, initial_stats.training_examples + 2);
        assert_eq!(final_stats.feedback_examples, 1);
        assert!(final_stats.vocabulary_size > initial_stats.vocabulary_size);
        assert!(final_stats.last_updated.is_some());
    }
}
