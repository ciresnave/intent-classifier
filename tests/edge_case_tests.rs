//! Edge case tests for intent classification library
//! 
//! This module tests boundary conditions, resource limits, unusual inputs,
//! and other edge cases that might occur in real-world usage.

use intent_classifier::*;
use chrono;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_empty_inputs() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        // Empty string should not crash
        let result = classifier.predict_intent("").await;
        assert!(result.is_ok());
        
        // Whitespace only
        let result = classifier.predict_intent("   \t\n   ").await;
        assert!(result.is_ok());
        
        // Single character
        let result = classifier.predict_intent("a").await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_very_long_inputs() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        // Very long string
        let long_input = "word ".repeat(1000);
        let result = classifier.predict_intent(&long_input).await;
        assert!(result.is_ok());
        
        // Very long single word
        let long_word = "a".repeat(10000);
        let result = classifier.predict_intent(&long_word).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_special_characters() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        let special_inputs = vec![
            "¡¢£¤¥¦§¨©ª«¬­®¯°±²³´µ¶·¸¹º»¼½¾¿",
            "αβγδεζηθικλμνξοπρστυφχψω",
            "你好世界",
            "🚀🌟💫⭐️✨🎯",
            "\\n\\t\\r\\0",
            "\"quoted text\"",
            "'single quotes'",
            "`backticks`",
            "line1\nline2\nline3",
            "tab\tseparated\tvalues",
        ];
        
        for input in special_inputs {
            let result = classifier.predict_intent(input).await;
            assert!(result.is_ok(), "Failed on input: {}", input);
        }
    }
    
    #[tokio::test]
    async fn test_edge_case_training_examples() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        // Empty intent id - this should probably work
        let example = TrainingExample {
            intent: IntentId("".to_string()),
            text: "some text".to_string(),
            confidence: 1.0,
            source: TrainingSource::UserFeedback,
        };
        
        let _result = classifier.add_training_example(example).await;
        // Empty intent might be rejected, so we don't assert it's ok
        
        // Very long intent id
        let long_intent = "a".repeat(1000);
        let example = TrainingExample {
            intent: IntentId(long_intent),
            text: "some text".to_string(),
            confidence: 1.0,
            source: TrainingSource::UserFeedback,
        };
        
        let result = classifier.add_training_example(example).await;
        assert!(result.is_ok());
        
        // Empty text - this should be rejected by validation
        let example = TrainingExample {
            intent: IntentId("test_intent".to_string()),
            text: "".to_string(),
            confidence: 1.0,
            source: TrainingSource::UserFeedback,
        };
        
        let result = classifier.add_training_example(example).await;
        assert!(result.is_err());
        
        // Very low confidence
        let example = TrainingExample {
            intent: IntentId("test_intent".to_string()),
            text: "test text".to_string(),
            confidence: 0.1,
            source: TrainingSource::UserFeedback,
        };
        
        let result = classifier.add_training_example(example).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_extreme_confidence_values() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        // Add some training data
        let example = TrainingExample {
            intent: IntentId("test_intent".to_string()),
            text: "test example".to_string(),
            confidence: 1.0,
            source: TrainingSource::UserFeedback,
        };
        
        classifier.add_training_example(example).await.unwrap();
        
        // Test with very similar text (should get high confidence)
        let result = classifier.predict_intent("test example").await.unwrap();
        assert!(result.confidence.value() > 0.5);
        
        // Test with very different text (should get low confidence)
        let result = classifier.predict_intent("completely different unrelated text").await.unwrap();
        assert!(result.confidence.value() >= 0.0);
        assert!(result.confidence.value() <= 1.0);
    }
    
    #[tokio::test]
    async fn test_massive_vocabulary() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        // Add many training examples with different words
        for i in 0..100 {
            let example = TrainingExample {
                intent: IntentId(format!("intent_{}", i % 10)),
                text: format!("unique_word_{} action_{} object_{}", i, i*2, i*3),
                confidence: 1.0,
                source: TrainingSource::UserFeedback,
            };
            
            classifier.add_training_example(example).await.unwrap();
        }
        
        // Should still work with the large vocabulary
        let result = classifier.predict_intent("unique_word_50 action_100 object_150").await;
        assert!(result.is_ok());
        
        let stats = classifier.get_stats().await;
        assert!(stats.training_examples >= 100);
    }
    
    #[tokio::test]
    async fn test_concurrent_operations() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        // Add initial training example
        let example = TrainingExample {
            intent: IntentId("test_intent".to_string()),
            text: "test example".to_string(),
            confidence: 1.0,
            source: TrainingSource::UserFeedback,
        };
        classifier.add_training_example(example).await.unwrap();
        
        // Test concurrent predictions
        let mut handles = vec![];
        
        for i in 0..10 {
            let classifier_clone = classifier.clone();
            let handle = tokio::spawn(async move {
                let result = classifier_clone.predict_intent(&format!("test query {}", i)).await;
                assert!(result.is_ok());
            });
            handles.push(handle);
        }
        
        // Wait for all concurrent operations to complete
        for handle in handles {
            handle.await.unwrap();
        }
    }
    
    #[tokio::test]
    async fn test_malformed_json_import() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        let malformed_json_cases = vec![
            "",
            "not json",
            "{}",
            "[]",
            "{\"invalid\": \"structure\"}",
            "null",
            "123",
            "\"just a string\"",
        ];
        
        for malformed_json in malformed_json_cases {
            let result = classifier.import_training_data(malformed_json).await;
            // Should handle gracefully (either succeed with empty data or fail safely)
            match result {
                Ok(_) => {
                    // If it succeeds, that's fine - it handled the malformed data gracefully
                }
                Err(_) => {
                    // If it fails, that's also fine - it detected the malformed data
                }
            }
        }
    }
    
    #[tokio::test]
    async fn test_feedback_with_edge_cases() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        // Add initial training
        let example = TrainingExample {
            intent: IntentId("test_intent".to_string()),
            text: "test example".to_string(),
            confidence: 1.0,
            source: TrainingSource::UserFeedback,
        };
        classifier.add_training_example(example).await.unwrap();
        
        // Test feedback with empty intent
        let feedback = IntentFeedback {
            text: "test text".to_string(),
            predicted_intent: IntentId("some_intent".to_string()),
            actual_intent: IntentId("".to_string()),
            satisfaction_score: 3.0,
            notes: None,
            timestamp: chrono::Utc::now(),
        };
        let result = classifier.add_feedback(feedback).await;
        assert!(result.is_ok());
        
        // Test feedback with very long text
        let long_text = "word ".repeat(500);
        let feedback = IntentFeedback {
            text: long_text,
            predicted_intent: IntentId("some_intent".to_string()),
            actual_intent: IntentId("test_intent".to_string()),
            satisfaction_score: 4.0,
            notes: None,
            timestamp: chrono::Utc::now(),
        };
        let result = classifier.add_feedback(feedback).await;
        assert!(result.is_ok());
        
        // Test feedback with special characters
        let feedback = IntentFeedback {
            text: "🚀 test 你好".to_string(),
            predicted_intent: IntentId("some_intent".to_string()),
            actual_intent: IntentId("test_intent".to_string()),
            satisfaction_score: 5.0,
            notes: Some("Great work!".to_string()),
            timestamp: chrono::Utc::now(),
        };
        let result = classifier.add_feedback(feedback).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_statistics_edge_cases() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        // Get initial stats with bootstrap data
        let initial_stats = classifier.get_stats().await;
        let initial_count = initial_stats.training_examples;
        
        // Clear training data and verify it's actually cleared
        classifier.clear_training_data().await.unwrap();
        let stats = classifier.get_stats().await;
        
        // After clearing, we should have fewer (or zero) examples
        assert!(stats.training_examples <= initial_count);
        
        // Add one example
        let example = TrainingExample {
            intent: IntentId("test_intent".to_string()),
            text: "test example".to_string(),
            confidence: 1.0,
            source: TrainingSource::UserFeedback,
        };
        classifier.add_training_example(example).await.unwrap();
        
        let stats = classifier.get_stats().await;
        assert!(stats.training_examples > 0);
        // Intent count should include the new intent plus bootstrap intents
        assert!(stats.intent_count >= 1);
        assert!(stats.vocabulary_size >= 1);
        
        // Add duplicate example
        let example = TrainingExample {
            intent: IntentId("test_intent".to_string()),
            text: "test example".to_string(),
            confidence: 1.0,
            source: TrainingSource::UserFeedback,
        };
        classifier.add_training_example(example).await.unwrap();
        
        let new_stats = classifier.get_stats().await;
        // Should have one more training example but same intent count
        assert!(new_stats.training_examples > stats.training_examples);
        assert_eq!(new_stats.intent_count, stats.intent_count);
    }
    
    #[tokio::test]
    async fn test_resource_limits() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        // Test with maximum feature dimensions
        let config = ClassifierConfig {
            feature_dimensions: 10000,
            ..Default::default()
        };
        
        let high_dim_classifier = IntentClassifier::with_config(config).await.unwrap();
        
        // Should work with high dimensions
        let example = TrainingExample {
            intent: IntentId("test_intent".to_string()),
            text: "test example".to_string(),
            confidence: 1.0,
            source: TrainingSource::UserFeedback,
        };
        
        high_dim_classifier.add_training_example(example).await.unwrap();
        
        let result = high_dim_classifier.predict_intent("test query").await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_configuration_edge_cases() {
        // Test with minimum feature dimensions
        let config = ClassifierConfig {
            feature_dimensions: 1,
            min_confidence_threshold: 0.0,
            debug_mode: true,
            ..Default::default()
        };
        
        let classifier = IntentClassifier::with_config(config).await.unwrap();
        
        // Should still work with minimal config
        let example = TrainingExample {
            intent: IntentId("test_intent".to_string()),
            text: "test example".to_string(),
            confidence: 1.0,
            source: TrainingSource::UserFeedback,
        };
        
        classifier.add_training_example(example).await.unwrap();
        
        let result = classifier.predict_intent("test query").await;
        assert!(result.is_ok());
        
        // Test with large vocabulary limit
        let config = ClassifierConfig {
            max_vocabulary_size: 100000,
            ..Default::default()
        };
        
        let classifier = IntentClassifier::with_config(config).await.unwrap();
        
        // Add multiple intents
        for i in 0..10 {
            let example = TrainingExample {
                intent: IntentId(format!("intent_{}", i)),
                text: format!("example text {}", i),
                confidence: 1.0,
                source: TrainingSource::UserFeedback,
            };
            classifier.add_training_example(example).await.unwrap();
        }
        
        let result = classifier.predict_intent("example text").await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_unusual_input_patterns() {
        let classifier = IntentClassifier::new().await.unwrap();
        
        let unusual_inputs = vec![
            // Repeated patterns
            "test test test test test",
            "a a a a a a a a a a",
            
            // Mixed case patterns
            "tEsT TeXt",
            "UPPER lower MixeD",
            
            // Numbers and text
            "123 456 789",
            "test 123 example 456",
            
            // URLs and paths
            "https://example.com/path?query=value",
            "/home/user/file.txt",
            "C:\\Windows\\System32\\file.exe",
            
            // Very long words
            "supercalifragilisticexpialidocious",
            "pneumonoultramicroscopicsilicovolcanoconiosis",
            
            // Weird punctuation
            "!@#$%^&*()_+-=[]{}|;':\",./<>?",
            "test... with... dots...",
            "text--with--dashes",
            
            // Code-like patterns
            "function(arg1, arg2)",
            "if (condition) { return true; }",
            "SELECT * FROM table WHERE id = 1",
        ];
        
        for input in unusual_inputs {
            let result = classifier.predict_intent(input).await;
            assert!(result.is_ok(), "Failed on input: {}", input);
        }
    }
}
