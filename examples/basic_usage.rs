//! Basic Intent Classification Example
//!
//! This example demonstrates the basic usage of the intent classification library.

use intent_classifier::{IntentClassifier, TrainingExample, TrainingSource, IntentId, IntentFeedback};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🚀 Intent Classification Library - Basic Example");
    println!("=================================================");
    
    // Create a new classifier
    println!("\n📚 Creating new classifier...");
    let classifier = IntentClassifier::new().await?;
    
    // Show initial statistics
    let stats = classifier.get_stats().await;
    println!("   Initial training examples: {}", stats.training_examples);
    println!("   Initial vocabulary size: {}", stats.vocabulary_size);
    println!("   Initial intent count: {}", stats.intent_count);
    
    // Test some basic classifications
    println!("\n🔍 Testing basic classifications:");
    
    let test_texts = vec![
        "merge these JSON files together",
        "split this large file into smaller parts",
        "analyze this dataset for patterns",
        "convert PDF to markdown",
        "make an API request to this URL",
        "check if this website is up",
        "extract text from this document",
        "validate this data against schema",
        "analyze this code for issues",
        "hello world",
    ];
    
    for text in test_texts {
        let prediction = classifier.predict_intent(text).await?;
        println!("   📝 '{}' -> {} (confidence: {:.3})", 
                 text, prediction.intent, prediction.confidence.value());
        
        if !prediction.alternative_intents.is_empty() {
            println!("      🔄 Alternatives: {:?}", 
                     prediction.alternative_intents.iter()
                         .map(|(intent, conf)| format!("{}({:.3})", intent, conf.value()))
                         .collect::<Vec<_>>());
        }
    }
    
    // Add custom training data
    println!("\n📖 Adding custom training data...");
    
    let custom_examples = vec![
        TrainingExample {
            text: "calculate the sum of these numbers".to_string(),
            intent: IntentId::from("math_operation"),
            confidence: 1.0,
            source: TrainingSource::Programmatic,
        },
        TrainingExample {
            text: "find the average of this dataset".to_string(),
            intent: IntentId::from("math_operation"),
            confidence: 1.0,
            source: TrainingSource::Programmatic,
        },
        TrainingExample {
            text: "solve this equation".to_string(),
            intent: IntentId::from("math_operation"),
            confidence: 1.0,
            source: TrainingSource::Programmatic,
        },
    ];
    
    for example in custom_examples {
        classifier.add_training_example(example).await?;
    }
    
    // Test the new intent
    println!("\n🧮 Testing new math operations:");
    let math_tests = vec![
        "calculate 2 + 2",
        "what is the average of 1, 2, 3",
        "solve for x in 2x + 3 = 7",
    ];
    
    for text in math_tests {
        let prediction = classifier.predict_intent(text).await?;
        println!("   📝 '{}' -> {} (confidence: {:.3})", 
                 text, prediction.intent, prediction.confidence.value());
    }
    
    // Demonstrate feedback learning
    println!("\n🎯 Demonstrating feedback learning...");
    
    let feedback = IntentFeedback {
        text: "combine these two files".to_string(),
        predicted_intent: IntentId::from("file_write"),
        actual_intent: IntentId::from("data_merge"),
        satisfaction_score: 4.0,
        notes: Some("This should be classified as data merge, not file write".to_string()),
        timestamp: chrono::Utc::now(),
    };
    
    classifier.add_feedback(feedback).await?;
    
    // Test the corrected classification
    let corrected_prediction = classifier.predict_intent("combine these two files").await?;
    println!("   📝 After feedback: 'combine these two files' -> {} (confidence: {:.3})", 
             corrected_prediction.intent, corrected_prediction.confidence.value());
    
    // Show final statistics
    println!("\n📊 Final statistics:");
    let final_stats = classifier.get_stats().await;
    println!("   Training examples: {}", final_stats.training_examples);
    println!("   Vocabulary size: {}", final_stats.vocabulary_size);
    println!("   Intent count: {}", final_stats.intent_count);
    println!("   Feedback examples: {}", final_stats.feedback_examples);
    
    // Export training data
    println!("\n💾 Exporting training data...");
    let exported_data = classifier.export_training_data().await?;
    println!("   Exported {} characters of training data", exported_data.len());
    
    println!("\n✅ Example completed successfully!");
    
    Ok(())
}
