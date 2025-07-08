# Intent Classification Library

A flexible few-shot intent classification library for natural language processing in Rust. This library provides a simple API for classifying user intents from text using machine learning and rule-based approaches.

## Features

- **Few-shot learning**: Train the classifier with minimal examples
- **Bootstrap data**: Comes with pre-trained examples for common intents
- **Feedback learning**: Improve accuracy through user feedback
- **Async support**: Fully async API for non-blocking operations
- **Serializable**: Export/import training data as JSON
- **Configurable**: Customize behavior through configuration
- **High performance**: Built with Rust for speed and safety

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
intent-classifier = "0.1.0"
```

## Quick Start

```rust
use intent_classifier::{IntentClassifier, TrainingExample, TrainingSource, IntentId};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new classifier
    let classifier = IntentClassifier::new().await?;

    // Predict an intent
    let prediction = classifier.predict_intent("merge these JSON files together").await?;
    println!("Intent: {}, Confidence: {:.3}", 
             prediction.intent, prediction.confidence.value());

    // Add custom training data
    let example = TrainingExample {
        text: "calculate the sum of these numbers".to_string(),
        intent: IntentId::from("math_operation"),
        confidence: 1.0,
        source: TrainingSource::Programmatic,
    };
    classifier.add_training_example(example).await?;

    // Get statistics
    let stats = classifier.get_stats().await;
    println!("Training examples: {}", stats.training_examples);

    Ok(())
}
```

## Built-in Intent Categories

The library comes with bootstrap training data for common intent categories:

### Data Operations
- `data_merge` - Combining multiple data files
- `data_split` - Splitting large files into smaller ones
- `data_transform` - Converting between data formats
- `data_analyze` - Analyzing datasets for patterns

### File Operations
- `file_read` - Reading file contents
- `file_write` - Writing data to files
- `file_convert` - Converting file formats
- `file_compare` - Comparing files

### Network Operations
- `network_request` - Making HTTP/API requests
- `network_download` - Downloading files from URLs
- `network_monitor` - Monitoring network services

### Processing Operations
- `extraction` - Extracting information from documents
- `validation` - Validating data against schemas
- `generation` - Generating reports or documentation
- `classification` - Categorizing content

### Code Operations
- `code_analyze` - Analyzing source code
- `text_process` - Processing text documents

## Configuration

Customize the classifier behavior:

```rust
use intent_classifier::{IntentClassifier, ClassifierConfig};

let config = ClassifierConfig {
    feature_dimensions: 1000,
    max_vocabulary_size: 15000,
    min_confidence_threshold: 0.4,
    retraining_threshold: 5,
    debug_mode: true,
};

let classifier = IntentClassifier::with_config(config).await?;
```

## Advanced Usage

### Adding Custom Training Data

```rust
let example = TrainingExample {
    text: "solve this mathematical equation".to_string(),
    intent: IntentId::from("math_operation"),
    confidence: 1.0,
    source: TrainingSource::Programmatic,
};

classifier.add_training_example(example).await?;
```

### Feedback Learning

```rust
use intent_classifier::IntentFeedback;

let feedback = IntentFeedback {
    text: "combine these files".to_string(),
    predicted_intent: IntentId::from("file_write"),
    actual_intent: IntentId::from("data_merge"),
    satisfaction_score: 4.0,
    notes: Some("Should be classified as data merge".to_string()),
    timestamp: chrono::Utc::now(),
};

classifier.add_feedback(feedback).await?;
```

### Structured Classification Requests

```rust
use intent_classifier::ClassificationRequest;

let request = ClassificationRequest {
    text: "analyze this dataset".to_string(),
    context: None,
    include_alternatives: true,
    include_reasoning: true,
};

let response = classifier.classify(request).await?;
println!("Processing time: {:.2}ms", response.processing_time_ms);
```

### Export/Import Training Data

```rust
// Export training data
let exported = classifier.export_training_data().await?;
std::fs::write("training_data.json", exported)?;

// Import training data
let imported = std::fs::read_to_string("training_data.json")?;
let new_classifier = IntentClassifier::new().await?;
new_classifier.import_training_data(&imported).await?;
```

## Use Cases

### Multi-LLM Orchestration

Route different types of tasks to specialized language models:

```rust
async fn route_task(classifier: &IntentClassifier, task: &str) -> String {
    let prediction = classifier.predict_intent(task).await?;
    
    match prediction.intent.0.as_str() {
        "code_analyze" => "code-specialist-llm",
        "data_analyze" => "data-science-llm",
        "writing_creative" => "creative-writing-llm",
        _ => "general-purpose-llm",
    }.to_string()
}
```

### Chatbot Intent Recognition

```rust
async fn handle_user_message(classifier: &IntentClassifier, message: &str) -> Response {
    let prediction = classifier.predict_intent(message).await?;
    
    match prediction.intent.0.as_str() {
        "greeting" => Response::Greeting,
        "question" => Response::Answer,
        "complaint" => Response::Support,
        _ => Response::Default,
    }
}
```

### Command-Line Tool Classification

```rust
async fn classify_command(classifier: &IntentClassifier, command: &str) -> ToolAction {
    let prediction = classifier.predict_intent(command).await?;
    
    match prediction.intent.0.as_str() {
        "file_read" => ToolAction::ReadFile,
        "data_merge" => ToolAction::MergeData,
        "network_request" => ToolAction::HttpRequest,
        _ => ToolAction::Help,
    }
}
```

## Examples

Run the examples to see the library in action:

```bash
# Basic usage example
cargo run --example basic_usage

# Multi-LLM orchestration example
cargo run --example multi_llm_orchestration
```

## Performance

The library is designed for high performance:

- **Async operations**: Non-blocking API for concurrent processing
- **Efficient data structures**: Uses `DashMap` for thread-safe concurrent access
- **Minimal allocations**: Optimized for low memory usage
- **Fast feature extraction**: Efficient text processing and vectorization

## Architecture

The library consists of several key components:

1. **IntentClassifier**: Main classifier engine
2. **Types**: Core data structures and error types
3. **Feature Extraction**: Text processing and vectorization
4. **Training**: Learning from examples and feedback
5. **Prediction**: Intent classification and confidence scoring

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Changelog

### v0.1.0
- Initial release
- Basic intent classification functionality
- Bootstrap training data for common intents
- Feedback learning system
- Async API
- Export/import functionality
- Configuration support
- Comprehensive examples and documentation
