//! Multi-LLM Orchestration Example
//!
//! This example demonstrates how to use the intent classification library
//! in a multi-LLM orchestration system where different types of tasks
//! are routed to different language models based on intent.

use intent_classifier::{
    IntentClassifier, TrainingExample, TrainingSource, IntentId, 
    ClassificationRequest, ClassifierConfig
};
use std::collections::HashMap;

/// Represents a language model in our orchestration system
#[derive(Debug, Clone)]
struct LLMModel {
    name: String,
    capabilities: Vec<String>,
    cost_per_token: f64,
    response_time_ms: u64,
}

/// Orchestrator that routes tasks to appropriate LLMs based on intent
struct MultiLLMOrchestrator {
    classifier: IntentClassifier,
    models: HashMap<String, LLMModel>,
    intent_to_model: HashMap<IntentId, String>,
}

impl MultiLLMOrchestrator {
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Create classifier with custom config
        let config = ClassifierConfig {
            feature_dimensions: 1000,
            max_vocabulary_size: 15000,
            min_confidence_threshold: 0.4,
            retraining_threshold: 5,
            debug_mode: true,
        };
        
        let classifier = IntentClassifier::with_config(config).await?;
        
        // Define available models
        let mut models = HashMap::new();
        
        models.insert("gpt-4".to_string(), LLMModel {
            name: "GPT-4".to_string(),
            capabilities: vec!["reasoning", "code", "analysis", "writing"].iter().map(|s| s.to_string()).collect(),
            cost_per_token: 0.03,
            response_time_ms: 2000,
        });
        
        models.insert("claude-3".to_string(), LLMModel {
            name: "Claude-3".to_string(),
            capabilities: vec!["reasoning", "analysis", "writing", "safety"].iter().map(|s| s.to_string()).collect(),
            cost_per_token: 0.025,
            response_time_ms: 1500,
        });
        
        models.insert("llama-70b".to_string(), LLMModel {
            name: "Llama-70B".to_string(),
            capabilities: vec!["general", "code", "fast"].iter().map(|s| s.to_string()).collect(),
            cost_per_token: 0.01,
            response_time_ms: 800,
        });
        
        models.insert("code-specialist".to_string(), LLMModel {
            name: "Code Specialist".to_string(),
            capabilities: vec!["code", "debugging", "refactoring"].iter().map(|s| s.to_string()).collect(),
            cost_per_token: 0.02,
            response_time_ms: 1200,
        });
        
        // Define intent to model mapping
        let mut intent_to_model = HashMap::new();
        intent_to_model.insert(IntentId::from("code_analyze"), "code-specialist".to_string());
        intent_to_model.insert(IntentId::from("code_debug"), "code-specialist".to_string());
        intent_to_model.insert(IntentId::from("code_refactor"), "code-specialist".to_string());
        intent_to_model.insert(IntentId::from("data_analyze"), "gpt-4".to_string());
        intent_to_model.insert(IntentId::from("reasoning_complex"), "gpt-4".to_string());
        intent_to_model.insert(IntentId::from("writing_creative"), "claude-3".to_string());
        intent_to_model.insert(IntentId::from("safety_check"), "claude-3".to_string());
        intent_to_model.insert(IntentId::from("general_question"), "llama-70b".to_string());
        intent_to_model.insert(IntentId::from("fast_response"), "llama-70b".to_string());
        
        Ok(Self {
            classifier,
            models,
            intent_to_model,
        })
    }
    
    async fn add_specialized_training_data(&self) -> Result<(), Box<dyn std::error::Error>> {
        let specialized_examples = vec![
            // Code-related intents
            TrainingExample {
                text: "debug this Python function".to_string(),
                intent: IntentId::from("code_debug"),
                confidence: 1.0,
                source: TrainingSource::Programmatic,
            },
            TrainingExample {
                text: "refactor this code to make it more efficient".to_string(),
                intent: IntentId::from("code_refactor"),
                confidence: 1.0,
                source: TrainingSource::Programmatic,
            },
            TrainingExample {
                text: "find bugs in this JavaScript".to_string(),
                intent: IntentId::from("code_debug"),
                confidence: 1.0,
                source: TrainingSource::Programmatic,
            },
            
            // Complex reasoning
            TrainingExample {
                text: "solve this complex mathematical problem step by step".to_string(),
                intent: IntentId::from("reasoning_complex"),
                confidence: 1.0,
                source: TrainingSource::Programmatic,
            },
            TrainingExample {
                text: "explain the philosophical implications of this concept".to_string(),
                intent: IntentId::from("reasoning_complex"),
                confidence: 1.0,
                source: TrainingSource::Programmatic,
            },
            
            // Creative writing
            TrainingExample {
                text: "write a creative story about space exploration".to_string(),
                intent: IntentId::from("writing_creative"),
                confidence: 1.0,
                source: TrainingSource::Programmatic,
            },
            TrainingExample {
                text: "compose a poem about nature".to_string(),
                intent: IntentId::from("writing_creative"),
                confidence: 1.0,
                source: TrainingSource::Programmatic,
            },
            
            // Safety checks
            TrainingExample {
                text: "is this content appropriate for children".to_string(),
                intent: IntentId::from("safety_check"),
                confidence: 1.0,
                source: TrainingSource::Programmatic,
            },
            TrainingExample {
                text: "check this text for harmful content".to_string(),
                intent: IntentId::from("safety_check"),
                confidence: 1.0,
                source: TrainingSource::Programmatic,
            },
            
            // General questions
            TrainingExample {
                text: "what's the weather like today".to_string(),
                intent: IntentId::from("general_question"),
                confidence: 1.0,
                source: TrainingSource::Programmatic,
            },
            TrainingExample {
                text: "tell me a joke".to_string(),
                intent: IntentId::from("fast_response"),
                confidence: 1.0,
                source: TrainingSource::Programmatic,
            },
        ];
        
        for example in specialized_examples {
            self.classifier.add_training_example(example).await?;
        }
        
        Ok(())
    }
    
    async fn route_task(&self, task: &str) -> Result<TaskRouting, Box<dyn std::error::Error>> {
        let request = ClassificationRequest {
            text: task.to_string(),
            context: None,
            include_alternatives: true,
            include_reasoning: true,
        };
        
        let response = self.classifier.classify(request).await?;
        let intent = &response.prediction.intent;
        
        // Find the appropriate model for this intent
        let model_id = self.intent_to_model.get(intent)
            .unwrap_or(&"llama-70b".to_string()) // Default to fastest model
            .clone();
        
        let model = self.models.get(&model_id).unwrap().clone();
        
        // Calculate estimated cost and time
        let estimated_tokens = task.len() / 4; // Rough estimate
        let estimated_cost = estimated_tokens as f64 * model.cost_per_token;
        
        Ok(TaskRouting {
            task: task.to_string(),
            intent: intent.clone(),
            confidence: response.prediction.confidence,
            selected_model: model.clone(),
            reasoning: response.prediction.reasoning,
            alternatives: response.prediction.alternative_intents,
            estimated_cost,
            estimated_time_ms: model.response_time_ms,
            processing_time_ms: response.processing_time_ms,
        })
    }
    
    async fn get_orchestrator_stats(&self) -> Result<OrchestratorStats, Box<dyn std::error::Error>> {
        let classifier_stats = self.classifier.get_stats().await;
        
        Ok(OrchestratorStats {
            classifier_stats,
            available_models: self.models.len(),
            intent_mappings: self.intent_to_model.len(),
        })
    }
}

#[derive(Debug, Clone)]
struct TaskRouting {
    task: String,
    intent: IntentId,
    confidence: intent_classifier::Confidence,
    selected_model: LLMModel,
    reasoning: String,
    alternatives: Vec<(IntentId, intent_classifier::Confidence)>,
    estimated_cost: f64,
    estimated_time_ms: u64,
    processing_time_ms: f64,
}

#[derive(Debug)]
struct OrchestratorStats {
    classifier_stats: intent_classifier::ClassifierStats,
    available_models: usize,
    intent_mappings: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🔄 Multi-LLM Orchestration Example");
    println!("==================================");
    
    // Create orchestrator
    println!("\n🚀 Creating multi-LLM orchestrator...");
    let orchestrator = MultiLLMOrchestrator::new().await?;
    
    // Add specialized training data
    println!("📚 Adding specialized training data...");
    orchestrator.add_specialized_training_data().await?;
    
    // Show orchestrator stats
    let stats = orchestrator.get_orchestrator_stats().await?;
    println!("   📊 Classifier training examples: {}", stats.classifier_stats.training_examples);
    println!("   🤖 Available models: {}", stats.available_models);
    println!("   🎯 Intent mappings: {}", stats.intent_mappings);
    
    // Test task routing
    println!("\n🎯 Testing task routing:");
    
    let test_tasks = vec![
        "debug this Python function that's not working correctly",
        "write a creative story about a robot learning to love",
        "solve this complex calculus problem step by step",
        "is this content safe for a family-friendly website",
        "what's the capital of France",
        "refactor this code to improve performance",
        "analyze this dataset and find patterns",
        "tell me a quick joke",
        "check this text for inappropriate language",
        "explain quantum computing in simple terms",
    ];
    
    for task in &test_tasks {
        let routing = orchestrator.route_task(task).await?;
        
        println!("\n   📝 Task: \"{}\"", task);
        println!("      🎯 Intent: {} (confidence: {:.3})", routing.intent, routing.confidence.value());
        println!("      🤖 Model: {} (${:.4} est. cost, {}ms est. time)", 
                 routing.selected_model.name, routing.estimated_cost, routing.estimated_time_ms);
        println!("      💭 Reasoning: {}", routing.reasoning);
        
        if !routing.alternatives.is_empty() {
            println!("      🔄 Alternatives: {:?}", 
                     routing.alternatives.iter()
                         .map(|(intent, conf)| format!("{}({:.3})", intent, conf.value()))
                         .collect::<Vec<_>>());
        }
    }
    
    // Show cost analysis
    println!("\n💰 Cost Analysis:");
    let mut total_cost = 0.0;
    let mut total_time = 0;
    
    for task in &test_tasks {
        let routing = orchestrator.route_task(task).await?;
        total_cost += routing.estimated_cost;
        total_time += routing.estimated_time_ms;
    }
    
    println!("   Total estimated cost: ${:.4}", total_cost);
    println!("   Total estimated time: {}ms", total_time);
    println!("   Average cost per task: ${:.4}", total_cost / test_tasks.len() as f64);
    
    // Show model utilization
    println!("\n📈 Model Utilization:");
    let mut model_usage = HashMap::new();
    
    for task in &test_tasks {
        let routing = orchestrator.route_task(task).await?;
        *model_usage.entry(routing.selected_model.name.clone()).or_insert(0) += 1;
    }
    
    for (model, count) in model_usage {
        let percentage = (count as f64 / test_tasks.len() as f64) * 100.0;
        println!("   🤖 {}: {} tasks ({:.1}%)", model, count, percentage);
    }
    
    println!("\n✅ Multi-LLM orchestration example completed successfully!");
    
    Ok(())
}
