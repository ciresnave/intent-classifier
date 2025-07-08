<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Intent Classification Library - Copilot Instructions

This is a Rust library for intent classification using few-shot learning and natural language processing.

## Library Overview

The intent classification library provides:
- Few-shot learning for intent classification
- Bootstrap training data for common intents
- Feedback learning to improve accuracy
- Async API for non-blocking operations
- Serializable training data (JSON export/import)
- Configurable behavior

## Key Components

1. **IntentClassifier**: Main classifier engine (`src/classifier.rs`)
2. **Types**: Core data structures and error types (`src/types.rs`)
3. **Examples**: Usage examples for different scenarios (`examples/`)

## Coding Guidelines

- Use async/await pattern for all operations
- Leverage Rust's type system for safety
- Implement proper error handling with `Result<T, IntentError>`
- Use `tracing` for logging and debugging
- Follow Rust naming conventions
- Write comprehensive documentation with examples

## Dependencies

- `tokio`: Async runtime
- `serde`: Serialization/deserialization
- `dashmap`: Thread-safe concurrent HashMap
- `ahash`: Fast hashing
- `tracing`: Logging and instrumentation
- `chrono`: Date/time handling
- `thiserror`: Error handling

## Architecture Patterns

- Use `Arc<RwLock<T>>` for shared mutable state
- Use `DashMap` for concurrent access to collections
- Implement `Default` trait for configuration structs
- Use builder pattern for complex request objects
- Follow the async/await pattern consistently

## Testing

- Write unit tests for each module
- Include integration tests in `src/lib.rs`
- Test both success and error cases
- Use `tokio_test` for async test utilities
- Include performance benchmarks where appropriate

## Examples

When creating examples:
- Include comprehensive error handling
- Show real-world use cases
- Demonstrate different API patterns
- Include performance measurements
- Add detailed explanations and comments

## API Design

- Keep the public API simple and intuitive
- Use builder patterns for complex configurations
- Provide both simple and advanced API variants
- Include detailed documentation with examples
- Follow Rust API guidelines
