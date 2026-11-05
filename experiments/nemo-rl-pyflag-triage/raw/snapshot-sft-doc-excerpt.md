# NVlabs sft.md 摘录（行 471-547，上游 d82b79c 逐字节）

471:     dataset_name: openai_format
472:     data_path: <PathToTrainingDataset>       # Path to training data
473:     chat_key: "messages"                     # Key for messages in the data (default: "messages")
474:     system_key: null                         # Key for system message in the data (optional)
475:     system_prompt: null                      # Default system prompt if not in data (optional)
476:     tool_key: "tools"                        # Key for tools in the data (default: "tools")
477:     use_preserving_dataset: false            # Set to true for heterogeneous tool schemas (see below)
478:   validation:
479:     ...
480: ```
481: 
482: #### Data Format
483: 
484: Your JSONL files should contain one JSON object per line following the [OpenAI Chat Completions function calling format](https://platform.openai.com/docs/guides/function-calling):
485: 
486: ```json
487: {
488:   "messages": [
489:     {"role": "system", "content": "You are a helpful assistant."},
490:     {"role": "user", "content": "What's the weather in Paris?"},
491:     {"role": "assistant", "content": "I'll check the weather for you.", "tool_calls": [
492:       {
493:         "id": "call_123",
494:         "type": "function",
495:         "function": {
496:           "name": "get_weather",
497:           "arguments": {"city": "Paris", "unit": "celsius"}
498:         }
499:       }
500:     ]},
501:     {"role": "tool", "content": "22°C, sunny", "tool_call_id": "call_123"},
502:     {"role": "assistant", "content": "The weather in Paris is currently 22°C and sunny."}
503:   ],
504:   "tools": [
505:     {
506:       "type": "function",
507:       "name": "get_weather",
508:       "description": "Get current weather for a city",
509:       "parameters": {
510:         "type": "object",
511:         "properties": {
512:           "city": {"type": "string", "description": "City name"},
513:           "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
514:         },
515:         "required": ["city"]
516:       }
517:     }
518:   ]
519: }
520: ```
521: 
522: > [!NOTE]
523: > NeMo RL passes `messages` and `tools` directly to the tokenizer's `apply_chat_template()`, so correct tool call rendering also depends on the model's chat template supporting this format.
524: 
525: #### Tool Calling with Heterogeneous Schemas
526: 
527: When your dataset contains tools with different argument structures (heterogeneous schemas), you should enable `use_preserving_dataset: true` to avoid data corruption:
528: 
529: ```yaml
530: data:
531:   dataset_name: openai_format
532:   ...
533:   use_preserving_dataset: true  # IMPORTANT: Enable this for tool calling datasets
534: ```
535: 
536: **Why this matters:** Standard HuggingFace dataset loading enforces uniform schemas by adding `None` values for missing keys. For example:
537: - Tool A has arguments: `{"query": "search term"}`
538: - Tool B has arguments: `{"expression": "2+2", "precision": 2}`
539: 
540: Without `use_preserving_dataset: true`, the loader would incorrectly add:
541: - Tool A becomes: `{"query": "search term", "expression": None, "precision": None}`
542: - Tool B becomes: `{"query": None, "expression": "2+2", "precision": 2}`
543: 
544: This corrupts your training data and can lead to models generating invalid tool calls. The `PreservingDataset` mode maintains the exact structure of each tool call.
545: 
546: 
547: ## Evaluate the Trained Model
