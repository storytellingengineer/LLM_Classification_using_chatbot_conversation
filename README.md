# Chatbot Conversation Classifier

This project implements an LLM-based classifier that analyzes chatbot conversations to determine sentiment and context. It provides a FastAPI-based REST API for easy integration.

## Features

- Sentiment analysis of chatbot conversations
- Context classification
- REST API endpoints for classification
- Docker support for easy deployment
- FastAPI for high-performance API serving

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── classifier.py
│   └── utils.py
├── tests/
│   └── test_classifier.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd chatbot-conversation-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run with Docker:
```bash
docker-compose up --build
```

## API Usage

The API provides the following endpoints:

### POST /classify
Classifies a chatbot conversation based on sentiment and context.

Request body:
```json
{
    "conversation": [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"}
    ]
}
```

Response:
```json
{
    "sentiment": "positive",
    "context": "greeting",
    "confidence_score": 0.95
}
```

## Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install development dependencies:
```bash
pip install -r requirements.txt
```

3. Run tests:
```bash
pytest tests/
```

## License

MIT License 
