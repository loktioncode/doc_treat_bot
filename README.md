# MedAssist AI

A medical assistant AI system that helps accelerate medical insights and provides healthcare support through natural language processing and medical image analysis.

## Features
- Medical diagnosis support and symptom analysis
- Medical image analysis (X-rays, CT scans, MRI)
- Treatment recommendations
- Medical research assistance
- Patient education
- Healthcare guidelines
- Medical equipment recommendations
- Healthcare provider recommendations

## Installation
```bash
pip install -r requirements.txt
```

## Start API
```bash
uvicorn app:app --host 127.0.0.1 --port 8001 --reload
```

## Setup Twilio
1. Create a Twilio account
2. Set up a WhatsApp sandbox
3. Configure your environment variables

## Setup ngrok webhook for Twilio
```bash
ngrok http://localhost:8001
#custom domain
ngrok http --url=ideal-krill-moderately.ngrok-free.app 80
```

## Start Celery Tasks
```bash
# Start Redis on localhost:55000
redis-server --port 55000

# Start Celery worker
celery -A celery_config worker --loglevel=info

# Start Celery beat
celery -A celery_config beat --loglevel=info
```

## Environment Variables
Create a `.env` file with the following variables:
```
GOOGLE_API_KEY=your_google_api_key
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_NUMBER=your_twilio_number
WEATHER_KEY=your_weather_api_key
EXA_KEY=your_exa_api_key
```
