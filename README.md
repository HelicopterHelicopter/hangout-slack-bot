# Hangout Session Slack Bot

A Slack bot to manage tech team hangout sessions and presentations.

## Features

- Maintain a pipeline of upcoming presentations
- Add presentations with or without dates
- Update presentation dates
- Anonymous polling for topic interest

## Setup

1. Create a Slack App in your workspace
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env`:
   ```
   SLACK_BOT_TOKEN=your-bot-token
   SLACK_SIGNING_SECRET=your-signing-secret
   ```
4. Run the bot:
   ```bash
   python app.py
   ```

## Commands

- `/add-presentation` - Add a new presentation to the pipeline
- `/list-presentations` - View all upcoming presentations
- `/update-date` - Assign/update date for a presentation
- `/create-poll` - Create an anonymous poll for a topic
- `/view-polls` - View active polls
