import os
from datetime import datetime
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import database as db
from dateutil import parser
import logging
import re
import json # Add json import if needed for formatting, using YAML here

# Set up logging to debug the issue
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check if we're in test mode
TEST_MODE = os.environ.get('TEST_MODE', 'False').lower() == 'true'

channel_name = os.environ.get('SLACK_CHANNEL')

ADMIN_USER_ID = "U07ADRA6HEH" # User ID to receive the DM

if TEST_MODE:
    # Mock App for testing
    class MockApp:
        def __init__(self):
            self.commands = {}
            self.views = {}
            self.actions = {}
            
        def command(self, cmd):
            def decorator(func):
                self.commands[cmd] = func
                return func
            return decorator
            
        def view(self, view_id):
            def decorator(func):
                self.views[view_id] = func
                return func
            return decorator
        
        def action(self, action_id):
            def decorator(func):
                self.actions[action_id] = func
                return func
            return decorator
    
    app = MockApp()
    print("Running in TEST MODE - no Slack connection will be established")
else:
    # Initialize the Slack app
    app = App(token=os.environ["SLACK_BOT_TOKEN"])

# Initialize database
db.init_db()
# Run migration to update poll_responses table if needed
db.migrate_poll_responses_table()

# Add middleware to log all incoming requests for better debugging
@app.middleware
def log_request(logger, body, next):
    logger.debug(f"Received request: {body}")
    return next()

@app.command("/add-hangout")
def add_presentation(ack, body, client):
    ack()
    
    # Get today's date for the datepicker
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Open a modal for adding presentation
    client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal",
            "callback_id": "add_presentation_modal",
            "title": {"type": "plain_text", "text": "Add Hangout"},
            "submit": {"type": "plain_text", "text": "Submit"},
            "blocks": [
                {
                    "type": "input",
                    "block_id": "topic_block",
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "topic_input",
                        "placeholder": {"type": "plain_text", "text": "Enter hangout topic"}
                    },
                    "label": {"type": "plain_text", "text": "Topic"}
                },
                {
                    "type": "input",
                    "block_id": "presenter_block",
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "presenter_input",
                        "placeholder": {"type": "plain_text", "text": "Enter presenter name"}
                    },
                    "label": {"type": "plain_text", "text": "Presenter"}
                },
                {
                    "type": "input",
                    "block_id": "date_block",
                    "element": {
                        "type": "datepicker",
                        "action_id": "date_input",
                        "initial_date": today,
                        "placeholder": {"type": "plain_text", "text": "Select a Thursday (optional)"},
                        "confirm": {
                            "title": {"type": "plain_text", "text": "Confirm Date"},
                            "text": {"type": "plain_text", "text": "Please ensure you've selected a Thursday as hangouts are only held on Thursdays."},
                            "confirm": {"type": "plain_text", "text": "Yes"},
                            "deny": {"type": "plain_text", "text": "No"}
                        }
                    },
                    "label": {"type": "plain_text", "text": "Date (Must be a Thursday)"},
                    "optional": True
                }
            ]
        }
    )

@app.view("add_presentation_modal")
def handle_add_presentation_submission(ack, body, client):
    values = body["view"]["state"]["values"]
    topic = values["topic_block"]["topic_input"]["value"]
    presenter = values["presenter_block"]["presenter_input"]["value"]
    date_value = values["date_block"]["date_input"].get("selected_date")
    
    if date_value:
        selected_date = parser.parse(date_value)
        # Check if selected date is a Thursday (weekday() returns 3 for Thursday)
        if selected_date.weekday() != 3:
            ack({
                "response_action": "errors",
                "errors": {
                    "date_block": "Hangouts can only be scheduled for Thursdays"
                }
            })
            return
            
        # Check if date is in the past
        today = datetime.now().date()
        if selected_date.date() < today:
            ack({
                "response_action": "errors",
                "errors": {
                    "date_block": "Cannot schedule hangouts for past dates"
                }
            })
            return
    
    scheduled_date = parser.parse(date_value) if date_value else None
    status = "scheduled" if scheduled_date else "pending"
    presentation = db.add_presentation(topic, presenter, scheduled_date, status)
    
    # Acknowledge the submission
    ack()
    
    # Notify channel about new presentation
    client.chat_postMessage(
        channel=channel_name,  # Replace with your channel
        text=f"New hangout added!\n*Topic:* {topic}\n*Presenter:* {presenter}" +
             (f"\n*Scheduled for:* {scheduled_date.strftime('%B %d, %Y')}" if scheduled_date else "")
    )

@app.command("/view-pipeline")
def view_pipeline(ack, respond, client, body=None):
    ack()
    presentations = db.get_presentations(include_completed=True)
    
    channel_id = body["channel_id"] if body else channel_name
    
    if not presentations:
        client.chat_postMessage(
            channel=channel_id,
            text="No Hangouts in the pipeline!"
        )
        return
    
    # Group presentations by status
    scheduled = [p for p in presentations if p['status'] == 'scheduled' and p['scheduled_date']]
    pending = [p for p in presentations if p['status'] == 'pending' or not p['scheduled_date']]
    completed = [p for p in presentations if p['status'] == 'completed']
    
    blocks = [{
        "type": "header",
        "text": {"type": "plain_text", "text": "Hangouts Pipeline", "emoji": True}
    }]
    
    # Add scheduled presentations
    if scheduled:
        blocks.append({
            "type": "header",
            "text": {"type": "plain_text", "text": "📅 Scheduled hangouts", "emoji": True}
        })
        
        # Sort by scheduled date
        scheduled.sort(key=lambda p: p['scheduled_date'])
        
        for p in scheduled:
            date_text = p['scheduled_date'].strftime('%B %d, %Y') if p['scheduled_date'] else "Date not set"
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Topic:* {p['topic']}\n*Presenter:* {p['presenter']}\n*Date:* {date_text}"
                }
            })
            
            # Add buttons for actions
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✅ Mark as Completed", "emoji": True},
                        "style": "primary",
                        "action_id": f"mark_complete_{p['id']}",
                        "value": str(p['id']),
                        "confirm": {
                            "title": {"type": "plain_text", "text": "Mark as Completed?"},
                            "text": {"type": "plain_text", "text": f"Are you sure the hangout '{p['topic']}' has been completed?"},
                            "confirm": {"type": "plain_text", "text": "Yes, Mark Complete"},
                            "deny": {"type": "plain_text", "text": "Cancel"}
                        }
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "📆 Reschedule", "emoji": True},
                        "action_id": f"schedule_presentation_{p['id']}",
                        "value": str(p['id'])
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "🗑️ Delete", "emoji": True},
                        "style": "danger",
                        "action_id": f"delete_hangout_{p['id']}",
                        "value": str(p['id']),
                        "confirm": {
                            "title": {"type": "plain_text", "text": "Delete Hangout?"},
                            "text": {"type": "plain_text", "text": f"Are you sure you want to delete the hangout '{p['topic']}'?"},
                            "confirm": {"type": "plain_text", "text": "Yes, Delete"},
                            "deny": {"type": "plain_text", "text": "Cancel"}
                        }
                    }
                ]
            })
        blocks.append({"type": "divider"})
    
    # Add pending presentations
    if pending:
        blocks.append({
            "type": "header",
            "text": {"type": "plain_text", "text": "⏳ Pending Hangouts", "emoji": True}
        })
        
        for p in pending:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Topic:* {p['topic']}\n*Presenter:* {p['presenter']}\n*Status:* Waiting to be scheduled"
                }
            })
            
            # Add buttons for actions
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "📅 Schedule", "emoji": True},
                        "style": "primary",
                        "action_id": f"schedule_presentation_{p['id']}",
                        "value": str(p['id'])
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "🗑️ Delete", "emoji": True},
                        "style": "danger",
                        "action_id": f"delete_hangout_{p['id']}",
                        "value": str(p['id']),
                        "confirm": {
                            "title": {"type": "plain_text", "text": "Delete Hangout?"},
                            "text": {"type": "plain_text", "text": f"Are you sure you want to delete the hangout '{p['topic']}'?"},
                            "confirm": {"type": "plain_text", "text": "Yes, Delete"},
                            "deny": {"type": "plain_text", "text": "Cancel"}
                        }
                    }
                ]
            })
        blocks.append({"type": "divider"})
    
    # Add completed presentations
    if completed:
        blocks.append({
            "type": "header",
            "text": {"type": "plain_text", "text": "✅ Completed Presentations", "emoji": True}
        })
        
        # Show most recent first
        completed.sort(key=lambda p: p['scheduled_date'] if p['scheduled_date'] else p['created_at'], reverse=True)
        
        for p in completed[:5]:  # Show only the 5 most recent
            date_text = p['scheduled_date'].strftime('%B %d, %Y') if p['scheduled_date'] else "Unknown date"
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Topic:* {p['topic']}\n*Presenter:* {p['presenter']}\n*Date:* {date_text}"
                }
            })
            
            # Add delete button for completed presentations
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "🗑️ Delete", "emoji": True},
                        "style": "danger",
                        "action_id": f"delete_hangout_{p['id']}",
                        "value": str(p['id']),
                        "confirm": {
                            "title": {"type": "plain_text", "text": "Delete Hangout?"},
                            "text": {"type": "plain_text", "text": f"Are you sure you want to delete the hangout '{p['topic']}'?"},
                            "confirm": {"type": "plain_text", "text": "Yes, Delete"},
                            "deny": {"type": "plain_text", "text": "Cancel"}
                        }
                    }
                ]
            })
        
        if len(completed) > 5:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"_...and {len(completed) - 5} more completed presentations_"}
            })
    
    # Add tips section
    blocks.extend([
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "💡 *Tips:*\n• Use `/add-hangout` to add a new hangout\n• Fear that your topic sucks? Just create an anonymous poll to check interest! Generally everyone is interested in something, even if it's just to learn more about something."
            }
        }
    ])
    
    # Send message to channel instead of an ephemeral message
    client.chat_postMessage(
        channel=channel_id,
        blocks=blocks
    )

@app.action(re.compile("mark_complete_\\d+"))
def handle_mark_complete(ack, body, respond, client, logger):
    ack()
    
    # Extract presentation ID from the action_id
    action_id = body["actions"][0]["action_id"]
    presentation_id = int(action_id.split("_")[-1])
    
    logger.debug(f"Marking presentation ID {presentation_id} as completed")
    
    # Mark the presentation as completed
    success = db.mark_presentation_completed(presentation_id)
    
    if success:
        # Post a confirmation message
        client.chat_postMessage(
            channel=body["channel"]["id"],
            text=f"Presentation has been marked as completed!"
        )
        
        # Send a new message with the updated pipeline view
        view_pipeline(
            lambda: None,  # ack replacement
            None,  # respond replacement (not used)
            client, 
            {"channel_id": body["channel"]["id"]}  # body with channel_id
        )
    else:
        # Handle error
        client.chat_postMessage(
            channel=body["channel"]["id"],
            text="Sorry, there was an error marking the presentation as completed."
        )

@app.command("/complete-presentation")
def complete_presentation(ack, body, client):
    ack()
    
    presentations = db.get_presentations()
    scheduled = [p for p in presentations if p['status'] == 'scheduled' and p['scheduled_date']]
    
    if not scheduled:
        client.chat_postMessage(
            channel=body["channel_id"],
            text="There are no scheduled presentations to mark as completed."
        )
        return
    
    # Create options for the presentations
    options = []
    for p in scheduled:
        date_text = p['scheduled_date'].strftime('%B %d, %Y') if p['scheduled_date'] else "Date not set"
        options.append({
            "text": {"type": "plain_text", "text": f"{p['topic']} by {p['presenter']} ({date_text})"},
            "value": str(p['id'])
        })
    
    # Open a modal to select which presentation to mark as completed
    client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal",
            "callback_id": "complete_presentation_modal",
            "title": {"type": "plain_text", "text": "Mark Presentation as Completed"},
            "submit": {"type": "plain_text", "text": "Mark Complete"},
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Select a presentation to mark as completed:"
                    }
                },
                {
                    "type": "section",
                    "block_id": "presentation_select_block",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Choose a presentation:"
                    },
                    "accessory": {
                        "type": "static_select",
                        "action_id": "presentation_select",
                        "placeholder": {
                            "type": "plain_text",
                            "text": "Select a presentation"
                        },
                        "options": options
                    }
                }
            ]
        }
    )

@app.view("complete_presentation_modal")
def handle_complete_presentation(ack, body, client):
    ack()
    
    values = body["view"]["state"]["values"]
    presentation_id = int(values["presentation_select_block"]["presentation_select"]["selected_option"]["value"])
    
    # Mark the presentation as completed
    success = db.mark_presentation_completed(presentation_id)
    
    if success:
        # Notify the channel
        client.chat_postMessage(
            channel=channel_name,
            text=f"A presentation has been marked as completed! Use `/view-pipeline` to see the updated presentation pipeline."
        )

@app.command("/create-poll")
def create_poll(ack, body, client):
    ack()
    
    client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal",
            "callback_id": "create_poll_modal",
            "title": {"type": "plain_text", "text": "Create Topic Poll"},
            "submit": {"type": "plain_text", "text": "Create Poll"},
            "blocks": [
                {
                    "type": "input",
                    "block_id": "poll_topic_block",
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "poll_topic_input",
                        "placeholder": {"type": "plain_text", "text": "Enter topic to poll"}
                    },
                    "label": {"type": "plain_text", "text": "Topic"}
                }
            ]
        }
    )

@app.view("create_poll_modal")
def handle_create_poll_submission(ack, body, client):
    ack()
    
    topic = body["view"]["state"]["values"]["poll_topic_block"]["poll_topic_input"]["value"]
    user_id = body["user"]["id"]
    
    poll = db.create_poll(topic, user_id)
    
    # Post poll to channel
    client.chat_postMessage(
        channel=channel_name,  # Replace with your channel
        blocks=[
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Anonymous Poll:* Are you interested in a hangout about:\n>{topic}"}
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "👍 Interested"},
                        "action_id": f"poll_vote_yes_{poll['id']}",  # Unique action_id for "yes" vote
                        "value": f"yes_{poll['id']}",
                        "style": "primary"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "👎 Not Interested"},
                        "action_id": f"poll_vote_no_{poll['id']}",  # Unique action_id for "no" vote
                        "value": f"no_{poll['id']}",
                        "style": "danger"
                    }
                ]
            }
        ]
    )

@app.action(re.compile("poll_vote_yes_\\d+"))
def handle_poll_vote_yes(ack, body, logger, client):
    ack()
    # Extract poll ID from the action_id
    action_id = body["actions"][0]["action_id"]
    poll_id = int(action_id.split("_")[-1])
    
    # Get user ID from the request
    user_id = body["user"]["id"]
    
    # Record the vote as interested (yes)
    logger.debug(f"Recording poll yes vote: poll_id={poll_id}, user_id={user_id}")
    db.add_poll_response(poll_id, True, user_id)
    
    # Send a confirmation message to just the user
    client.chat_postEphemeral(
        channel=body["channel"]["id"],
        user=user_id,
        text=f"Your vote has been recorded. You voted: 👍 interested"
    )

@app.action(re.compile("poll_vote_no_\\d+"))
def handle_poll_vote_no(ack, body, logger, client):
    ack()
    # Extract poll ID from the action_id
    action_id = body["actions"][0]["action_id"]
    poll_id = int(action_id.split("_")[-1])
    
    # Get user ID from the request
    user_id = body["user"]["id"]
    
    # Record the vote as not interested (no)
    logger.debug(f"Recording poll no vote: poll_id={poll_id}, user_id={user_id}")
    db.add_poll_response(poll_id, False, user_id)
    
    # Send a confirmation message to just the user
    client.chat_postEphemeral(
        channel=body["channel"]["id"],
        user=user_id,
        text=f"Your vote has been recorded. You voted: 👎 not interested"
    )

@app.command("/view-polls")
def view_polls(ack, respond, client, body=None):
    ack()
    polls = db.get_active_polls()
    
    channel_id = body["channel_id"] if body else channel_name
    
    if not polls:
        client.chat_postMessage(
            channel=channel_id,
            text="No active polls!"
        )
        return
    
    blocks = [{
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*Active Polls:*"}
    }]
    
    for poll in polls:
        results = db.get_poll_results(poll['id'])
        
        # Add poll info
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Topic:* {poll['topic']}\n" +
                       f"👍 Interested: {results['interested']}\n" +
                       f"👎 Not Interested: {results['not_interested']}\n" +
                       f"Total Responses: {results['total_responses']}"
            }
        })
        
        # Add delete button with exact action_id
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "🗑️ Delete Poll", "emoji": True},
                    "style": "danger",
                    "action_id": f"delete_poll_{poll['id']}",  # Direct action_id match
                    "value": str(poll['id']),  # Store poll ID in value
                    "confirm": {
                        "title": {"type": "plain_text", "text": "Delete Poll?"},
                        "text": {"type": "plain_text", "text": "Are you sure you want to delete this poll?"},
                        "confirm": {"type": "plain_text", "text": "Yes, Delete"},
                        "deny": {"type": "plain_text", "text": "Cancel"}
                    }
                }
            ]
        })
        
        # Add a divider between polls
        if polls.index(poll) < len(polls) - 1:
            blocks.append({"type": "divider"})
    
    client.chat_postMessage(
        channel=channel_id,
        blocks=blocks
    )

@app.action(re.compile("delete_poll_\\d+"))
def handle_delete_poll(ack, body, respond, client, logger):
    ack()
    
    # Extract poll ID from the action_id
    action_id = body["actions"][0]["action_id"]
    poll_id = int(action_id.split("_")[-1])
    
    logger.debug(f"Handling delete for poll ID: {poll_id}")
    handle_poll_deletion(poll_id, body, respond, client)

# Common function for poll deletion
def handle_poll_deletion(poll_id, body, respond, client):
    # Get the poll details for confirmation message
    polls = db.get_active_polls()
    poll_to_delete = next((p for p in polls if p['id'] == poll_id), None)
    
    if poll_to_delete:
        # Delete the poll
        success = db.delete_poll(poll_id)
        
        if success:
            # Send a confirmation message
            client.chat_postMessage(
                channel=body["channel"]["id"],
                text=f"Poll *{poll_to_delete['topic']}* has been deleted."
            )
            
            # Send an updated list of polls
            view_polls(
                lambda: None,  # ack replacement
                None,  # respond replacement (not used)
                client,
                {"channel_id": body["channel"]["id"]}  # body with channel_id
            )
        else:
            # Fallback notification
            client.chat_postMessage(
                channel=body["channel"]["id"],
                text="There was an error deleting the poll."
            )

@app.action(re.compile("schedule_presentation_\\d+"))
def handle_schedule_presentation(ack, body, respond, client, logger):
    ack()
    
    # Extract presentation ID from the action_id
    action_id = body["actions"][0]["action_id"]
    presentation_id = int(action_id.split("_")[-1])
    
    logger.debug(f"Opening date picker for presentation ID {presentation_id}")
    handle_schedule_presentation_action(presentation_id, body, respond, client, logger)

# Common function for presentation scheduling
def handle_schedule_presentation_action(presentation_id, body, respond, client, logger):
    logger.debug(f"Opening date picker for presentation ID {presentation_id}")
    
    # Get presentation details
    presentation = db.get_presentation(presentation_id)
    
    if not presentation:
        client.chat_postMessage(
            channel=body["channel"]["id"],
            text="Sorry, that presentation doesn't exist anymore."
        )
        return
    
    # Get today's date for the datepicker
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Create a modal with date picker focused on Thursdays
    client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal",
            "callback_id": "schedule_presentation_modal",
            "private_metadata": str(presentation_id),  # Store the presentation ID
            "title": {"type": "plain_text", "text": "Schedule Presentation"},
            "submit": {"type": "plain_text", "text": "Schedule"},
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Scheduling:* {presentation['topic']} by {presentation['presenter']}"
                    }
                },
                {
                    "type": "input",
                    "block_id": "date_block",
                    "element": {
                        "type": "datepicker",
                        "action_id": "selected_date",
                        "initial_date": today,
                        "placeholder": {"type": "plain_text", "text": "Select a Thursday"},
                        "confirm": {
                            "title": {"type": "plain_text", "text": "Confirm Date"},
                            "text": {"type": "plain_text", "text": "Please ensure you've selected a Thursday as hangouts are only held on Thursdays."},
                            "confirm": {"type": "plain_text", "text": "Yes"},
                            "deny": {"type": "plain_text", "text": "No"}
                        }
                    },
                    "label": {"type": "plain_text", "text": "Date (Must be a Thursday)"}
                }
            ]
        }
    )

@app.view("schedule_presentation_modal")
def handle_schedule_presentation_submission(ack, body, client, logger):
    # Get the values from the modal
    values = body["view"]["state"]["values"]
    selected_date = values["date_block"]["selected_date"]["selected_date"]
    presentation_id = int(body["view"]["private_metadata"])
    
    # Parse the date
    date_obj = parser.parse(selected_date)
    
    # Check if the selected date is a Thursday (weekday() returns 3 for Thursday)
    if date_obj.weekday() != 3:
        ack({
            "response_action": "errors",
            "errors": {
                "date_block": "hangouts can only be scheduled for Thursdays"
            }
        })
        return
    
    # Update the presentation with the new date
    success = db.update_presentation_status_and_date(
        presentation_id, 
        "scheduled", 
        date_obj
    )
    
    if success:
        # Acknowledge the submission
        ack()
        
        # Get the updated presentation details
        presentation = db.get_presentation(presentation_id)
        
        # Notify the channel
        client.chat_postMessage(
            channel=channel_name,
            text=f"Presentation scheduled!\n*Topic:* {presentation['topic']}\n*Presenter:* {presentation['presenter']}\n*Date:* {date_obj.strftime('%B %d, %Y')}"
        )
    else:
        # Handle the error case
        ack()
        logger.error(f"Failed to update presentation {presentation_id}")
        
        client.chat_postMessage(
            channel=channel_name,
            text="Sorry, there was an error scheduling the presentation."
        )

@app.command("/schedule-presentation")
def schedule_presentation_command(ack, body, client):
    ack()
    
    # Get pending presentations
    presentations = db.get_presentations()
    pending = [p for p in presentations if p['status'] == 'pending' or not p['scheduled_date']]
    
    if not pending:
        client.chat_postMessage(
            channel=body["channel_id"],
            text="There are no pending presentations to schedule."
        )
        return
    
    # Create options for the presentations
    options = []
    for p in pending:
        options.append({
            "text": {"type": "plain_text", "text": f"{p['topic']} by {p['presenter']}"},
            "value": str(p['id'])
        })
    
    # Open a modal to select which presentation to schedule
    client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal",
            "callback_id": "select_presentation_modal",
            "title": {"type": "plain_text", "text": "Schedule Presentation"},
            "submit": {"type": "plain_text", "text": "Next"},
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Select a pending presentation to schedule:"
                    }
                },
                {
                    "type": "section",
                    "block_id": "presentation_select_block",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Choose a presentation:"
                    },
                    "accessory": {
                        "type": "static_select",
                        "action_id": "presentation_select",
                        "placeholder": {
                            "type": "plain_text",
                            "text": "Select a presentation"
                        },
                        "options": options
                    }
                }
            ]
        }
    )

@app.view("select_presentation_modal")
def handle_select_presentation(ack, body, client):
    ack()
    
    # Get the selected presentation ID
    values = body["view"]["state"]["values"]
    presentation_id = int(values["presentation_select_block"]["presentation_select"]["selected_option"]["value"])
    
    # Get the presentation details
    presentation = db.get_presentation(presentation_id)
    
    # Get today's date for the datepicker
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Open the date selection modal
    client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal",
            "callback_id": "schedule_presentation_modal",
            "private_metadata": str(presentation_id),  # Store the presentation ID
            "title": {"type": "plain_text", "text": "Schedule Presentation"},
            "submit": {"type": "plain_text", "text": "Schedule"},
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Scheduling:* {presentation['topic']} by {presentation['presenter']}"
                    }
                },
                {
                    "type": "input",
                    "block_id": "date_block",
                    "element": {
                        "type": "datepicker",
                        "action_id": "selected_date",
                        "initial_date": today,
                        "placeholder": {"type": "plain_text", "text": "Select a Thursday"},
                        "confirm": {
                            "title": {"type": "plain_text", "text": "Confirm Date"},
                            "text": {"type": "plain_text", "text": "Please ensure you've selected a Thursday as hangouts are only held on Thursdays."},
                            "confirm": {"type": "plain_text", "text": "Yes"},
                            "deny": {"type": "plain_text", "text": "No"}
                        }
                    },
                    "label": {"type": "plain_text", "text": "Date (Must be a Thursday)"}
                }
            ]
        }
    )

@app.action(re.compile("delete_hangout_\\d+"))
def handle_delete_hangout(ack, body, client, logger):
    ack()
    
    # Extract presentation ID from the action_id
    action_id = body["actions"][0]["action_id"]
    presentation_id = int(action_id.split("_")[-1])
    
    logger.debug(f"Deleting presentation ID {presentation_id}")
    
    # Get presentation details before deletion
    presentation = db.get_presentation(presentation_id)
    
    if not presentation:
        client.chat_postMessage(
            channel=body["channel"]["id"],
            text="Sorry, that hangout doesn't exist anymore."
        )
        return
    
    # Delete the presentation
    success = db.delete_presentation(presentation_id)
    
    if success:
        # Send a confirmation message
        client.chat_postMessage(
            channel=body["channel"]["id"],
            text=f"Hangout *{presentation['topic']}* has been deleted."
        )
        
        # Send an updated pipeline view
        view_pipeline(
            lambda: None,  # ack replacement
            None,  # respond replacement (not used)
            client,
            {"channel_id": body["channel"]["id"]}  # body with channel_id
        )
    else:
        # Handle error
        client.chat_postMessage(
            channel=body["channel"]["id"],
            text="Sorry, there was an error deleting the hangout."
        )

# --- Archive Request Slash Command ---
@app.command("/archive-request")
def handle_archive_request_command(ack, body, client, logger):
    """Handles the /archive-request slash command and opens the initial modal."""
    ack()
    try:
        # Initial blocks - only common fields
        initial_blocks = [
            # --- Common Fields ---
            {
                "type": "input",
                "block_id": "source_name_block",
                "element": {"type": "plain_text_input", "action_id": "source_name_input"},
                "label": {"type": "plain_text", "text": "Source Name"},
            },
            {
                "type": "input",
                "block_id": "environment_block",
                "element": {
                    "type": "static_select",
                    "placeholder": {"type": "plain_text", "text": "Select environment"},
                    "action_id": "environment_select",
                    "options": [
                        {"text": {"type": "plain_text", "text": "Development"}, "value": "development"},
                        {"text": {"type": "plain_text", "text": "Staging"}, "value": "staging"},
                        {"text": {"type": "plain_text", "text": "Production"}, "value": "production"},
                    ]
                },
                "label": {"type": "plain_text", "text": "Environment"},
            },
            {
                "type": "input",
                "block_id": "source_type_block",
                "dispatch_action": True, # Trigger action on select
                "element": {
                    "type": "static_select",
                    "placeholder": {"type": "plain_text", "text": "Select source type"},
                    "action_id": "source_type_select",
                    "options": [
                        {"text": {"type": "plain_text", "text": "SQL"}, "value": "sql"},
                        {"text": {"type": "plain_text", "text": "MongoDB"}, "value": "mongodb"},
                        {"text": {"type": "plain_text", "text": "OpenSearch"}, "value": "opensearch"},
                        {"text": {"type": "plain_text", "text": "Elasticsearch"}, "value": "elasticsearch"},
                    ],
                },
                "label": {"type": "plain_text", "text": "Source Type"},
            },
            # --- Source Specific placeholder (initially empty) ---
            {
                "type": "context",
                "block_id": "source_specific_context",
                 "elements": [
                    {"type": "plain_text", "text": "Select a Source Type to see specific fields."}
                ]
            },
             # --- Common Fields continued ---
            { "type": "divider" },
            {
                "type": "input",
                "block_id": "destination_block",
                "element": {
                    "type": "static_select",
                    "placeholder": {"type": "plain_text", "text": "Select destination"},
                    "action_id": "destination_select",
                    "options": [
                        {"text": {"type": "plain_text", "text": "s3_archive"}, "value": "s3_archive"},
                        {"text": {"type": "plain_text", "text": "s3_glacier"}, "value": "s3_glacier"},
                    ]
                },
                "label": {"type": "plain_text", "text": "Destination"},
            },
            {
                "type": "input",
                "block_id": "delete_after_archive_block",
                "optional": True,
                "element": {
                    "type": "radio_buttons",
                    "action_id": "delete_after_archive_radio",
                    "initial_option": {"text": {"type": "plain_text", "text": "No"}, "value": "false"},
                    "options": [
                        {"text": {"type": "plain_text", "text": "Yes"}, "value": "true"},
                        {"text": {"type": "plain_text", "text": "No"}, "value": "false"}
                    ]
                },
                "label": {"type": "plain_text", "text": "Delete After Archive?"},
            },
            { "type": "divider" },
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Filter Details (Age)"}
            },
            {
                "type": "input", "block_id": "filter_days_block",
                "element": {"type": "number_input", "is_decimal_allowed": False, "action_id": "filter_days_input"},
                "label": {"type": "plain_text", "text": "Archive data older than (days)"},
            },
            {
                "type": "input", "block_id": "filter_date_column_block",
                "element": {"type": "plain_text_input", "action_id": "filter_date_column_input"},
                "label": {"type": "plain_text", "text": "Date Column/Field Name"},
                "hint": {"type": "plain_text", "text": "e.g., created_at, @timestamp, createdAt"}
            },
            { "type": "divider" },
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Other Options"}
            },
            {
                "type": "input", "block_id": "group_by_date_block", "optional": True,
                "element": {
                    "type": "radio_buttons", "action_id": "group_by_date_radio",
                     "initial_option": {"text": {"type": "plain_text", "text": "No"}, "value": "false"},
                     "options": [
                        {"text": {"type": "plain_text", "text": "Yes"}, "value": "true"},
                        {"text": {"type": "plain_text", "text": "No"}, "value": "false"}
                    ]
                },
                "label": {"type": "plain_text", "text": "Group By Date?"}, # Simplified label
            },
        ]

        result = client.views_open(
            trigger_id=body["trigger_id"],
            view={
                "type": "modal",
                "callback_id": "archive_request_submission",
                "title": {"type": "plain_text", "text": "Archive Request Form"},
                "submit": {"type": "plain_text", "text": "Submit"},
                "close": {"type": "plain_text", "text": "Cancel"},
                "blocks": initial_blocks # Use the initial set of blocks
            }
        )
        logger.info(f"Opened initial archive request modal for user {body['user_id']}")

    except Exception as e:
        logger.error(f"Error opening initial archive request modal: {e}")

# --- Action Handler for Source Type Selection ---
@app.action("source_type_select")
def update_archive_modal_for_source_type(ack, body, client, logger):
    """Updates the archive request modal when the source type is selected."""
    ack()
    view = body["view"]
    view_id = view["id"]
    hash_value = view["hash"] # Get the hash for updating
    all_current_blocks = view["blocks"]
    
    # Extract selected source type from the action payload
    selected_source_type = None
    actions = body.get("actions")
    if actions and len(actions) > 0:
        selected_option = actions[0].get("selected_option")
        if selected_option:
             selected_source_type = selected_option.get("value")

    # Fallback: try getting from view state if not in action payload (e.g., if view was previously updated)
    if not selected_source_type:
        try:
            selected_source_type = view["state"]["values"]["source_type_block"]["source_type_select"]["selected_option"]["value"]
        except (KeyError, TypeError):
             logger.warning("Could not determine selected source type from action or view state.")
             # Optionally return or keep the view as is
             return 

    logger.info(f"Source type selected: {selected_source_type}. Updating view {view_id}")

    # --- Define Source Specific Blocks --- 
    # (Definitions remain the same: sql_blocks, mongo_blocks, opensearch_blocks, elasticsearch_blocks)
    sql_blocks = [
        { "type": "divider", "block_id": "source_specific_divider" },
        { "type": "header", "block_id": "source_specific_header", "text": {"type": "plain_text", "text": "SQL Specific Details"} },
        {"type": "input", "block_id": "sql_table_block", "element": {"type": "plain_text_input", "action_id": "sql_table_input"}, "label": {"type": "plain_text", "text": "SQL: Table Name"}},        
        {"type": "input", "block_id": "sql_primary_key_block", "element": {"type": "plain_text_input", "action_id": "sql_primary_key_input"}, "label": {"type": "plain_text", "text": "SQL: Primary Key Column"}},
    ]
    mongo_blocks = [
        { "type": "divider", "block_id": "source_specific_divider" },
        { "type": "header", "block_id": "source_specific_header", "text": {"type": "plain_text", "text": "MongoDB Specific Details"} },
        {"type": "input", "block_id": "mongo_database_block", "element": {"type": "plain_text_input", "action_id": "mongo_database_input"}, "label": {"type": "plain_text", "text": "MongoDB: Database Name"}},        
        {"type": "input", "block_id": "mongo_collection_block", "element": {"type": "plain_text_input", "action_id": "mongo_collection_input"}, "label": {"type": "plain_text", "text": "MongoDB: Collection Name"}},
    ]
    opensearch_blocks = [
        { "type": "divider", "block_id": "source_specific_divider" },
        { "type": "header", "block_id": "source_specific_header", "text": {"type": "plain_text", "text": "OpenSearch Specific Details"} },
        {"type": "input", "block_id": "opensearch_index_block", "element": {"type": "plain_text_input", "action_id": "opensearch_index_input"}, "label": {"type": "plain_text", "text": "OpenSearch: Index Name"}},        
        {"type": "input", "block_id": "opensearch_use_scroll_block", "optional": True, "element": {"type": "radio_buttons", "action_id": "opensearch_use_scroll_radio", "initial_option": {"text": {"type": "plain_text", "text": "No"}, "value": "false"}, "options": [{"text": {"type": "plain_text", "text": "Yes"}, "value": "true"}, {"text": {"type": "plain_text", "text": "No"}, "value": "false"}]}, "label": {"type": "plain_text", "text": "OpenSearch: Use Scroll API?"}},
        {"type": "input", "block_id": "opensearch_scroll_size_block", "optional": True, "element": {"type": "number_input", "is_decimal_allowed": False, "action_id": "opensearch_scroll_size_input"}, "label": {"type": "plain_text", "text": "OpenSearch: Scroll Size"}},
    ]
    elasticsearch_blocks = [
        { "type": "divider", "block_id": "source_specific_divider" },
        { "type": "header", "block_id": "source_specific_header", "text": {"type": "plain_text", "text": "Elasticsearch Specific Details"} },
        {"type": "input", "block_id": "elasticsearch_index_block", "element": {"type": "plain_text_input", "action_id": "elasticsearch_index_input"}, "label": {"type": "plain_text", "text": "Elasticsearch: Index Name"}},
        {"type": "input", "block_id": "elasticsearch_use_scroll_block", "optional": True, "element": {"type": "radio_buttons", "action_id": "elasticsearch_use_scroll_radio", "initial_option": {"text": {"type": "plain_text", "text": "No"}, "value": "false"}, "options": [{"text": {"type": "plain_text", "text": "Yes"}, "value": "true"}, {"text": {"type": "plain_text", "text": "No"}, "value": "false"}]}, "label": {"type": "plain_text", "text": "Elasticsearch: Use Scroll API?"}},
        {"type": "input", "block_id": "elasticsearch_scroll_size_block", "optional": True, "element": {"type": "number_input", "is_decimal_allowed": False, "action_id": "elasticsearch_scroll_size_input"}, "label": {"type": "plain_text", "text": "Elasticsearch: Scroll Size"}},
    ]
    
    # --- Rebuild Blocks --- 
    prefix_blocks = []
    suffix_blocks = []
    source_type_found = False
    destination_found = False

    for block in all_current_blocks:
        block_id = block.get("block_id")

        if not source_type_found:
            prefix_blocks.append(block)
            if block_id == "source_type_block":
                source_type_found = True
        elif block_id == "destination_block":
            destination_found = True
            suffix_blocks.append(block)
        elif destination_found:
            # Append all subsequent blocks to the suffix
             suffix_blocks.append(block)
        # else: block is part of the old source-specific section or the placeholder context, so skip it

    if not source_type_found or not destination_found:
         logger.error("Failed to find source_type_block or destination_block to rebuild view.")
         # Attempt fallback using initial blocks structure? Or just fail.
         return 

    # Choose the specific blocks to insert
    specific_blocks_to_insert = []
    if selected_source_type == "sql":
        specific_blocks_to_insert = sql_blocks
    elif selected_source_type == "mongodb":
        specific_blocks_to_insert = mongo_blocks
    elif selected_source_type == "opensearch":
        specific_blocks_to_insert = opensearch_blocks
    elif selected_source_type == "elasticsearch":
        specific_blocks_to_insert = elasticsearch_blocks

    # Combine the blocks
    updated_blocks = prefix_blocks + specific_blocks_to_insert + suffix_blocks

    try:
        client.views_update(
            view_id=view_id,
            hash=hash_value, # Use the hash from the original view payload
            view={
                "type": "modal",
                "callback_id": "archive_request_submission", # Must match original
                "title": {"type": "plain_text", "text": "Archive Request Form"},
                "submit": {"type": "plain_text", "text": "Submit"},
                "close": {"type": "plain_text", "text": "Cancel"},
                "blocks": updated_blocks
            }
        )
        logger.info(f"Successfully updated view {view_id} for source type {selected_source_type}")
    except Exception as e:
        logger.error(f"Error updating archive request modal view {view_id}: {e}")

# --- Archive Request View Submission Handler ---
@app.view("archive_request_submission")
def handle_archive_request_submission(ack, body, client, view, logger):
    """Handles the submission of the archive request modal."""
    values = view["state"]["values"]
    requester_user_id = body["user"]["id"]

    # Helper to safely get values from modal state
    def get_value(block_id, action_id, value_type="value", is_selected_option=False, is_radio=False):
        try:
            element = values.get(block_id, {}).get(action_id, {})
            if not element: # Handle case where optional block might not exist if conditions aren't met
                return None
            if is_selected_option:
                selected_option = element.get("selected_option")
                return selected_option.get("value") if selected_option else None
            if is_radio:
                 selected_option = element.get("selected_option")
                 return selected_option.get("value") if selected_option else None
            # Handle number input specifically as value is string
            if element.get("type") == "number_input":
                 val = element.get("value")
                 return val if val else None # Return None if empty string
            return element.get(value_type)
        except Exception as e:
            logger.error(f"Error getting value for {block_id}/{action_id}: {e}")
            return None

    # Extract all values using the helper
    # Need to handle optional fields potentially not being in `values` if type wasn't selected/submitted
    source_name = get_value("source_name_block", "source_name_input")
    environment = get_value("environment_block", "environment_select", is_selected_option=True)
    source_type = get_value("source_type_block", "source_type_select", is_selected_option=True)
    destination = get_value("destination_block", "destination_select", is_selected_option=True)
    delete_after_archive = get_value("delete_after_archive_block", "delete_after_archive_radio", is_radio=True)

    sql_table = get_value("sql_table_block", "sql_table_input") if source_type == "sql" else None
    sql_primary_key = get_value("sql_primary_key_block", "sql_primary_key_input") if source_type == "sql" else None
    mongo_database = get_value("mongo_database_block", "mongo_database_input") if source_type == "mongodb" else None
    mongo_collection = get_value("mongo_collection_block", "mongo_collection_input") if source_type == "mongodb" else None
    opensearch_index = get_value("opensearch_index_block", "opensearch_index_input") if source_type == "opensearch" else None
    opensearch_use_scroll = get_value("opensearch_use_scroll_block", "opensearch_use_scroll_radio", is_radio=True) if source_type == "opensearch" else None
    opensearch_scroll_size_str = get_value("opensearch_scroll_size_block", "opensearch_scroll_size_input") if source_type == "opensearch" else None
    elasticsearch_index = get_value("elasticsearch_index_block", "elasticsearch_index_input") if source_type == "elasticsearch" else None
    elasticsearch_use_scroll = get_value("elasticsearch_use_scroll_block", "elasticsearch_use_scroll_radio", is_radio=True) if source_type == "elasticsearch" else None
    elasticsearch_scroll_size_str = get_value("elasticsearch_scroll_size_block", "elasticsearch_scroll_size_input") if source_type == "elasticsearch" else None

    filter_days_str = get_value("filter_days_block", "filter_days_input")
    filter_date_column = get_value("filter_date_column_block", "filter_date_column_input")
    group_by_date = get_value("group_by_date_block", "group_by_date_radio", is_radio=True)

    # --- Validation ---
    # (Validation logic remains largely the same, but needs to respect that source-specific fields might be None)
    errors = {}
    filter_days = None
    opensearch_scroll_size = None
    elasticsearch_scroll_size = None

    if not source_name: errors["source_name_block"] = "Source Name is required."
    if not environment: errors["environment_block"] = "Environment is required."
    if not source_type: errors["source_type_block"] = "Source Type is required."
    if not destination: errors["destination_block"] = "Destination is required."

    if not filter_days_str:
        errors["filter_days_block"] = "Filter days is required."
    else:
        try:
            filter_days = int(filter_days_str)
            if filter_days <= 0:
                 errors["filter_days_block"] = "Filter days must be a positive number."
        except ValueError:
            errors["filter_days_block"] = "Filter days must be a whole number."

    if not filter_date_column: errors["filter_date_column_block"] = "Filter Date Column/Field is required."

    # Type-specific validation (Only validate if the type matches)
    if source_type == "sql":
        if not sql_table: errors["sql_table_block"] = "SQL Table Name is required for type SQL."
        if not sql_primary_key: errors["sql_primary_key_block"] = "SQL Primary Key is required for type SQL."
    elif source_type == "mongodb":
        if not mongo_database: errors["mongo_database_block"] = "MongoDB Database Name is required for type MongoDB."
        if not mongo_collection: errors["mongo_collection_block"] = "MongoDB Collection Name is required for type MongoDB."
    elif source_type == "opensearch":
        if not opensearch_index: errors["opensearch_index_block"] = "OpenSearch Index Name is required for type OpenSearch."
        if opensearch_scroll_size_str:
            try:
                opensearch_scroll_size = int(opensearch_scroll_size_str)
                if opensearch_scroll_size <= 0:
                     errors["opensearch_scroll_size_block"] = "Scroll Size must be a positive number."
            except (ValueError, TypeError):
                errors["opensearch_scroll_size_block"] = "Scroll Size must be a whole number."
        if opensearch_use_scroll == "true" and opensearch_scroll_size is None:
             errors["opensearch_scroll_size_block"] = "Scroll Size is required if Use Scroll is enabled."
    elif source_type == "elasticsearch":
        if not elasticsearch_index: errors["elasticsearch_index_block"] = "Elasticsearch Index Name is required for type Elasticsearch."
        if elasticsearch_scroll_size_str:
            try:
                elasticsearch_scroll_size = int(elasticsearch_scroll_size_str)
                if elasticsearch_scroll_size <= 0:
                     errors["elasticsearch_scroll_size_block"] = "Scroll Size must be a positive number."
            except (ValueError, TypeError):
                errors["elasticsearch_scroll_size_block"] = "Scroll Size must be a whole number."
        if elasticsearch_use_scroll == "true" and elasticsearch_scroll_size is None:
             errors["elasticsearch_scroll_size_block"] = "Scroll Size is required if Use Scroll is enabled."

    if delete_after_archive is None: errors["delete_after_archive_block"] = "Please select Yes or No for Delete After Archive."

    if errors:
        ack(response_action="errors", errors=errors)
        logger.warning(f"Archive request validation failed for user {requester_user_id}: {errors}")
        return

    # Acknowledge the view submission successfully
    ack()

    # --- Format the message (using YAML structure) ---
    # (YAML formatting logic should already handle None values correctly based on previous changes)
    delete_after_archive_bool = delete_after_archive == "true"
    group_by_date_bool = group_by_date == "true" if group_by_date is not None else None
    opensearch_use_scroll_bool = opensearch_use_scroll == "true" if opensearch_use_scroll is not None else None
    elasticsearch_use_scroll_bool = elasticsearch_use_scroll == "true" if elasticsearch_use_scroll is not None else None

    config_lines = [
        f"*New Archive Request from <@{requester_user_id}>*",
        "```yaml",
        f"sources:",
        f"  - name: {source_name}",
        f"    environment: {environment}",
        f"    type: {source_type}",
        f"    # connection_string: Automatically determined by environment '{environment}'",
    ]

    # Add type-specific fields (Only add if type matches and value exists)
    if source_type == "sql":
        if sql_table: config_lines.append(f"    table: {sql_table}")
        if sql_primary_key: config_lines.append(f"    primary_key: {sql_primary_key}")
    elif source_type == "mongodb":
        if mongo_database: config_lines.append(f"    database: {mongo_database}")
        if mongo_collection: config_lines.append(f"    collection: {mongo_collection}")
    elif source_type == "opensearch":
        if opensearch_index: config_lines.append(f"    index: {opensearch_index}")
        if opensearch_use_scroll_bool is not None: config_lines.append(f"    use_scroll: {str(opensearch_use_scroll_bool).lower()}")
        if opensearch_scroll_size is not None: config_lines.append(f"    scroll_size: {opensearch_scroll_size}")
    elif source_type == "elasticsearch":
        if elasticsearch_index: config_lines.append(f"    index: {elasticsearch_index}")
        if elasticsearch_use_scroll_bool is not None: config_lines.append(f"    use_scroll: {str(elasticsearch_use_scroll_bool).lower()}")
        if elasticsearch_scroll_size is not None: config_lines.append(f"    scroll_size: {elasticsearch_scroll_size}")

    # Add common fields
    config_lines.extend([
        f"    filter:",
        f"      type: age",
        f"      days: {filter_days}",
        f"      date_field: {filter_date_column}",
        f"    delete_after_archive: {str(delete_after_archive_bool).lower()}",
        f"    destination: {destination}",
    ])
    # Add group_by_date if selected 'Yes'
    if group_by_date_bool is not None:
        config_lines.append(f"    group_by_date: {str(group_by_date_bool).lower()}")

    config_lines.append("```")
    message_text = "\n".join(config_lines)

    # --- Send DM to Admin --- 
    # (DM Sending Logic Remains the Same)
    try:
        dm_result = client.chat_postMessage(
            channel=ADMIN_USER_ID,
            text=message_text,
            mrkdwn=True # Ensure markdown formatting is applied
        )
        logger.info(f"Sent archive request DM (channel: {dm_result.get('channel')}) to {ADMIN_USER_ID} from {requester_user_id}")

        # Optionally send confirmation to the user who submitted
        client.chat_postMessage(
             channel=requester_user_id,
             text="Your archive request has been submitted for review."
        )

    except Exception as e:
        logger.error(f"Error sending DM to {ADMIN_USER_ID} or confirmation to {requester_user_id}: {e}")
        # Optionally inform the submitting user about the error
        try:
             client.chat_postMessage(
                 channel=requester_user_id,
                 text=f"Sorry, there was an error submitting your request. Please contact the administrator. Error: {e}"
             )
        except Exception as inner_e:
             logger.error(f"Failed to send error notification to user {requester_user_id}: {inner_e}")

if __name__ == "__main__":
    # Start the app
    if TEST_MODE:
        print("Test mode active - database is initialized.")
        print("Commands available:")
        for cmd in app.commands:
            print(f"  {cmd}")
        print("\nViews registered:")
        for view in app.views:
            print(f"  {view}")
    else:
        handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
        handler.start() 