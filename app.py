import os
from datetime import datetime
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import database as db
from dateutil import parser
import logging
import re

# Set up logging to debug the issue
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check if we're in test mode
TEST_MODE = os.environ.get('TEST_MODE', 'False').lower() == 'true'

channel_name = os.environ.get('SLACK_CHANNEL')


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