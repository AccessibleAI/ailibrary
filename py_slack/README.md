
#  Send a custom message to a Slack channel

A component that levrages Slack's incoming webhook app to send a custom message to a slack channel.

## How It Works

Using Slack's built-in incoming webhook capabilities to send a message:
https://slack.dev/python-slack-sdk/webhook/index.html

## Running

**In Slack:**

1. Go here to create a [new slack app](https://api.slack.com/apps). 
Select **Create New App** -> **From Scratch**.
Choose a **Name** & **Workspace**.

2. Install app to workspace & choose the target slack channel.
From app page, select **Settings** -> **Install App**

3. Copy the **webhook URL** from the bottom of the page.

**In cnvrg:**

4. Define a new secret in your cnvrg project:
Go to **Settings** -> **Secrets** -> **Add**. Set the secret's name to be ```SLACK_WEBHOOK_URL```, and the value to be the URL.

5. Choose the Slack AI Library component, and pass the text you'd wish to send, to the **message** argument.
                                     
## Demo Inputs

Send a custom text message:
```
--message "this is my message"
```

Send the value of an environemnt variable:
```
--message $SOME_ENV_VAR
```

Send the value of a local variable within your code runtime:
```
--message local_variable
```

Send the value of a previous task's tag (read more about flow tags [here](https://app.cnvrg.io/docs/core_concepts/flows.html#tags-parameters-flow)):
```
--message {{training.length}}
```

## Example Demo from command line:
You can use the following command to try the demo:
```
python3 slack.py --message "my bonnie is over the ocean, my bonnie is over the sea"        
```

## Customizing the Slack app appearance
You can customize the "bot user" that will send the messages in your webhook app, by visiting the Slack app page and going to **Settings** -> **Basic Information** -> then scroll to bottom for **Display Information**
