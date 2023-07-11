from argparse import ArgumentParser, SUPPRESS
from slack_sdk.webhook import WebhookClient
import os

def slack(webhook_url,message):
    job_url = os.getenv("CNVRG_JOB_URL")
    job_name = os.getenv("CNVRG_JOB_NAME")
    webhook = WebhookClient(webhook_url)
    response = webhook.send(
    text="fallback",
    blocks=[
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*:small_blue_diamond:<{job_url}|Sent from cnvrg.io job: {job_name}>:small_blue_diamond:*\n{message}"
                }
         }
        ]
    )

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    # TODO --- ADD your parameters
    parser.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    parser.add_argument('-m', '--message', action='store', help="the string that will be sent as message to the slack channel", required=True, type=str)
    parser.add_argument('--project_dir', action='store', dest='project_dir', help="""--- For inner use of cnvrg.io ---""")
    parser.add_argument('--output_dir', action='store', dest='output_dir', help="""--- For inner use of cnvrg.io ---""")

    args = parser.parse_args()
    print("these are the arguments sent:")
    print(args)
    #TODO --- execut your AI Component code
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    slack(webhook_url,args.message)
