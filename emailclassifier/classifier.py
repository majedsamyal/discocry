# emailclassifier/classifier.py
import base64
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return build('gmail', 'v1', credentials=creds)

def get_emails_for_training(max_results=1000, query=""):
    service = authenticate_gmail()
    
    # Get all message IDs first
    results = service.users().messages().list(
        userId='me', maxResults=max_results, q=query
    ).execute()
    
    messages = results.get('messages', [])
    emails = []
    
    # Process in batches of 100 (Gmail API limit)
    batch_size = 100
    
    for i in range(0, len(messages), batch_size):
        batch_messages = messages[i:i + batch_size]
        batch = service.new_batch_http_request()
        
        def extract_email_data(request_id, response, exception):
            if exception:
                return
                
            payload = response.get('payload', {})
            headers = payload.get('headers', [])
            
            subject = sender = body = ""
            for header in headers:
                name = header['name'].lower()
                if name == 'subject':
                    subject = header['value']
                elif name == 'from':
                    sender = header['value']
            
            body = extract_body(payload)
            
            emails.append({
                'subject': subject,
                'sender': sender,
                'body': body,
                'labels': response.get('labelIds', [])
            })
        
        # Add requests to batch (max 100)
        for msg in batch_messages:
            batch.add(
                service.users().messages().get(userId='me', id=msg['id']),
                callback=extract_email_data
            )
        
        # Execute this batch
        batch.execute()
        print(f"Processed {len(batch_messages)} emails...")
    
    return emails

def extract_body(payload):
    """Extract email body text"""
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain':
                data = part['body']['data']
                return base64.urlsafe_b64decode(data).decode('utf-8')
    elif payload['mimeType'] == 'text/plain':
        data = payload['body']['data']
        return base64.urlsafe_b64decode(data).decode('utf-8')
    return ""
