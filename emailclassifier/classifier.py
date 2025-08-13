# emailclassifier/classifier.py
import base64
import os
import pandas as pd
from datetime import datetime
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

import time

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

import time
from googleapiclient.errors import HttpError

def get_emails_for_training(target_emails=1000, query="after:2025/01/01"):
   service = authenticate_gmail()
   
   messages = []
   page_token = None
   
   while len(messages) < target_emails * 1.5:
       results = service.users().messages().list(
           userId='me', 
           maxResults=500,
           q=query,
           pageToken=page_token
       ).execute()
       
       batch = results.get('messages', [])
       if not batch:
           break
           
       messages.extend(batch)
       page_token = results.get('nextPageToken')
       
       if not page_token:
           break
   
   print(f"Total message IDs fetched: {len(messages)}")
   
   emails = []
   failed_ids = []
   batch_size = 25
   delay = 0.5
   
   i = 0
   while len(emails) < target_emails and i < len(messages):
       batch_end = min(i + batch_size, len(messages))
       batch_messages = messages[i:batch_end]
       
       batch = service.new_batch_http_request()
       batch_results = []
       
       def extract_email_data(request_id, response, exception):
           if exception:
               if 'rateLimitExceeded' in str(exception):
                   failed_ids.append(batch_messages[int(request_id)]['id'])
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
           
           batch_results.append({
               'subject': subject,
               'sender': sender,
               'body': body,
               'labels': response.get('labelIds', [])
           })
       
       for idx, msg in enumerate(batch_messages):
           batch.add(
               service.users().messages().get(userId='me', id=msg['id']),
               callback=extract_email_data,
               request_id=str(idx)
           )
       
       try:
           batch.execute()
           emails.extend(batch_results)
           
           if batch_results:
               delay = max(0.5, delay * 0.9)
           
           print(f"Extracted {len(emails)} emails so far...")
           
       except HttpError as e:
           if 'rateLimitExceeded' in str(e):
               print(f"Rate limited. Waiting {delay:.1f}s...")
               time.sleep(delay)
               delay = min(delay * 2, 30)
               continue
       
       i += batch_size
       time.sleep(delay)
       
       if len(emails) >= target_emails:
           emails = emails[:target_emails]
           break
   
   if len(emails) < target_emails and failed_ids:
       print(f"Retrying {len(failed_ids)} failed emails...")
       time.sleep(5)
       
       for msg_id in failed_ids[:target_emails - len(emails)]:
           try:
               msg = service.users().messages().get(userId='me', id=msg_id).execute()
               time.sleep(0.5)
           except:
               pass
   
   print(f"\nSuccessfully extracted {len(emails)} emails")
   
   df = pd.DataFrame(emails)
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   filename = f"data/raw/2023/emails_{timestamp}.csv"
   
   os.makedirs("data/raw/2023", exist_ok=True)
   df.to_csv(filename, index=False)
   print(f"Saved to {filename}")
   
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
