import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import email
from bs4 import BeautifulSoup
import nltk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import os
import json

# Download necessary NLTK data (if needed)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def get_gmail_service():
    """Authenticates with the Gmail API and returns a service object."""
    creds = None
    # The file token.json stores the user's access and refresh tokens,
    # and is created automatically when the authorization flow completes
    # for the first time.
    if os.path.exists('token.json'):
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/gmail.modify'])
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(google.auth.transport.requests.Request())
            except Exception as e:
                print(f"Error refreshing credentials: {e}")
                # Handle error appropriately, perhaps by re-authenticating
                creds = None # Force re-authentication
        else:
            from google_auth_oauthlib.flow import InstalledAppFlow
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', ['https://www.googleapis.com/auth/gmail.modify'])
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        if creds:
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
    if not creds:
        raise Exception("Failed to obtain credentials")

    try:
        service = build('gmail', 'v1', credentials=creds)
        return service
    except Exception as e:
        print(f"Error building Gmail service: {e}")
        return None

def get_emails(service, query='in:inbox is:unread'):
    """Retrieves emails from Gmail based on a query."""
    try:
        results = service.users().messages().list(userId='me', q=query).execute()
        messages = results.get('messages', [])
        if not messages:
            print('No emails found.')
            return []
        return messages
    except HttpError as error:
        print(f'An error occurred: {error}')
        return []

def get_email_content(service, msg_id):
    """Retrieves the content of a specific email."""
    try:
        message = service.users().messages().get(userId='me', id=msg_id, format='raw').execute()
        msg_bytes = base64.urlsafe_b64decode(message['raw'].encode('ASCII'))
        msg = email.message_from_bytes(msg_bytes)
        return msg
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None
import base64

def extract_email_features(email_message):
    """Extracts features from an email message."""
    features = {}
    features['subject'] = email_message.get('subject', '')
    features['sender'] = email.utils.parseaddr(email_message.get('from', ''))[1]
    # Get the date
    date_str = email_message.get('date')
    if date_str:
        try:
            features['date'] = email.utils.parsedate_to_datetime(date_str)
        except TypeError:
            features['date'] = None

    body_text = ""
    if email_message.is_multipart():
        for part in email_message.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get('Content-Disposition'))

            if content_type == 'text/plain' and 'attachment' not in content_disposition:
                try:
                    body_text += part.get_payload(decode=True).decode()
                except:
                    body_text += part.get_payload(decode=True).decode('utf-8', errors='ignore')

            elif content_type == 'text/html' and 'attachment' not in content_disposition:
                try:
                    html = part.get_payload(decode=True).decode()
                    soup = BeautifulSoup(html, 'html.parser')
                    body_text += soup.get_text()
                except:
                    html = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    soup = BeautifulSoup(html, 'html.parser')
                    body_text += soup.get_text(errors='ignore')
    else:
        try:
            body_text = email_message.get_payload(decode=True).decode()
        except:
            body_text = email_message.get_payload(decode=True).decode('utf-8', errors='ignore')
    features['body'] = body_text

    features['links'] = [url for _, url in BeautifulSoup(body_text, 'html.parser').find_all('a', href=True)]
    return features


def preprocess_text(text):
    """Preprocesses text (e.g., removes HTML, punctuation)."""
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = text.replace('\r', '').replace('\n', ' ')  # Remove newlines and carriage returns
    text = ' '.join(text.split())  # Remove extra spaces
    return text

def get_keywords(text):
    """Extracts keywords from text using NLTK."""
    text = preprocess_text(text)
    tokens = nltk.word_tokenize(text)
    # Remove stop words and non-alphanumeric tokens
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    keywords = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    return keywords

def vectorize_features(features_list, vectorizer=None):
    """Converts email features to numerical vectors (e.g., using TF-IDF)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if vectorizer is None:
        # Combine subject and body for vectorization
        text_data = [f"{email['subject']} {email['body']}" for email in features_list]
        vectorizer = TfidfVectorizer(max_features=5000)  # Limit features for performance
        feature_vectors = vectorizer.fit_transform(text_data)
        return feature_vectors, vectorizer
    else:
        text_data = [f"{email['subject']} {email['body']}" for email in features_list]
        feature_vectors = vectorizer.transform(text_data)
        return feature_vectors, vectorizer

def train_model(feature_vectors, labels):
    """Trains a machine learning model to classify emails."""
    model = MultinomialNB()
    model.fit(feature_vectors, labels)
    return model

def classify_emails(model, feature_vectors):
    """Classifies emails using the trained model."""
    predictions = model.predict(feature_vectors)
    return predictions

def get_email_labels(service, email_ids):
    """
    Retrieves labels for given email IDs from Gmail.
    Args:
        service: The Gmail API service object.
        email_ids: A list of email IDs.
    Returns:
        A list of labels corresponding to the email IDs.  Returns "UNLABELED" if cannot get.
    """
    labels = []
    for email_id in email_ids:
        try:
            message = service.users().messages().get(userId='me', id=email_id, format='metadata', metadataHeaders=['labelIds']).execute()
            label_ids = message.get('labelIds', [])
            if 'IMPORTANT' in label_ids:
                labels.append('important')
            elif 'CATEGORY_PROMOTIONS' in label_ids or 'CATEGORY_UPDATES' in label_ids:
                labels.append('unimportant')
            else:
                labels.append('other')
        except HttpError as e:
            print(f"Error getting labels for email {email_id}: {e}")
            labels.append('UNLABELED')  # Handle the error, you might want to log it
    return labels
def perform_email_actions(service, email_ids, predictions):
    """Performs actions on emails based on their classifications."""
    for email_id, prediction in zip(email_ids, predictions):
        try:
            if prediction == 'important':
                service.users().messages().modify(userId='me', id=email_id, body={'addLabelIds': ['IMPORTANT']}).execute()
                print(f"Marked email {email_id} as important.")
            elif prediction == 'unimportant':
                service.users().messages().modify(userId='me', id=email_id, body={'removeLabelIds': ['UNREAD'], 'addLabelIds': ['READ']}).execute()
                print(f"Marked email {email_id} as read.")
            else:  # prediction == 'other'
                service.users().messages().modify(userId='me', id=email_id, body={'removeLabelIds': ['UNREAD'], 'addLabelIds': ['READ']}).execute()
                print(f"Marked email {email_id} as read.")
        except HttpError as error:
            print(f'An error occurred: {error}')

def main():
    """Main function to orchestrate the email processing."""
    service = get_gmail_service()
    if not service:
        print("Failed to get Gmail service. Exiting.")
        return

    emails = get_emails(service, query='in:inbox is:unread')
    if not emails:
        print("No unread emails found. Exiting.")
        return

    email_data = []
    email_ids = []
    for email_message in emails:
        email_ids.append(email_message['id'])
        msg = get_email_content(service, email_message['id'])
        if msg:
            features = extract_email_features(msg)
            email_data.append(features)
        else:
            print(f"Skipping email {email_message['id']} due to error retrieving content.")
            email_ids.remove(email_message['id']) #remove the email_id if you cannot get the message

    if not email_data:
        print("No email data to process. Exiting.")
        return

    labels = get_email_labels(service, email_ids)
    if not labels:
        print("No labels found. Exiting.")
        return

    X_train, X_test, y_train, y_test = train_test_split(email_data, labels, test_size=0.2, random_state=42)
    try:
        feature_vectors, vectorizer = vectorize_features(X_train)
        model = train_model(feature_vectors, y_train)

        test_vectors = vectorizer.transform(X_test)
        predictions = classify_emails(model, test_vectors)
        print(classification_report(y_test, predictions))

        # Classify all emails, not just the test set, for action
        all_vectors = vectorizer.transform(email_data)
        predictions = classify_emails(model, all_vectors)
        perform_email_actions(service, email_ids, predictions)
    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
