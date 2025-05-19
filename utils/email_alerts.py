import smtplib
from email.mime.text import MIMEText
from config import Config

class EmailAlerts:
    def __init__(self):
        self.sender = Config.EMAIL_SENDER
        self.password = Config.EMAIL_PASSWORD
        self.recipient = Config.EMAIL_RECIPIENT
    
    def send_alert(self, subject, body):
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.sender
        msg['To'] = self.recipient
        
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(self.sender, self.password)
                smtp.send_message(msg)
            print("\nAlert email sent successfully")
        except Exception as e:
            print(f"\nError sending email alert: {e}")