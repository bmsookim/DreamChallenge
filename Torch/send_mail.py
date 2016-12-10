"""Send email via smtp_host."""
import smtplib
from email.mime.text import MIMEText
from email.header    import Header

def send(text):
    smtp_host = 'smtp.gmail.com'
    login, password = 'dmis.dreamchallenge','dmisinfos#1'
    recipients_emails = [
            'meliketoy@gmail.com',
            'hwejin23@gmail.com',
            'minhwan90@gmail.com']

    text = "In AWS Cluster~!\n\n" + text
    msg = MIMEText(text, 'plain', 'utf-8')
    msg['Subject'] = Header('Dream Challenge: HTCondor result (in AWS)', 'utf-8')
    msg['From'] = login
    msg['To'] = ", ".join(recipients_emails)

    s = smtplib.SMTP(smtp_host, 587, timeout=10)
    s.set_debuglevel(1)

    try:
        s.starttls()
        s.login(login, password)
        s.sendmail(msg['From'], recipients_emails, msg.as_string())
    finally:
        s.quit()

if __name__ == '__main__':
    send('Test!')
