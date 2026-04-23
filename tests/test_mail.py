from unittest.mock import MagicMock

from recommender.mail import send


def test_send_calls_smtp_ssl_with_auth_and_multipart(mocker):
    smtp_class = mocker.patch("recommender.mail.smtplib.SMTP_SSL")
    server = MagicMock()
    smtp_class.return_value.__enter__.return_value = server

    send(
        subject="Test subject",
        markdown_body="# Heading\n\nBody text.",
        to_addr="to@example.com",
        from_addr="from@example.com",
        smtp_password="app-password",
        smtp_host="smtp.gmail.com",
        smtp_port=465,
    )

    smtp_class.assert_called_once_with("smtp.gmail.com", 465)
    server.login.assert_called_once_with("from@example.com", "app-password")
    server.send_message.assert_called_once()
    sent_msg = server.send_message.call_args[0][0]
    assert sent_msg["Subject"] == "Test subject"
    assert sent_msg["From"] == "from@example.com"
    assert sent_msg["To"] == "to@example.com"
    # multipart: must contain both plain and html parts
    parts = list(sent_msg.iter_parts()) if sent_msg.is_multipart() else [sent_msg]
    content_types = {p.get_content_type() for p in parts}
    assert "text/plain" in content_types
    assert "text/html" in content_types
