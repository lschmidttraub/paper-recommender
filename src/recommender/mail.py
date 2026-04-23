from __future__ import annotations

import smtplib
from email.message import EmailMessage

import markdown as md_lib


def send(
    *,
    subject: str,
    markdown_body: str,
    to_addr: str,
    from_addr: str,
    smtp_password: str,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 465,
) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(markdown_body)
    html = md_lib.markdown(markdown_body, extensions=["extra"])
    msg.add_alternative(html, subtype="html")

    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        server.login(from_addr, smtp_password)
        server.send_message(msg)
