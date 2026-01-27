##/home/vedant/Mental Health/login_page/authplayground/users/db.py
## Tell Django how to reach MongoDB ##
# users/db.py
from mongoengine import connect

connect(
    db="auth_db",
    host="localhost",
    port=27017,
    alias="default"
)


### This line is the handshake:
###“Hey MongoDB, here’s Django. Let’s talk.” ###