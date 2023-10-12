# chat2db 




A chatbot can respond question based on a postgresql vector db using Llama 2 + Langchain + VectorDB (pgvector or chroma)

## 1. Setup Chatbot with Chroma as vector DB 

run `ingest/chroma/ingest_chroma.py` to ingest all .pdf, .md, .txt files into the chroma vector DB.

run `app/chat_chroma.py` to start the chatbot, then just go to the link in the terminal/console to chat with the bot.

## 2. Setup Chatbot with pgvector as vector DB (docker) 

cd into `ingest/chroma` folder, run docker-compose file to setup postgres DB with pgvector extension (This will also run the script in `init.sql`)
```shell
sudo docker-compose up -d
```
run `ingest/pgvector/ingest_chroma.py` to ingest all .pdf, .md, .txt files into the pgvector DB.

run `app/chat_pgvector.py` to start the chatbot, then just go to the link in the terminal/console to chat with the bot.


## 3. Setup PostgreSQL Locally (optional)

**Note**: You don't need to do this if you just setup pgvector with docker like the previous section

### 3.1 Install PostgreSQL 15.2 +
**SKIP** *if you already have postgresql 15.2 and above (15.2 + is required for pgvector extension)*

PostgreSQL 15 package is not available in the default package repository, so enable its official package repository using following commands.
```shell
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget -qO- https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo tee /etc/apt/trusted.gpg.d/pgdg.asc &>/dev/null
```
Fetch the latest versions of the packages. 
```shell
sudo apt update
```
Let’s install the PostgreSQL client and server using the below apt command:
```shell
sudo apt install postgresql postgresql-client -y
```

**OPTIONAL - Enable, Start, Restart, Check Status**
Enable the service so it starts on boot:
```shell
sudo systemctl start postgresql@15-main.service
```
Start the service now without rebooting:
```shell
sudo systemctl start postgresql@15-main.service
```
Restart the service now without rebooting:
```shell
sudo systemctl restart postgresql@15-main.service
```
Verify that the PostgreSQL service is up and running (Press "Q" or "q" to exit)
```shell
sudo systemctl status postgresql@15-main.service
```
Check the psql version
```shell
psql --version
```
reference: https://www.linuxtechi.com/how-to-install-postgresql-on-ubuntu/#google_vignette

### 3.2 Create New Database

Access the PostgreSQL database shell as the default superuser 'postgres'
```shell
sudo su - postgres
psql
```
You will then see this : `postgres=#`

Creating new server and user
```sql
postgres=# create user User_Name with superuser password 'User_Password';
```

Exit psql
```sql
postgres=# \q
```

Exit user postgres.
```shell
exit
```

Log in with the default superuser 'postgres' and create a database. I use 'vectordb' as the name of the DB.
```shell
sudo -u postgres createdb vectordb
```
Verify if your database is created and working. 1. Connect to the PostgreSQL server. 2. List all databases
```sql
postgres=# sudo -u postgres psql
postgres=# \l
```

Reference: https://stackoverflow.com/questions/53267642/create-new-local-server-in-pgadmin

### 3.3 Install pgvector
PostgreSQL's extensions often require the PostgreSQL development package which contains the necessary header files (like postgres.h). You can install it with:
```shell
sudo apt install postgresql-server-dev-15
```
Note: Replace `15` with your Postgres server version

Install gcc
```shell
sudo apt-get install gcc
```
Set the Correct Compiler:
```shell
make clean  # Clean any previous build attempts.
make CC=gcc
```
Set PostgreSQL Configuration Path (you may not need this step)
```shell
export PG_CONFIG=/usr/bin/pg_config
```

Install pgvector
```shell
cd /tmp
git clone --branch v0.5.0 https://github.com/pgvector/pgvector.git
cd pgvector
make clean
make
make install # may need sudo
```

Connect to PostgreSQL server using psql client:
- specifying the user (-U sqladmin), the host (-h localhost)
- asking it to prompt for the password (-W)
- connecting to the postgres database (-d vectordb)
```shell
psql -U sqladmin -h localhost -W -d vectordb
```

Run this for each database you are using for storing vectors
```sql
postgres=# CREATE EXTENSION vector;
```

### 3.4 (OPTIONAL) Install pgAdmin
**SKIP** *if you already have a postgresql client with virtual UI*

Install the public key for the repository (if not done previously):
```shell
curl -fsS https://www.pgadmin.org/static/packages_pgadmin_org.pub | sudo gpg --dearmor -o /usr/share/keyrings/packages-pgadmin-org.gpg
```
Create the repository configuration file:
```shell
sudo sh -c 'echo "deb [signed-by=/usr/share/keyrings/packages-pgadmin-org.gpg] https://ftp.postgresql.org/pub/pgadmin/pgadmin4/apt/$(lsb_release -cs) pgadmin4 main" > /etc/apt/sources.list.d/pgadmin4.list && apt update'
```
Install for both desktop and web modes:
```shell
sudo apt update
sudo apt install pgadmin4
```