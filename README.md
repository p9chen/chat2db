# chat2db 

A chatbot can respond question based on a postgresql vector db using Llama 2 7B + Langchain + Postgresql + pgvector 

## Install PostgreSQL 15.2 +
**SKIP** *if you already have postgresql 15.2 and above (15.2 + is required for pgvector extension)*

PostgreSQL 15 package is not available in the default package repository, so enable its official package repository using following commands.
```
$ sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
$ wget -qO- https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo tee /etc/apt/trusted.gpg.d/pgdg.asc &>/dev/null
```
Fetch the latest versions of the packages. 
```
$ sudo apt update
```
Let’s install the PostgreSQL client and server using the below apt command:
```
$ sudo apt install postgresql postgresql-client -y
```
verify that the PostgreSQL service is up and running
```
$ sudo systemctl status postgresql
```
check the PostgreSQL version
```
$ psql --version
```

reference: https://www.linuxtechi.com/how-to-install-postgresql-on-ubuntu/#google_vignette
