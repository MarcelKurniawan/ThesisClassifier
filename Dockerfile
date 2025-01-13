# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY . /app/

# Expose port 5000 to the outside world
EXPOSE 5000

# Install PostgreSQL client
RUN apt-get update && apt-get install -y postgresql-client

# Initialize the database
RUN service postgresql start && psql -U postgres -f thesis_db.sql

# Define environment variable
ENV FLASK_APP=app.py

# Run the application
CMD ["flask", "run", "--host=0.0.0.0"]
