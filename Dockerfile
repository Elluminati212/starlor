# FROM python:3.12.2-alpine3.19
# RUN mkdir -p /code
# RUN addgroup python && adduser -D -G python python
# RUN chown -R python:python /code
# WORKDIR /code
# USER root
# RUN pip install pipenv
# USER python
# COPY --chown=python . .
# RUN pipenv install
# CMD ["pipenv", "run", "python", "main.py"]

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose a port (optional, if your app uses a port)
EXPOSE 80

# Define environment variable (optional, if needed)
ENV NAME World

# Run main.py when the container launches
CMD ["python", "main.py"]