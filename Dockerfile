# Use the official Python image as base image
FROM python:3

# Set the working directory in the container
WORKDIR /app

# Copy the local main.py file to the container
COPY main.py .

# Run the Python script when the container launches
CMD ["python", "main.py"]
