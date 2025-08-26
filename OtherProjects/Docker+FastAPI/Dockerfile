# Use Python 3.12.4 base image
FROM python:3.12.4-slim

# Set working directory to where main.py is located
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project into the container
COPY . .

# # Set working directory to where main.py is located
WORKDIR /app/main-app/FastAPI

# Expose the port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]