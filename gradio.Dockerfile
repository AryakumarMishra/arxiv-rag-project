# Start from a Python base image
FROM python:3.11-slim

# Install system-level dependencies and build tools
# 'build-essential' provides the 'cc' compiler and other tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the launcher script and the entire 'src' directory
COPY gradio_launcher.py .
COPY src/ ./src/

# Expose the port Gradio will run on
EXPOSE 7861

# The command to start the Gradio app
CMD ["python", "gradio_launcher.py"]