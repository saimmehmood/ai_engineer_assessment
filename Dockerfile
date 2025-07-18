FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=application.py
ENV FLASK_CONFIG=DEV

# Run the application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]