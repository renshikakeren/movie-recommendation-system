# Use a pre-built PyTorch image that includes all necessary data science libraries
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install the remaining dependencies
COPY requirements.txt .

# Install the remaining, smaller dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Command to run the application when the container starts
CMD uvicorn main:app --host 0.0.0.0 --port 8000
