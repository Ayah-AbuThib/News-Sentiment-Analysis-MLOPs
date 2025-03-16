# Step 1: Use a Python base image with the necessary version
FROM python:3.8-slim

# Step 2: Set environment variables for non-interactive installation
ENV PYTHONUNBUFFERED=1

# Step 3: Set the working directory inside the container
WORKDIR /app

# Step 4: Copy the requirements.txt file to the working directory
COPY requirements.txt /app/

# Step 5: Install Python dependencies (including Streamlit)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Step 6: Copy the entire application into the container
COPY . /app/

# Step 7: Expose port 8501 for the Streamlit app (default port)
EXPOSE 8501

# Step 8: Command to run the Streamlit app
CMD ["streamlit", "run", "src/api/streamlit_ui.py"]
