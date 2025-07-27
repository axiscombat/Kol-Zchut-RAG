# Use an official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy code into the image
COPY ./src /app/src
# COPY ./src/kol_zchut_rag_storage_1000_v3 /app/src/kol_zchut_rag_storage_1000_v3
COPY requirements.txt /app/

# Install dependencies
RUN pip install -r requirements.txt

# Set the default command to run your query
CMD ["python", "src/query.py"]
