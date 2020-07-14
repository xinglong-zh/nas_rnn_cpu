FROM python:3.7-slim
WORKDIR  /app
COPY . /app
RUN pip install --no-cache-dir -r  /app/requirements.txt
EXPOSE 19010
CMD ["python","/app/app.py"]








