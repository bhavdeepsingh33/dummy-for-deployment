FROM continuumio/anaconda3:4.4.0
WORKDIR /app

# Expose port
EXPOSE 8080

RUN pip install -r requirements.txt
COPY . /app/
CMD ["gunicorn", "app:app"]