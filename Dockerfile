FROM continuumio/anaconda3:4.4.0
COPY . /app/

WORKDIR /app
RUN pip install -r requirements.txt
# Expose port 
ENV PORT 5000
CMD ["gunicorn", "app:app"]