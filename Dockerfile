FROM python:3.8.10-slim

RUN pip install pipenv

WORKDIR /app                                                                

COPY [".", "./"]

RUN pipenv install --deploy --system

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]