FROM public.ecr.aws/docker/library/python:3.13-slim

RUN mkdir -p /app
COPY ./main.py /app/
COPY model_new/ /app/model_new/
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
WORKDIR /app
EXPOSE 8080
CMD [ "main.py" ]
ENTRYPOINT [ "python" ]