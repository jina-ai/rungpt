FROM jinaai/jina:3.18.0-py310-standard

COPY . /workspace/
WORKDIR /workspace

RUN curl -sSL https://install.python-poetry.org | python3 -
RUN poetry config virtualenvs.create false
RUN poetry install --only main


RUN echo "\
!Gateway\n\
py_modules:\n\
  - open_gpt.serve.gateway\n\
with:\n\
  cors: False\n\
" > config.yml


ENTRYPOINT ["jina", "gateway", "--uses", "config.yml"]