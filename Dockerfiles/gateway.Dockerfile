FROM jinaai/jina:3.18.0-py310-standard

COPY . /run_gpt/
WORKDIR /run_gpt

RUN python3 -m pip install -e .


RUN echo "\
!Gateway\n\
py_modules:\n\
  - run_gpt.serve.gateway\n\
" > config.yml


ENTRYPOINT ["jina", "gateway", "--uses", "config.yml"]