FROM python:3.12-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /usr/src/app/

# Install curl and other dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# install poetry and generate requirements.txt
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.8.1 python3 -
ENV PATH="${PATH}:/root/.local/bin"
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=0 \
    POETRY_NO_ANSI=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache


# copying project into python image
COPY . .

RUN poetry install && rm -rf $POETRY_CACHE_DIR

USER root