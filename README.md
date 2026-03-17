# rag-anything


## Development


### Pre-requisites
* Install Poetry
```
curl -sSL https://install.python-poetry.org | python3 -
```

 * Create .env from .env.sample and populate 
 the values

* Run the dev depencencies
```
docker compose -f docker-compose.dev.yml -d
```


### Running App
```
poetry run uvicorn src.api.main:app --reload
```

### Running tests
```
poetry run pytest tests
```