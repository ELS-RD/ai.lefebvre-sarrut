# Lefebvre Sarrut AI Blog

## Preview the static site locally

The easiest and least intrusive way is to use docker.

```shell
# Building a Material for MkDocs docker image with the non built-in plugins
docker build -t mkdocs-material-insiders-plugins -f Dockerfile docs
```

```shell
# Previewing the site in watch mode
docker run --rm -it -p 8000:8000 -v ${PWD}:/docs mkdocs-material-insiders-plugins
```

```shell
# Build the static site
docker run --rm -it -v ${PWD}:/docs mkdocs-material-insiders-plugins build
```

## Ajouter une catégorie ou un tag

todo