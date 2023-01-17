# Lefebvre Sarrut AI Blog

## Preview the static site locally

The easiest and least intrusive way is to use docker.

```shell
# Pull the docker image from the private registry (you must have the rights)
docker pull ghcr.io/els-rd/mkdocs-material-insiders
```

```shell
# Previewing the site in watch mode
docker run --rm -it -p 8000:8000 -v ${PWD}:/docs ghcr.io/els-rd/mkdocs-material-insiders
```

```shell
# Build the static site
docker run --rm -it -v ${PWD}:/docs ghcr.io/els-rd/mkdocs-material-insiders build
```