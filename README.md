# Lefebvre Sarrut AI Blog

## Material for MkDocs Insiders version

To avoid any breaking change, we target a specific version.

For the preview (`Dockerfile`),
and in the Github Action workflow (`.github/workflows/deploy-static-site.yml`)

### How to upgrade version

To take advantage of the latest features, check the [online changelog](https://squidfunk.github.io/mkdocs-material/insiders/changelog/).

According to it, **synchronize the fork**, **update the docker image** and **the workflow**.

If necessary, take into account the breaking change by consulting [How to upgrade](https://squidfunk.github.io/mkdocs-material/upgrade/).

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