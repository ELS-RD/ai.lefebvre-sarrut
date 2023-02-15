<img width="75%" src="/assets/ai-lefebvre-sarrut-logo-light.svg#gh-light-mode-only" alt="Lefebvre Sarrut's AI blog">
<img width="75%" src="/assets/ai-lefebvre-sarrut-logo-dark.svg#gh-dark-mode-only" alt="Lefebvre Sarrut's AI blog">

## Material for MkDocs Insiders version

To avoid any breaking change, we target a specific version.

For the preview ([`Dockerfile`](Dockerfile)),
and in the Github Action workflow ([`deploy-static-site.yml`](.github/workflows/deploy-static-site.yml))

### How to upgrade version

To take advantage of the latest features, check
the [online changelog](https://squidfunk.github.io/mkdocs-material/insiders/changelog/).

According to it, **synchronize the fork**, **update the docker image** and **the workflow**.

If necessary, take into account the breaking change by
consulting [How to upgrade](https://squidfunk.github.io/mkdocs-material/upgrade/).

## Preview the static site locally

The easiest and least intrusive way is to use docker.

### Important❗

You **need to be logged** in to github to pull the image (
see [docker login](https://docs.docker.com/engine/reference/commandline/login/)).

### About the **`CI`'s Environment variable**

An environment variable `CI` is set to `true` when deploying continuously to Github pages. This variable allows a number
of plugins to be run.

If you want the same behavior when running locally, you can set this variable to `true` by adding the `-e CI=true`
option to the `docker run` command.

> _The execution of the optimization plugin is also conditioned by another environment variable: `CI_OPTIMIZATION`. You
can also run it locally, but this requires the installation of third party libraries (see documentation)._

```shell
# Building a Material for MkDocs docker image with the non built-in plugins
docker build -t mkdocs-material-insiders-plugins -f Dockerfile docs
```

```shell
# Previewing the site in watch mode
docker run --rm -it -p 8000:8000 -v ${PWD}:/docs mkdocs-material-insiders-plugins

# Previewing the site in watch mode with plugins
docker run --rm -it -p 8000:8000 -e CI=true -v ${PWD}:/docs mkdocs-material-insiders-plugins
```

```shell
# Build the static site
docker run --rm -it -v ${PWD}:/docs mkdocs-material-insiders-plugins build

# Build the static site with plugins
docker run --rm -it -e CI=true -v ${PWD}:/docs mkdocs-material-insiders-plugins build
```

## Writing article

### Recommendations

#### Metadata

The metadata must have at least:

- a date
- an author or authors
- a category or categories
- a tag or tags

Please refer to [the documentation](https://squidfunk.github.io/mkdocs-material/setup/setting-up-a-blog/#usage).

Note that you can use the `draft` flag. If it is set to `true`, the article is visible in preview but will not be
published online.

Also note, if you are a recurring author, that you can add yourself to the file at [`.authors.yml`](docs/.authors.yml)
to centralize your information.
See [documentation](https://squidfunk.github.io/mkdocs-material/setup/setting-up-a-blog/#adding-authors).

Example:

```yaml
---
draft: false
date: 2023-01-01
authors:
  - mbenesty
categories:
  - Justice
tags:
  - Justice
  - Data Science
  - Technology
  - Programming
  - Machine Learning
---
```

#### Categories and tags

If you want to add categories or tags, first add them to the closed list in the file [`mkdocs.yml`](mkdocs.yml).

See:

```yaml
plugins:
  - blog:
    categories_allowed:

  - tags:
    tags_allowed:
```

#### Adding an excerpt

Don't forget to add the separator `<!-- more -->` to delimit the visible extract in the article index (
see [documentation](https://squidfunk.github.io/mkdocs-material/setup/setting-up-a-blog/#adding-an-excerpt)).

#### Management of related resources

##### Add an image

To add an image, use the `<figure>` tag.
Don't forget to add the property `loading=lazy`.

```markdown
<figure markdown>
  ![Alternative](path_to_image){ width="100%", loading=lazy }
  <figcaption>Text</figcaption>
</figure>
```

`<figcaption>` is optional. If it contains a link, it must be tagged with an `<a>` tag.
The `target` and `rel` attributes are automatically set according to the `href`.

```markdown
<figure markdown>
  ![Alternative](image.png){ width="100%", loading=lazy }
  <figcaption>Text with a <a href="https://www.domain.com/">link</a></figcaption>
</figure>
```

## Errors

It is possible that during the build, you have an error like :
`Blog post 'path/your-article.md' has no date set.`

This is most likely due to badly formatted or missing metadata.  
